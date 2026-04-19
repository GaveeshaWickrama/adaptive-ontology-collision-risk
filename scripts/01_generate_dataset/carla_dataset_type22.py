#!/usr/bin/env python3
# CARLA >= 0.9.14
import os, csv, math, time, random, argparse
from typing import Tuple, List, Optional
from contextlib import contextmanager
import carla

# ------------------------------- small math helpers -------------------------------
def clamp(v, a, b): return max(a, min(b, v))
def vec2(x, y): return carla.Vector3D(x, y, 0.0)

def dot2(a: carla.Vector3D, b: carla.Vector3D) -> float:
    return a.x * b.x + a.y * b.y

def norm2(a: carla.Vector3D) -> float:
    return math.sqrt(a.x * a.x + a.y * a.y)

def unit2(a: carla.Vector3D) -> carla.Vector3D:
    n = norm2(a) or 1.0
    return vec2(a.x / n, a.y / n)

def speed_mps(v: carla.Vector3D) -> float:
    return (v.x * v.x + v.y * v.y + v.z * v.z) ** 0.5

def yaw_unit(yaw_deg: float) -> carla.Vector3D:
    r = math.radians(yaw_deg)
    return vec2(math.cos(r), math.sin(r))

def time_to_point(loc: carla.Location, vel: carla.Vector3D, yaw_deg: float, point: carla.Location):
    """Return (tti_sec, dist_along_heading_m). inf if not moving toward point."""
    dir_u = yaw_unit(yaw_deg)
    rel = vec2(point.x - loc.x, point.y - loc.y)
    along = dot2(rel, dir_u)
    v_al = dot2(vec2(vel.x, vel.y), dir_u)
    if v_al <= 0.05:
        return float("inf"), max(along, 0.0)
    return along / v_al, max(along, 0.0)

# ------------------------------- sync & weather -------------------------------
@contextmanager
def sync_world(client: carla.Client, fixed_dt=0.05):
    world = client.get_world()
    original = world.get_settings()
    new = carla.WorldSettings(
        synchronous_mode=True,
        no_rendering_mode=getattr(original, "no_rendering_mode", False),
        fixed_delta_seconds=float(fixed_dt),
        substepping=getattr(original, "substepping", True),
        max_substep_delta_time=getattr(original, "max_substep_delta_time", 0.01),
        max_substeps=getattr(original, "max_substeps", 10),
        max_culling_distance=getattr(original, "max_culling_distance", 0.0),
        deterministic_ragdolls=getattr(original, "deterministic_ragdolls", False),
        tile_stream_distance=getattr(original, "tile_stream_distance", 3000.0),
        actor_active_distance=getattr(original, "actor_active_distance", 2000.0),
        spectator_as_ego=getattr(original, "spectator_as_ego", True),
    )
    world.apply_settings(new)
    try:
        yield world
    finally:
        world.apply_settings(original)

def set_weather(world: carla.World, scheme: str):
    s = scheme.lower()
    if s == "clear":
        W = carla.WeatherParameters.ClearNoon
    elif s == "rain":
        W = carla.WeatherParameters.ClearNoon
        W.precipitation = 70.0
        W.precipitation_deposits = 70.0
        W.cloudiness = 80.0
        W.wetness = 70.0
    elif s == "fog":
        W = carla.WeatherParameters.ClearNoon
        W.fog_density = 60.0
        W.fog_distance = 20.0
        W.cloudiness = 30.0
    elif s == "night":
        W = carla.WeatherParameters.ClearNight
    else:
        W = carla.WeatherParameters.ClearNoon

    if s == "rain":
        W.precipitation = 70.0
        W.precipitation_deposits = 70.0
        W.cloudiness = 80.0
        W.wetness = 70.0
    if s == "fog":
        W.fog_density = 60.0
        W.fog_distance = 20.0
        W.cloudiness = 30.0

    world.set_weather(W)

# ------------------------------- junction selection -------------------------------
def choose_perp_junction(world: carla.World) -> Tuple[
    Tuple[carla.Waypoint, carla.Waypoint, carla.Waypoint],
    Tuple[carla.Waypoint, carla.Waypoint, carla.Waypoint],
    carla.Location
]:
    amap = world.get_map()
    wps = amap.generate_waypoints(3.0)

    def angle_deg(a: carla.Vector3D, b: carla.Vector3D) -> float:
        dot = max(-1.0, min(1.0, a.x * b.x + a.y * b.y))
        return math.degrees(math.acos(dot))

    jset = {}
    for wp in wps:
        if wp.is_junction:
            j = wp.get_junction()
            jset[j.id] = j

    for j in jset.values():
        center = j.bounding_box.location
        entries: List[carla.Waypoint] = []
        try:
            if hasattr(j, "get_entries"):
                entries = list(j.get_entries())  # type: ignore
        except Exception:
            entries = []
        if not entries:
            try:
                pairs = j.get_waypoints(carla.LaneType.Driving)
                entries = [p[0] for p in pairs if p and p[0] is not None]
            except Exception:
                entries = []

        if len(entries) < 2:
            continue

        best, best_err = None, 1e9
        for i in range(len(entries)):
            for k in range(i + 1, len(entries)):
                a, b = entries[i], entries[k]
                fa, fb = a.transform.get_forward_vector(), b.transform.get_forward_vector()
                na = unit2(vec2(fa.x, fa.y))
                nb = unit2(vec2(fb.x, fb.y))
                ang = angle_deg(na, nb)
                err = abs(ang - 90.0)
                if err < best_err:
                    best, best_err = (a, b), err
        if not best or best_err >= 25.0:
            continue

        a_entry, b_entry = best

        def back(wp: carla.Waypoint, dist=25.0) -> Optional[carla.Waypoint]:
            cur, traveled = wp, 0.0
            while cur and traveled < dist:
                prev = cur.previous(2.0)
                if not prev:
                    return cur
                cur = prev[0]
                traveled += 2.0
            return cur

        def fwd(wp: carla.Waypoint, dist=35.0) -> Optional[carla.Waypoint]:
            cur, traveled = wp, 0.0
            while cur and traveled < dist:
                nxt = cur.next(2.0)
                if not nxt:
                    return cur
                cur = nxt[0]
                traveled += 2.0
            return cur

        a0, b0 = back(a_entry, 25.0), back(b_entry, 25.0)
        a_exit, b_exit = fwd(a_entry, 35.0), fwd(b_entry, 35.0)
        if all([a0, a_entry, a_exit, b0, b_entry, b_exit]):
            return (a0, a_entry, a_exit), (b0, b_entry, b_exit), center

    raise RuntimeError("No perpendicular 4-way junction found. Try Town03 or Town10HD_Opt.")

# ------------------------------- spawn & control -------------------------------
def wp_transform_raised(wp: carla.Waypoint, z_raise: float = 0.8) -> carla.Transform:
    tr = carla.Transform(wp.transform.location, wp.transform.rotation)
    tr.location.z += z_raise
    return tr

def pick_any_road_car(lib: carla.BlueprintLibrary) -> carla.ActorBlueprint:
    cand = []
    for q in [
        "vehicle.tesla.model3", "vehicle.audi.tt", "vehicle.lincoln*",
        "vehicle.nissan.micra", "vehicle.toyota.*", "vehicle.bmw.*",
        "vehicle.mini.cooper*", "vehicle.*"
    ]:
        items = list(lib.filter(q))
        cand.extend(items)
    cand = [
        bp for bp in cand
        if bp.has_attribute("number_of_wheels")
        and bp.get_attribute("number_of_wheels").as_int() == 4
    ]
    if not cand:
        cand = list(lib.filter("vehicle.*"))
    return random.choice(cand)

def try_spawn(world: carla.World, bp: carla.ActorBlueprint, tr: carla.Transform) -> Optional[carla.Actor]:
    try:
        a = world.try_spawn_actor(bp, tr)
        return a
    except Exception:
        return None

def attach_collision_sensor(world: carla.World, parent: carla.Actor, cb):
    bp = world.get_blueprint_library().find("sensor.other.collision")
    s = world.spawn_actor(bp, carla.Transform(), attach_to=parent)
    s.listen(cb)
    return s

# ------------------------------- episode runner -------------------------------
def run_episode(world: carla.World,
                tm,
                center: carla.Location,
                a0: carla.Waypoint, a_entry: carla.Waypoint,
                b0: carla.Waypoint, b_entry: carla.Waypoint,
                frames_per_scenario: int,
                writer,
                weather_name: str,
                fixed_dt: float = 0.05) -> Tuple[int, bool]:
    """
    Returns: (frames_written, collision_happened)
    """
    lib = world.get_blueprint_library()
    ego_bp = pick_any_road_car(lib)
    other_bp = pick_any_road_car(lib)

    ego = try_spawn(world, ego_bp, wp_transform_raised(a0))
    other = try_spawn(world, other_bp, wp_transform_raised(b0))
    if not (ego and other):
        if ego:
            ego.destroy()
        if other:
            other.destroy()
        return 0, False

    ego.set_autopilot(False)
    other.set_autopilot(False)

    try:
        tm.ignore_lights_percentage(ego, 100)
        tm.ignore_lights_percentage(other, 100)
    except Exception:
        pass

    col_flag = {"hit": False}
    sensor = attach_collision_sensor(world, ego, lambda e: col_flag.__setitem__("hit", True))

    ego_speed = random.uniform(7.5, 9.0)
    other_speed = random.uniform(7.5, 9.0)

    delay_other_ticks = random.randint(0, 1)

    frames = 0
    try:
        tick = 0
        while frames < frames_per_scenario:
            et = ego.get_transform()
            ot = other.get_transform()
            ef = et.get_forward_vector()
            of = ot.get_forward_vector()

            ego.set_target_velocity(carla.Vector3D(ef.x * ego_speed, ef.y * ego_speed, 0.0))

            if tick >= delay_other_ticks:
                other.set_target_velocity(carla.Vector3D(of.x * other_speed, of.y * other_speed, 0.0))
            else:
                other.set_target_velocity(carla.Vector3D(0, 0, 0))

            world.tick()
            tick += 1

            ts = world.get_snapshot().timestamp
            et = ego.get_transform()
            ev = ego.get_velocity()
            ot = other.get_transform()
            ov = other.get_velocity()
            epos, opos = et.location, ot.location
            es, os = speed_mps(ev), speed_mps(ov)

            d_vec = vec2(opos.x - epos.x, opos.y - epos.y)
            dist_between = norm2(d_vec)
            ego_tti, ego_along = time_to_point(epos, ev, et.rotation.yaw, center)
            oth_tti, oth_along = time_to_point(opos, ov, ot.rotation.yaw, center)

            line_u = unit2(d_vec) if dist_between > 0.05 else vec2(0.0, 0.0)
            rel_v = vec2(ov.x - ev.x, ov.y - ev.y)
            rel_speed_towards = -dot2(rel_v, line_u)
            ttc = (dist_between / rel_speed_towards) if rel_speed_towards > 0.05 else float("inf")

            cflag = int(col_flag["hit"])

            writer.writerow([
                ts.frame, f"{ts.elapsed_seconds:.3f}", world.get_map().name, weather_name,
                ego.type_id, other.type_id,
                ego.id, f"{epos.x:.3f}", f"{epos.y:.3f}", f"{et.rotation.yaw:.1f}", f"{es:.3f}",
                other.id, f"{opos.x:.3f}", f"{opos.y:.3f}", f"{ot.rotation.yaw:.1f}", f"{os:.3f}",
                f"{dist_between:.3f}",
                f"{center.x:.3f}", f"{center.y:.3f}",
                f"{ego_along:.3f}", f"{ego_tti:.3f}" if math.isfinite(ego_tti) else "inf",
                f"{oth_along:.3f}", f"{oth_tti:.3f}" if math.isfinite(oth_tti) else "inf",
                f"{rel_speed_towards:.3f}", f"{ttc:.3f}" if math.isfinite(ttc) else "inf",
                cflag
            ])
            frames += 1

    finally:
        try:
            sensor.stop()
            sensor.destroy()
        except:
            pass
        try:
            ego.destroy()
        except:
            pass
        try:
            other.destroy()
        except:
            pass

    return frames, bool(col_flag["hit"])

# ------------------------------- main orchestrator -------------------------------
def main():
    ap = argparse.ArgumentParser("Type-22 dataset generator (1000 crash scenarios, 1000 frames each)")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--map", type=str, default="Town10HD_Opt")
    ap.add_argument("--weather-mode", type=str, default="per_scenario", choices=["per_scenario", "fixed"])
    ap.add_argument("--weather", type=str, default="clear", choices=["clear", "rain", "fog", "night"])
    ap.add_argument("--scenarios", type=int, default=1000, help="Total number of scenarios (episodes) to generate")
    ap.add_argument("--frames-per-scenario", type=int, default=1000, help="Frames to record per scenario")
    ap.add_argument("--fixed-dt", type=float, default=0.05)
    ap.add_argument("--out", type=str, default="type22_dataset.csv")
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    world = client.get_world()
    if args.map and args.map not in world.get_map().name:
        world = client.load_world(args.map)
        world.wait_for_tick()

    with sync_world(client, fixed_dt=args.fixed_dt) as world:
        tm = client.get_trafficmanager(args.tm_port)
        tm.set_synchronous_mode(True)

        (a0, a_entry, _), (b0, b_entry, _), center = choose_perp_junction(world)

        need_header = not os.path.exists(args.out)
        f = open(args.out, "a", newline="")
        w = csv.writer(f)
        if need_header:
            w.writerow([
                "frame", "time", "map", "weather",
                "ego_type", "other_type",
                "ego_id", "ego_x", "ego_y", "ego_yaw_deg", "ego_speed_mps",
                "other_id", "other_x", "other_y", "other_yaw_deg", "other_speed_mps",
                "dist_between_m",
                "conflict_x", "conflict_y",
                "ego_dist_to_conflict_m", "ego_tti_s",
                "other_dist_to_conflict_m", "other_tti_s",
                "rel_speed_towards_mps", "ttc_s",
                "collision_flag"
            ])

        collision_scenarios = 0
        weather_options = ["clear", "rain", "fog", "night"]

        try:
            for i in range(args.scenarios):
                scn = i + 1

                if args.weather_mode == "fixed":
                    scenario_weather = args.weather
                else:
                    scenario_weather = random.choice(weather_options)

                set_weather(world, scenario_weather)

                attempts = 0
                while True:
                    attempts += 1
                    frames_written, hit = run_episode(
                        world, tm, center,
                        a0, a_entry, b0, b_entry,
                        frames_per_scenario=args.frames_per_scenario,
                        writer=w,
                        weather_name=scenario_weather,
                        fixed_dt=args.fixed_dt
                    )

                    if frames_written == 0:
                        continue

                    if hit:
                        collision_scenarios += 1
                        print(
                            f"[SCN {scn:04d}] weather={scenario_weather} "
                            f"frames={frames_written} | COLLISION ✅ (attempt {attempts}) "
                            f"| collision_scenarios={collision_scenarios}"
                        )
                        break

                    time.sleep(0.03)

                time.sleep(0.05)

        finally:
            f.close()

        print("\nDone.")
        print(f"CSV: {args.out}")
        print(f"Scenarios: {args.scenarios}, collision scenarios: {collision_scenarios}")

if __name__ == "__main__":
    main()