import math
from pathlib import Path
import pandas as pd

INPUT_CSV = "data/processed/type22_final_labeled_dataset.csv"
OUTPUT_TTL = "outputs/metricsexp3_instances.ttl"
BASE_IRI = "http://example.org/avrisk#"

REQUIRED = [
    "frame", "time", "scenario_id", "weather", "ego_type", "other_type",
    "ego_tti_s", "other_tti_s", "TTC_conflict", "DRAC"
]

def sanitize(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out).strip("_")
    while "__" in s:
        s = s.replace("__", "_")
    return s or "x"

def weather_lighting(raw_weather: str):
    w = str(raw_weather).strip().lower()
    if w == "night":
        return ("clearWeather", "nightLighting")
    if w == "rain":
        return ("rainyWeather", "dayLighting")
    if w == "fog":
        return ("foggyWeather", "dayLighting")
    return ("clearWeather", "dayLighting")

def main():
    df = pd.read_csv(INPUT_CSV)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    keep = (
        df["ego_tti_s"].notna() &
        df["other_tti_s"].notna() &
        df["TTC_conflict"].notna() &
        df["DRAC"].notna() &
        (df["TTC_conflict"] > 0) &
        (df["ego_tti_s"] >= 0) &
        (df["other_tti_s"] >= 0)
    )
    df = df.loc[keep].copy()

    lines = []
    lines.append('@prefix : <http://example.org/avrisk#> .')
    lines.append('@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .')
    lines.append('')

    for row in df.itertuples(index=False):
        fid = int(getattr(row, "frame"))
        sid = str(getattr(row, "scenario_id"))

        eid = f"enc_{sanitize(sid)}_{fid}"
        env_id = f"env_{sanitize(sid)}_{fid}"
        ci_id = f"ci_{sanitize(sid)}_{fid}"
        ego_id = f"ego_{sanitize(sid)}_{fid}"
        other_id = f"other_{sanitize(sid)}_{fid}"

        ego_tti = float(getattr(row, "ego_tti_s"))
        other_tti = float(getattr(row, "other_tti_s"))
        max_tti = max(ego_tti, other_tti)
        sync_diff = abs(ego_tti - other_tti)
        ttc = float(getattr(row, "TTC_conflict"))
        drac = float(getattr(row, "DRAC"))

        weather_ind, lighting_ind = weather_lighting(getattr(row, "weather"))

        lines.append(f":{eid} a :Encounter ;")
        lines.append(f'  :frameId "{fid}"^^xsd:int ;')
        lines.append(f'  :timestampS "{float(getattr(row, "time"))}"^^xsd:double ;')
        lines.append(f'  :scenarioId "{sid}"^^xsd:string ;')
        lines.append(f"  :hasEnvironment :{env_id} ;")
        lines.append(f"  :hasConflictInteraction :{ci_id} ;")
        lines.append(f"  :hasEgoVehicle :{ego_id} ;")
        lines.append(f"  :hasOtherVehicle :{other_id} ;")
        lines.append("  :hasRiskLevel :safeRisk .")
        lines.append("")

        lines.append(f":{env_id} a :EnvironmentContext ;")
        lines.append(f"  :hasWeatherCondition :{weather_ind} ;")
        lines.append(f"  :hasLightingCondition :{lighting_ind} .")
        lines.append("")

        lines.append(f":{ci_id} a :ConflictInteraction ;")
        lines.append(f'  :egoTTI "{ego_tti}"^^xsd:double ;')
        lines.append(f'  :otherTTI "{other_tti}"^^xsd:double ;')
        lines.append(f'  :maxTTI "{max_tti}"^^xsd:double ;')
        lines.append(f'  :syncTTIDiff "{sync_diff}"^^xsd:double ;')
        lines.append(f'  :ttcConflict "{ttc}"^^xsd:double ;')
        lines.append(f'  :dracValue "{drac}"^^xsd:double .')
        lines.append("")

        lines.append(f":{ego_id} a :EgoVehicle ;")
        lines.append(f'  :egoVehicleType "{str(getattr(row, "ego_type"))}"^^xsd:string .')
        lines.append("")

        lines.append(f":{other_id} a :OtherVehicle ;")
        lines.append(f'  :otherVehicleType "{str(getattr(row, "other_type"))}"^^xsd:string .')
        lines.append("")

    Path(OUTPUT_TTL).parent.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_TTL).write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {OUTPUT_TTL} with {len(df):,} encounter individuals.")

if __name__ == "__main__":
    main()