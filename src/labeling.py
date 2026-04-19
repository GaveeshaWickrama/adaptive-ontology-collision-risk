def is_valid_precollision_row(ttc_value, drac_value, ttc_max, drac_max):
    return (ttc_value <= ttc_max) and (drac_value <= drac_max)