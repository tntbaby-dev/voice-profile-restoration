def clamp(x, min_val=-6.0, max_val=6.0):
    return max(min(x, max_val), min_val)

def map_to_eq_settings(imbalance: dict) -> dict:
    eq = {}

    # nonlinear scaling (more natural sound)
    def scale(x):
        return clamp(x * 10)
    
    eq["low_shelf"] = scale(imbalance["low"])
    eq["low_mid"] = scale(imbalance["low_mid"])
    eq["mid"] = scale(imbalance["mid"])
    eq["presence"] = scale(imbalance["presence"])
    eq["high_shelf"] = scale(imbalance["air"])

    return eq
