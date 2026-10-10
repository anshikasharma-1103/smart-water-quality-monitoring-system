"""
Water Quality Logic, Reference Ranges, and Validation
Smart Water Quality Monitoring System
Tagline: "Monitoring Today, Protecting Tomorrow."
"""

from typing import Dict, Any, Tuple, Optional
import math

# Reference standards (Default aligned with WHO / BIS IS 10500 / EPA guidelines for potable water screening)
DEFAULT_REFERENCE_RANGES: Dict[str, Dict[str, Any]] = {
    "ph": {
        "name": "pH",
        "unit": "pH",
        "min_safe": 6.5,
        "max_safe": 8.5,
        "min_caution": 6.0,
        "max_caution": 9.0,
        "physical_min": 0.0,
        "physical_max": 14.0,
        "weight": 0.25,
        "ideal": 7.0,
        "description": "Measures hydrogen-ion activity indicating water acidity or basicity.",
        "significance": "Extreme pH alters taste, accelerates piping corrosion, and impacts disinfectant efficacy."
    },
    "turbidity": {
        "name": "Turbidity",
        "unit": "NTU",
        "min_safe": 0.0,
        "max_safe": 5.0,
        "min_caution": 0.0,
        "max_caution": 10.0,
        "physical_min": 0.0,
        "physical_max": 1000.0,
        "weight": 0.25,
        "ideal": 0.5,
        "description": "Measures water cloudiness or haziness caused by suspended colloids and particulates.",
        "significance": "High turbidity shields waterborne pathogens from UV and chemical disinfection."
    },
    "tds": {
        "name": "Total Dissolved Solids (TDS)",
        "unit": "ppm",
        "min_safe": 50.0,
        "max_safe": 500.0,
        "min_caution": 30.0,
        "max_caution": 900.0,
        "physical_min": 0.0,
        "physical_max": 5000.0,
        "weight": 0.20,
        "ideal": 150.0,
        "description": "Concentration of inorganic salts and small amounts of organic matter dissolved in water.",
        "significance": "Elevated TDS causes mineral deposits, bitter salty taste, and gastrointestinal irritation."
    },
    "conductivity": {
        "name": "Electrical Conductivity (EC)",
        "unit": "µS/cm",
        "min_safe": 100.0,
        "max_safe": 800.0,
        "min_caution": 50.0,
        "max_caution": 1200.0,
        "physical_min": 0.0,
        "physical_max": 10000.0,
        "weight": 0.15,
        "ideal": 300.0,
        "description": "Ability of water to conduct electrical current, directly proportional to dissolved ionic species.",
        "significance": "Rapid surges indicate agricultural runoff, industrial effluent, or seawater intrusion."
    },
    "temperature": {
        "name": "Water Temperature",
        "unit": "°C",
        "min_safe": 10.0,
        "max_safe": 28.0,
        "min_caution": 5.0,
        "max_caution": 35.0,
        "physical_min": -5.0,
        "physical_max": 100.0,
        "weight": 0.15,
        "ideal": 20.0,
        "description": "Thermal condition affecting dissolved oxygen saturation, biological activity, and chemical kinetics.",
        "significance": "Warmer water promotes bacterial proliferation and accelerates microbial biofouling."
    }
}

LEGAL_DISCLAIMER = (
    "PRELIMINARY SCREENING NOTICE: This prototype provides physical sensor-based screening indicators only. "
    "Sensor readings alone do not constitute proof of drinking-water potability. Formal certification requires "
    "laboratory testing for microbiological pathogens (e.g. E. coli), toxic heavy metals (lead, arsenic), "
    "and synthetic organic compounds pursuant to WHO / EPA / BIS IS 10500 standards."
)


def validate_reading(param: str, value: Any, ranges: Optional[Dict[str, Dict[str, Any]]] = None) -> Tuple[bool, Optional[str]]:
    """
    Validates a parameter reading for non-null, numeric type, and physically plausible bounds.
    """
    if ranges is None:
        ranges = DEFAULT_REFERENCE_RANGES

    if param not in ranges:
        return False, f"Unknown parameter '{param}'"

    if value is None:
        return False, f"{param.upper()}: Missing reading (None/null)"

    try:
        val = float(value)
    except (ValueError, TypeError):
        return False, f"{param.upper()}: Non-numeric value received ('{value}')"

    if math.isnan(val) or math.isinf(val):
        return False, f"{param.upper()}: Invalid numeric value (NaN or Inf)"

    p_min = ranges[param]["physical_min"]
    p_max = ranges[param]["physical_max"]
    if val < p_min or val > p_max:
        return False, f"{param.upper()}: Value {val} outside physical bounds [{p_min}, {p_max}]"

    return True, None


def evaluate_parameter_status(param: str, value: float, ranges: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Evaluates individual parameter value against safe and caution reference thresholds.
    Returns status: 'Optimal', 'Safe', 'Caution', 'Unsafe' or 'Invalid'.
    """
    if ranges is None:
        ranges = DEFAULT_REFERENCE_RANGES

    cfg = ranges.get(param)
    if not cfg:
        return {"status": "Unknown", "message": "Unknown parameter", "color": "gray"}

    val_ok, err = validate_reading(param, value, ranges)
    if not val_ok:
        return {"status": "Invalid", "message": err, "color": "red"}

    min_s, max_s = cfg["min_safe"], cfg["max_safe"]
    min_c, max_c = cfg["min_caution"], cfg["max_caution"]

    if min_s <= value <= max_s:
        return {
            "status": "Safe",
            "message": f"Within reference range ({min_s} - {max_s} {cfg['unit']})",
            "color": "green"
        }
    elif min_c <= value <= max_c:
        direction = "low" if value < min_s else "high"
        return {
            "status": "Caution",
            "message": f"Marginally {direction} ({value:.2f} {cfg['unit']})",
            "color": "yellow"
        }
    else:
        direction = "critically low" if value < min_c else "critically high"
        return {
            "status": "Unsafe",
            "message": f"Significantly {direction} ({value:.2f} {cfg['unit']})",
            "color": "red"
        }


def calculate_water_quality_index(readings: Dict[str, float], ranges: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Computes a weighted Water Quality Index (WQI) scaled from 0 (heavily degraded) to 100 (excellent).
    """
    if ranges is None:
        ranges = DEFAULT_REFERENCE_RANGES

    sub_indices = {}
    total_weight = 0.0
    weighted_sum = 0.0
    abnormal_count = 0
    reasons = []

    for param, cfg in ranges.items():
        if param not in readings or readings[param] is None:
            continue

        val = float(readings[param])
        weight = cfg["weight"]
        total_weight += weight

        # Evaluate deviation from ideal
        ideal = cfg["ideal"]
        max_s = cfg["max_safe"]
        min_s = cfg["min_safe"]

        status = evaluate_parameter_status(param, val, ranges)
        if status["status"] in ("Caution", "Unsafe"):
            abnormal_count += 1
            reasons.append(f"{cfg['name']}: {status['message']}")

        # Sub-index formula: 100 when ideal, linear drops beyond safe limits
        if min_s <= val <= max_s:
            sub = 100.0 - 20.0 * (abs(val - ideal) / max((max_s - min_s), 1e-5))
        elif cfg["min_caution"] <= val <= cfg["max_caution"]:
            sub = 60.0 - 20.0 * (abs(val - ideal) / max((cfg["max_caution"] - cfg["min_caution"]), 1e-5))
        else:
            sub = 25.0

        sub = max(0.0, min(100.0, sub))
        sub_indices[param] = sub
        weighted_sum += sub * weight

    if total_weight > 0:
        overall_wqi = round(weighted_sum / total_weight, 1)
    else:
        overall_wqi = 0.0

    if overall_wqi >= 80 and abnormal_count == 0:
        overall_status = "Safe-range indication"
        status_color = "emerald"
    elif overall_wqi >= 60 and abnormal_count <= 2:
        overall_status = "Caution"
        status_color = "amber"
    else:
        overall_status = "Unsafe"
        status_color = "red"

    return {
        "wqi": overall_wqi,
        "overall_status": overall_status,
        "status_color": status_color,
        "sub_indices": sub_indices,
        "abnormal_count": abnormal_count,
        "reasons": reasons,
        "disclaimer": LEGAL_DISCLAIMER
    }
