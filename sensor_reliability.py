"""
Sensor Reliability Intelligence Engine
Smart Water Quality Monitoring System
Tagline: "Monitoring Today, Protecting Tomorrow."

Checks:
1. Missing readings
2. Out-of-physical bounds / invalid values
3. Repeated identical values (flatline / sensor freeze)
4. Sudden implausible jumps (rate-of-change spike)
5. Excessive noise / variance instability
6. Sensor drift relative to user calibration standards
7. Sensor last-seen freshness / latency
"""

from typing import Dict, List, Any, Optional
import time
import math
import numpy as np

# Plausible maximum jumps per single reading interval (physical dynamics limit)
MAX_PLAUSIBLE_JUMP: Dict[str, float] = {
    "ph": 1.8,          # pH cannot plausibly jump > 1.8 units in seconds under continuous flow
    "turbidity": 40.0,  # Turbidity sudden spike threshold
    "tds": 250.0,       # TDS ppm jump threshold
    "conductivity": 350.0, # EC µS/cm jump threshold
    "temperature": 5.0  # Water temperature thermal inertia prevents >5°C jumps in seconds
}

# Max allowed identical readings before flagging flatline (frozen probe or ADC lock)
MAX_REPEATED_IDENTICAL = 6

# Excessive noise variance threshold (sample standard deviation in rolling window)
MAX_STABLE_VARIANCE: Dict[str, float] = {
    "ph": 0.35,
    "turbidity": 8.0,
    "tds": 45.0,
    "conductivity": 60.0,
    "temperature": 1.2
}


def check_sensor_health(
    param: str,
    recent_readings: List[float],
    last_seen_epoch: Optional[float] = None,
    calibration_reference: Optional[float] = None,
    now_epoch: Optional[float] = None
) -> Dict[str, Any]:
    """
    Evaluates individual sensor operational reliability across 7 operational diagnostics.
    Returns:
      health_status: 'Healthy' | 'Warning' | 'Needs Inspection'
      score: 0 - 100
      issues: list of detected anomalies
      corrective_action: recommended maintenance step
    """
    if now_epoch is None:
        now_epoch = time.time()

    issues: List[str] = []
    corrective_actions: List[str] = []
    score = 100

    # 1. Check Data Freshness / Stale readings
    if last_seen_epoch is not None:
        latency_sec = now_epoch - last_seen_epoch
        if latency_sec > 120:
            score -= 40
            issues.append(f"Telemetry stale: no new payload for {int(latency_sec)}s")
            corrective_actions.append("Inspect microcontroller power, Wi-Fi connectivity, or MQTT broker status.")
        elif latency_sec > 30:
            score -= 15
            issues.append(f"High telemetry latency ({int(latency_sec)}s)")

    # 2. Check for missing or empty history
    if not recent_readings:
        return {
            "sensor": param,
            "status": "Needs Inspection",
            "score": 0,
            "issues": ["No telemetry data received for this sensor."],
            "corrective_action": "Verify wiring pinout (Analog/I2C), power supply (3.3V/5V), and ADC connection.",
            "is_reliable": False
        }

    latest = recent_readings[-1]

    # 3. Invalid value / Physical bounds check
    if latest is None or math.isnan(latest) or math.isinf(latest):
        score -= 60
        issues.append("Received NaN / null invalid numeric reading.")
        corrective_actions.append("Check sensor probe connection; probe may be disconnected or short-circuited.")
    else:
        # Check against absolute bounds
        from water_quality import DEFAULT_REFERENCE_RANGES, validate_reading
        valid, err = validate_reading(param, latest, DEFAULT_REFERENCE_RANGES)
        if not valid:
            score -= 50
            issues.append(f"Physical bounds violation: {err}")
            corrective_actions.append("Verify analog voltage-divider ratio, sensor gain calibration, or probe contamination.")

    # 4. Repeated identical readings (Flatline freeze check)
    if len(recent_readings) >= MAX_REPEATED_IDENTICAL:
        tail = recent_readings[-MAX_REPEATED_IDENTICAL:]
        # Check if all values in the tail are bitwise identical or within float epsilon
        if all(abs(x - tail[0]) < 1e-4 for x in tail):
            score -= 35
            issues.append(f"Frozen output: {MAX_REPEATED_IDENTICAL} consecutive identical readings ({tail[0]}).")
            corrective_actions.append("Inspect ADC conversion register and probe for biological fouling or hardware latch-up.")

    # 5. Sudden implausible jump (Rate-of-change check)
    if len(recent_readings) >= 2 and latest is not None and recent_readings[-2] is not None:
        delta = abs(latest - recent_readings[-2])
        jump_limit = MAX_PLAUSIBLE_JUMP.get(param, 50.0)
        if delta > jump_limit:
            score -= 30
            issues.append(f"Implausible jump: changed by {delta:.2f} within one cycle (limit: {jump_limit}).")
            corrective_actions.append("Check for loose signal wiring, electromagnetic interference, or power ripple.")

    # 6. Excessive noise / variance instability
    if len(recent_readings) >= 5:
        valid_vals = [x for x in recent_readings[-10:] if x is not None and not math.isnan(x)]
        if len(valid_vals) >= 5:
            std_dev = float(np.std(valid_vals))
            max_std = MAX_STABLE_VARIANCE.get(param, 10.0)
            if std_dev > max_std:
                score -= 25
                issues.append(f"Excessive noise: standard deviation {std_dev:.2f} exceeds threshold ({max_std}).")
                corrective_actions.append("Add hardware RC low-pass filter, enable digital averaging filter, or verify ground shielding.")

    # 7. Potential Sensor Drift against user calibration reference
    if calibration_reference is not None and latest is not None:
        drift_delta = abs(latest - calibration_reference)
        # Drift limits
        drift_limit = {
            "ph": 0.4,
            "turbidity": 5.0,
            "tds": 50.0,
            "conductivity": 75.0,
            "temperature": 1.5
        }.get(param, 2.0)

        if drift_delta > drift_limit:
            score -= 20
            issues.append(f"Calibration drift: measured {latest:.2f} vs reference standard {calibration_reference:.2f} (offset: {drift_delta:.2f}).")
            corrective_actions.append(f"Perform two-point calibration using certified reference standard buffers.")

    # Bound score
    score = max(0, min(100, score))

    if score >= 80:
        status = "Healthy"
    elif score >= 50:
        status = "Warning"
    else:
        status = "Needs Inspection"

    if not corrective_actions:
        corrective_actions.append("Sensor is operating normally within expected electro-chemical parameters. Routine inspection recommended.")

    return {
        "sensor": param,
        "status": status,
        "score": score,
        "latest_value": latest,
        "issues": issues,
        "corrective_action": " ".join(corrective_actions),
        "is_reliable": status != "Needs Inspection",
        "disclaimer": (
            "NOTICE: Automated diagnostics identify operational anomalies (noise, flatline, jump) "
            "but cannot conclusively prove absolute metrological accuracy without calibrated reference standards."
        )
    }


def assess_all_sensors(
    data_history: List[Dict[str, Any]],
    calibration_refs: Optional[Dict[str, float]] = None,
    last_seen_times: Optional[Dict[str, float]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Runs reliability diagnostics on all five water quality sensors.
    """
    from water_quality import DEFAULT_REFERENCE_RANGES
    params = list(DEFAULT_REFERENCE_RANGES.keys())
    now_epoch = time.time()
    results = {}

    for p in params:
        series = [row[p] for row in data_history if p in row] if data_history else []
        last_seen = last_seen_times.get(p, now_epoch) if last_seen_times else now_epoch
        ref = calibration_refs.get(p) if calibration_refs else None
        results[p] = check_sensor_health(p, series, last_seen, ref, now_epoch)

    return results
