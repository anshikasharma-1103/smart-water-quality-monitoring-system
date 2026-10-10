"""
Explainable Alert Engine
Smart Water Quality Monitoring System
Tagline: "Monitoring Today, Protecting Tomorrow."

Synthesizes:
1. Physical reference threshold breaches
2. Sensor reliability flags (to distinguish dirty water from faulty hardware)
3. Trend persistence and velocity
Produces transparent, actionable alerts with clear verification workflows.
"""

from typing import Dict, List, Any, Optional
import time
import uuid


class WaterAlert:
    """Represents an explainable, actionable alert."""
    def __init__(
        self,
        alert_id: str,
        timestamp: str,
        severity: str,  # 'Critical' | 'Warning' | 'Watch' | 'Info'
        param: str,
        param_name: str,
        observed_value: float,
        threshold_info: str,
        reason: str,
        is_sensor_unreliable: bool,
        sensor_status_note: str,
        recommended_action: str,
        acknowledged: bool = False
    ):
        self.alert_id = alert_id
        self.timestamp = timestamp
        self.severity = severity
        self.param = param
        self.param_name = param_name
        self.observed_value = observed_value
        self.threshold_info = threshold_info
        self.reason = reason
        self.is_sensor_unreliable = is_sensor_unreliable
        self.sensor_status_note = sensor_status_note
        self.recommended_action = recommended_action
        self.acknowledged = acknowledged

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp,
            "severity": self.severity,
            "param": self.param,
            "param_name": self.param_name,
            "observed_value": self.observed_value,
            "threshold_info": self.threshold_info,
            "reason": self.reason,
            "is_sensor_unreliable": self.is_sensor_unreliable,
            "sensor_status_note": self.sensor_status_note,
            "recommended_action": self.recommended_action,
            "acknowledged": self.acknowledged
        }


def generate_explainable_alerts(
    latest_reading: Dict[str, Any],
    sensor_health_map: Dict[str, Dict[str, Any]],
    trend_map: Dict[str, Dict[str, Any]],
    ref_ranges: Optional[Dict[str, Any]] = None,
    existing_alerts: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Evaluates current reading against physical limits, sensor reliability, and trend intelligence
    to generate deduplicated, explainable alerts.
    """
    from water_quality import DEFAULT_REFERENCE_RANGES
    ranges = ref_ranges or DEFAULT_REFERENCE_RANGES
    now_str = latest_reading.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
    alerts: List[Dict[str, Any]] = []

    # Map existing unacknowledged alert keys to prevent flood
    active_keys = set()
    if existing_alerts:
        for a in existing_alerts:
            if not a.get("acknowledged", False):
                active_keys.add(f"{a['param']}_{a['severity']}")

    for param, cfg in ranges.items():
        if param not in latest_reading or latest_reading[param] is None:
            continue

        val = float(latest_reading[param])
        sensor_health = sensor_health_map.get(param, {})
        sensor_issues = sensor_health.get("issues", [])
        is_sensor_bad = sensor_health.get("status") == "Needs Inspection" or len(sensor_issues) > 0
        sensor_note = (
            f"Hardware caution: {'; '.join(sensor_issues)}"
            if sensor_issues else "Sensor telemetry diagnostics report normal operation."
        )

        trend = trend_map.get(param, {})
        trend_state = trend.get("state", "Normal")
        rate_pct = trend.get("rate_of_change_pct", 0.0)

        min_safe = cfg["min_safe"]
        max_safe = cfg["max_safe"]
        min_caution = cfg["min_caution"]
        max_caution = cfg["max_caution"]

        # Scenario 1: Severe physical breach
        if val < min_caution or val > max_caution:
            severity = "Critical"
            direction = "extremely low" if val < min_caution else "extremely high"
            threshold_info = f"Configured Safe: [{min_safe}, {max_safe}] {cfg['unit']}"
            
            if is_sensor_bad:
                reason = (
                    f"{cfg['name']} measured {val:.2f} {cfg['unit']} ({direction}), "
                    f"BUT sensor reliability checks flagged hardware anomalies ({', '.join(sensor_issues)})."
                )
                action = (
                    "FIRST verify sensor probe physical connection, clean electrode, and check calibration with buffer. "
                    "Do NOT assume water is toxic until physical sensor reliability is verified with a secondary measurement."
                )
            else:
                reason = (
                    f"{cfg['name']} breached safety threshold ({val:.2f} {cfg['unit']}, {direction}). "
                    f"Trend velocity: {rate_pct:+.1f}%. Sensor hardware diagnostics are Healthy."
                )
                action = (
                    "Isolate water distribution line, activate reserve filtration, and collect water sample "
                    "for formal laboratory verification."
                )

            alert_key = f"{param}_{severity}"
            if alert_key not in active_keys:
                alerts.append(WaterAlert(
                    alert_id=str(uuid.uuid4())[:8],
                    timestamp=now_str,
                    severity=severity,
                    param=param,
                    param_name=cfg["name"],
                    observed_value=val,
                    threshold_info=threshold_info,
                    reason=reason,
                    is_sensor_unreliable=is_sensor_bad,
                    sensor_status_note=sensor_note,
                    recommended_action=action
                ).to_dict())

        # Scenario 2: Moderate deviation or persistent Watch/Warning trend
        elif val < min_safe or val > max_safe or trend_state in ("Warning", "Watch"):
            severity = "Warning" if trend_state == "Warning" or val < min_safe or val > max_safe else "Watch"
            direction = "below minimum" if val < min_safe else "above maximum"
            threshold_info = f"Configured Safe: [{min_safe}, {max_safe}] {cfg['unit']}"

            if is_sensor_bad:
                reason = (
                    f"{cfg['name']} reading of {val:.2f} {cfg['unit']} shows irregular behavior coinciding with "
                    f"sensor reliability issues ({'; '.join(sensor_issues)})."
                )
                action = "Inspect sensor wiring, verify grounding, and perform zero/span calibration test."
            else:
                reason = (
                    f"{cfg['name']} at {val:.2f} {cfg['unit']} is {direction} reference limits. "
                    f"Trend direction is {trend.get('direction', 'Stable')} ({rate_pct:+.1f}% shift)."
                )
                action = (
                    f"Inspect upstream filtration, check aeration/settling stages, and repeat reading in 10 minutes."
                )

            alert_key = f"{param}_{severity}"
            if alert_key not in active_keys:
                alerts.append(WaterAlert(
                    alert_id=str(uuid.uuid4())[:8],
                    timestamp=now_str,
                    severity=severity,
                    param=param,
                    param_name=cfg["name"],
                    observed_value=val,
                    threshold_info=threshold_info,
                    reason=reason,
                    is_sensor_unreliable=is_sensor_bad,
                    sensor_status_note=sensor_note,
                    recommended_action=action
                ).to_dict())

    return alerts
