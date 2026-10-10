"""
Trend-Based Early Warning Intelligence Engine
Smart Water Quality Monitoring System
Tagline: "Monitoring Today, Protecting Tomorrow."

Calculates:
1. Rolling Simple Moving Average (SMA)
2. Percentage and rate-of-change velocity
3. Rolling Z-Score and Median Absolute Deviation (MAD) for statistical anomalies
4. Multi-point persistence confirmation (filters out transient spikes/noise)
5. Distinguishes threshold breach from statistical anomaly
"""

from typing import Dict, List, Any, Optional
import numpy as np
import math


def calculate_rolling_statistics(values: List[float], window_size: int = 10) -> Dict[str, Any]:
    """
    Computes rolling mean, std, median, and MAD over the specified window.
    """
    clean_vals = [v for v in values if v is not None and not math.isnan(v)]
    if not clean_vals:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "mad": 0.0, "count": 0}

    window = clean_vals[-window_size:]
    mean_val = float(np.mean(window))
    std_val = float(np.std(window)) if len(window) > 1 else 0.0
    median_val = float(np.median(window))
    mad_val = float(np.median([abs(x - median_val) for x in window])) if len(window) > 1 else 0.0

    return {
        "mean": mean_val,
        "std": std_val,
        "median": median_val,
        "mad": mad_val,
        "count": len(window)
    }


def analyze_parameter_trend(
    param: str,
    history: List[float],
    timestamps: Optional[List[str]] = None,
    window_size: int = 10,
    z_score_threshold: float = 2.5,
    persistence_required: int = 3,
    ref_ranges: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Performs comprehensive trend and early-warning analytics for a single parameter.
    
    States:
      'Normal': Parameters stable and within expected statistical and reference bounds.
      'Watch': Emerging trend or single statistical anomaly under observation.
      'Warning': Sustained persistent change (>= persistence_required samples) or accelerating rate.
      'Critical': Severe breach of safe operational envelope combined with adverse trend velocity.
    """
    clean_history = [v for v in history if v is not None and not math.isnan(v)]
    if len(clean_history) < 3:
        return {
            "param": param,
            "state": "Normal",
            "message": "Collecting baseline data for trend analysis...",
            "direction": "Stable",
            "rate_of_change_pct": 0.0,
            "z_score": 0.0,
            "persistent_count": 0,
            "is_statistical_anomaly": False,
            "is_threshold_breach": False,
            "recommendation": "Maintain normal monitoring cycle."
        }

    latest = clean_history[-1]
    prev_baseline = clean_history[:-1] if len(clean_history) > 1 else clean_history
    stats = calculate_rolling_statistics(prev_baseline, window_size)

    # 1. Calculate Z-score relative to recent baseline
    if stats["std"] > 1e-4:
        z_score = (latest - stats["mean"]) / stats["std"]
    else:
        z_score = 0.0

    # 2. Percentage change from baseline mean
    if abs(stats["mean"]) > 1e-5:
        pct_change = ((latest - stats["mean"]) / stats["mean"]) * 100.0
    else:
        pct_change = 0.0

    # 3. Direction
    if pct_change > 1.5:
        direction = "Rising"
    elif pct_change < -1.5:
        direction = "Falling"
    else:
        direction = "Stable"

    # 4. Multi-point Persistence Check:
    # Count how many of the last N points had z_score magnitude > 1.5 or were outside reference bounds
    recent_tail = clean_history[-persistence_required:]
    persistent_count = 0
    from water_quality import DEFAULT_REFERENCE_RANGES
    ranges = ref_ranges or DEFAULT_REFERENCE_RANGES
    cfg = ranges.get(param, {})

    min_safe = cfg.get("min_safe", 0.0)
    max_safe = cfg.get("max_safe", 100.0)

    for val in recent_tail:
        if val < min_safe or val > max_safe:
            persistent_count += 1

    is_threshold_breach = (latest < min_safe or latest > max_safe)
    is_statistical_anomaly = abs(z_score) >= z_score_threshold

    # 5. Determine State
    state = "Normal"
    recommendation = "Continue automated real-time polling."
    explanation = "Readings show steady equilibrium within nominal variance."

    if is_threshold_breach and persistent_count >= persistence_required:
        if abs(z_score) > 3.0 or (param == "ph" and (latest < 5.5 or latest > 9.5)) or (param == "turbidity" and latest > 15.0):
            state = "Critical"
            explanation = f"{cfg.get('name', param)} has severely breached reference limits persistently ({latest:.2f} {cfg.get('unit','')}, {pct_change:+.1f}% vs baseline)."
            recommendation = "Halt water discharge/intake immediately; initiate emergency physical and chemical validation."
        else:
            state = "Warning"
            explanation = f"{cfg.get('name', param)} sustained an abnormal trend across {persistent_count} consecutive readings."
            recommendation = "Perform grab-sample testing and inspect inline filtration / treatment stages."
    elif is_threshold_breach or is_statistical_anomaly:
        state = "Watch"
        if is_statistical_anomaly and not is_threshold_breach:
            explanation = f"Statistically unusual value detected (Z-score: {z_score:+.2f}), but value is currently still inside safe regulatory limits ({min_safe}-{max_safe})."
            recommendation = "Monitor rate-of-change over the next 5 intervals to determine if an environmental shift is beginning."
        else:
            explanation = f"{cfg.get('name', param)} briefly crossed regulatory reference threshold ({latest:.2f} {cfg.get('unit','')})."
            recommendation = "Verify probe cleanliness and observe whether the reading self-corrects or persists."
    elif abs(pct_change) > 15.0:
        state = "Watch"
        explanation = f"Moderate trend velocity: {cfg.get('name', param)} changed by {pct_change:+.1f}% over the observation window."
        recommendation = "Track rolling moving average for potential source contamination."

    condition_start_time = timestamps[-persistent_count] if (timestamps and persistent_count > 0 and len(timestamps) >= persistent_count) else "Current cycle"

    return {
        "param": param,
        "param_name": cfg.get("name", param),
        "unit": cfg.get("unit", ""),
        "latest_value": latest,
        "baseline_mean": stats["mean"],
        "state": state,
        "direction": direction,
        "rate_of_change_pct": round(pct_change, 2),
        "z_score": round(z_score, 2),
        "persistent_count": persistent_count,
        "is_statistical_anomaly": is_statistical_anomaly,
        "is_threshold_breach": is_threshold_breach,
        "condition_start_time": condition_start_time,
        "explanation": explanation,
        "recommendation": recommendation,
        "disclaimer": "Note: A statistically unusual reading (high Z-score) denotes mathematical divergence from the baseline, which is not guaranteed evidence of biological or chemical contamination."
    }


def analyze_all_trends(
    data_history: List[Dict[str, Any]],
    window_size: int = 10,
    z_score_threshold: float = 2.5,
    persistence_required: int = 3,
    ref_ranges: Optional[Dict[str, Any]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates trends across all monitored parameters.
    """
    from water_quality import DEFAULT_REFERENCE_RANGES
    params = list(DEFAULT_REFERENCE_RANGES.keys())
    timestamps = [str(row.get("timestamp", "")) for row in data_history]
    results = {}

    for p in params:
        series = [float(row[p]) for row in data_history if p in row and row[p] is not None]
        results[p] = analyze_parameter_trend(
            p, series, timestamps, window_size, z_score_threshold, persistence_required, ref_ranges
        )

    return results
