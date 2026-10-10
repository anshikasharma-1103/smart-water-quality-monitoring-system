"""
Data Service, Simulation Controls, and Cloud/IoT Integration Layer
Smart Water Quality Monitoring System
Tagline: "Monitoring Today, Protecting Tomorrow."
"""

import time
import random
import datetime
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

# Standard baseline parameters for realistic simulated freshwater
BASELINE_VALUES = {
    "ph": 7.35,
    "turbidity": 1.8,
    "tds": 240.0,
    "conductivity": 380.0,
    "temperature": 21.5
}


def generate_single_reading(
    last_reading: Optional[Dict[str, Any]] = None,
    source: str = "Demo (Simulated)"
) -> Dict[str, Any]:
    """
    Generates a realistic, correlated sample reading with natural Brownian random drift.
    """
    if last_reading is None:
        last_reading = BASELINE_VALUES

    # Small realistic Gaussian drift
    ph = round(float(last_reading.get("ph", 7.35)) + random.gauss(0, 0.04), 2)
    # Ensure natural bounds within typical freshwater range
    ph = max(6.6, min(8.3, ph))

    turbidity = round(max(0.2, float(last_reading.get("turbidity", 1.8)) + random.gauss(0, 0.15)), 2)
    
    tds = round(max(80.0, float(last_reading.get("tds", 240.0)) + random.gauss(0, 4.0)), 1)
    
    # Conductivity strongly correlates with TDS (typical ratio ~1.5 - 1.7)
    conductivity = round(tds * random.uniform(1.55, 1.62), 1)

    temp = round(max(15.0, min(27.0, float(last_reading.get("temperature", 21.5)) + random.gauss(0, 0.1))), 2)

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "timestamp": now_str,
        "ph": ph,
        "turbidity": turbidity,
        "tds": tds,
        "conductivity": conductivity,
        "temperature": temp,
        "source": source
    }


def generate_initial_history(count: int = 25) -> List[Dict[str, Any]]:
    """
    Constructs an initial realistic sequence of timestamped readings for the dashboard.
    """
    readings: List[Dict[str, Any]] = []
    current = BASELINE_VALUES.copy()
    base_time = datetime.datetime.now() - datetime.timedelta(seconds=count * 5)

    for i in range(count):
        step_time = (base_time + datetime.timedelta(seconds=i * 5)).strftime("%Y-%m-%d %H:%M:%S")
        ph = round(max(6.8, min(8.0, current["ph"] + random.gauss(0, 0.03))), 2)
        turbidity = round(max(0.5, min(3.5, current["turbidity"] + random.gauss(0, 0.1))), 2)
        tds = round(max(180.0, min(320.0, current["tds"] + random.gauss(0, 3.0))), 1)
        conductivity = round(tds * random.uniform(1.56, 1.60), 1)
        temp = round(max(19.0, min(23.5, current["temperature"] + random.gauss(0, 0.05))), 2)

        current = {"ph": ph, "turbidity": turbidity, "tds": tds, "conductivity": conductivity, "temperature": temp}
        readings.append({
            "timestamp": step_time,
            "ph": ph,
            "turbidity": turbidity,
            "tds": tds,
            "conductivity": conductivity,
            "temperature": temp,
            "source": "Demo (Simulated)"
        })

    return readings


def inject_pollution_spike(last_reading: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Simulates a sudden industrial chemical effluent or agricultural runoff event:
    Severe drop in pH (acidic), sharp jump in turbidity, surge in TDS and conductivity.
    """
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "timestamp": now_str,
        "ph": 4.85,             # Acidic breach (< 6.5)
        "turbidity": 38.6,       # High turbidity (> 5 NTU)
        "tds": 980.0,            # High TDS (> 500 ppm)
        "conductivity": 1640.0,  # Surge in electrical conductivity
        "temperature": 27.2,
        "source": "Demo (Pollution Spike Injected)"
    }


def inject_faulty_sensor(last_reading: Optional[Dict[str, Any]] = None, fault_type: str = "freeze") -> Dict[str, Any]:
    """
    Simulates hardware failure modes:
    - 'freeze': Flatline identical reading
    - 'jump': Implausible impossible step change (e.g. pH jumps to 13.9)
    - 'noise': Extreme jitter
    """
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prev = last_reading or BASELINE_VALUES

    if fault_type == "freeze":
        return {
            "timestamp": now_str,
            "ph": float(prev.get("ph", 7.20)),  # Repeated identical
            "turbidity": float(prev.get("turbidity", 2.1)),
            "tds": float(prev.get("tds", 250.0)),
            "conductivity": float(prev.get("conductivity", 400.0)),
            "temperature": float(prev.get("temperature", 22.0)),
            "source": "Demo (Faulty Sensor - Frozen ADC)"
        }
    elif fault_type == "jump":
        return {
            "timestamp": now_str,
            "ph": 13.85,         # Sudden impossible jump from ~7.3 to ~13.8
            "turbidity": float(prev.get("turbidity", 2.1)),
            "tds": float(prev.get("tds", 250.0)),
            "conductivity": float(prev.get("conductivity", 400.0)),
            "temperature": float(prev.get("temperature", 22.0)),
            "source": "Demo (Faulty Sensor - Voltage Spike)"
        }
    else:
        return {
            "timestamp": now_str,
            "ph": round(random.choice([2.1, 12.8, 0.4, 11.2]), 2),  # Extreme noise
            "turbidity": round(random.uniform(80.0, 300.0), 1),
            "tds": float(prev.get("tds", 250.0)),
            "conductivity": float(prev.get("conductivity", 400.0)),
            "temperature": float(prev.get("temperature", 22.0)),
            "source": "Demo (Faulty Sensor - Unshielded Noise)"
        }


def validate_csv_upload(df: pd.DataFrame) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """
    Validates user-uploaded CSV file for required columns, data types, and physical sanity.
    """
    required_cols = ["ph", "turbidity", "tds", "conductivity", "temperature"]
    # Check case-insensitive match
    col_map = {c.lower().strip(): c for c in df.columns}
    missing = [c for c in required_cols if c not in col_map]

    if missing:
        return False, f"Missing required parameter column(s): {', '.join(missing)}", None

    standardized = pd.DataFrame()
    if "timestamp" in col_map:
        standardized["timestamp"] = df[col_map["timestamp"]].astype(str)
    else:
        standardized["timestamp"] = [
            (datetime.datetime.now() - datetime.timedelta(seconds=(len(df)-i)*5)).strftime("%Y-%m-%d %H:%M:%S")
            for i in range(len(df))
        ]

    for col in required_cols:
        orig = col_map[col]
        try:
            numeric_series = pd.to_numeric(df[orig], errors="coerce")
            if numeric_series.isna().all():
                return False, f"Column '{orig}' contains no valid numeric values.", None
            standardized[col] = numeric_series
        except Exception as e:
            return False, f"Failed parsing column '{orig}': {str(e)}", None

    standardized["source"] = "CSV Upload"
    return True, f"Successfully validated {len(standardized)} rows.", standardized


def fetch_thingspeak_data(channel_id: str, api_key: str, results_limit: int = 15) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """
    Fetches real-time sensor streams from ThingSpeak Cloud REST API.
    Fails gracefully if credentials are not configured or connection drops.
    """
    if not channel_id or not api_key:
        return False, "ThingSpeak Channel ID or Read API Key is not configured.", []

    try:
        import requests
        url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.json?api_key={api_key}&results={results_limit}"
        resp = requests.get(url, timeout=5.0)
        
        if resp.status_code == 200:
            data = resp.json()
            feeds = data.get("feeds", [])
            if not feeds:
                return False, "ThingSpeak channel returned empty feed history.", []
            
            parsed: List[Dict[str, Any]] = []
            for item in feeds:
                try:
                    parsed.append({
                        "timestamp": item.get("created_at", ""),
                        "ph": float(item.get("field1", 7.0)),
                        "turbidity": float(item.get("field2", 1.0)),
                        "tds": float(item.get("field3", 200.0)),
                        "conductivity": float(item.get("field4", 320.0)),
                        "temperature": float(item.get("field5", 22.0)),
                        "source": f"ThingSpeak (Channel {channel_id})"
                    })
                except (ValueError, TypeError):
                    continue
            
            return True, f"Retrieved {len(parsed)} readings from ThingSpeak.", parsed
        else:
            return False, f"ThingSpeak API responded with status HTTP {resp.status_code}", []
    except Exception as e:
        return False, f"ThingSpeak network connection failed: {str(e)}", []
