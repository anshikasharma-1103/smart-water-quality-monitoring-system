# Smart Water Quality Monitoring System
**Tagline:** *“Monitoring Today, Protecting Tomorrow.”*

An AI- and IoT-driven environmental technology prototype designed to monitor water parameters, classify water conditions using machine learning, detect emerging contamination trends, evaluate physical sensor reliability, and generate explainable, actionable alerts.

---

## 🏆 Project Overview & Innovation Highlights

1. **Sensor Reliability Intelligence (Unique Feature A):**  
   Automatically flags hardware anomalies (frozen ADC outputs, sudden implausible jumps, excessive electrical noise, missing telemetry, and calibration drift) to prevent false pollution alarms.
2. **Trend-Based Early Warning (Unique Feature B):**  
   Employs rolling statistics, Z-scores, and multi-point persistence confirmation to detect gradual deterioration and rapid contamination velocity before critical regulatory limits are breached.
3. **Explainable Alert Engine (Unique Feature C):**  
   Provides human-understandable alert explanations detailing observed values, threshold margins, whether the reading may be caused by an unreliable sensor, and prescriptive next actions.
4. **Interactive Multi-Modal Dashboard:**  
   Responsive 8-section environmental dashboard built with Python, Streamlit, and Plotly, accompanied by an interactive browser interface.

> ⚠️ **Pre-Screening Disclaimer:** Sensor readings and AI predictions serve as preliminary screening indicators and do not replace certified laboratory testing for microbiological pathogens (e.g. *E. coli*) or heavy metals pursuant to WHO, EPA, or BIS IS 10500 standards.

---

## 📁 Project File Tree

```
├── app.py                      # Main Streamlit web application entry point
├── requirements.txt            # Python dependencies (Streamlit, Pandas, Plotly, Scikit-learn, etc.)
├── water_quality.py            # Parameter bounds, WQI computation, and physical validation
├── sensor_reliability.py       # 7-point sensor diagnostics and calibration checks (Feature A)
├── trend_detection.py          # Rolling statistics, Z-scores, and persistence filter (Feature B)
├── alert_engine.py             # Explainable alert generator with hardware cross-checking (Feature C)
├── data_service.py             # Realistic Brownian simulator, CSV import/export, and IoT handlers
├── train_model.py              # Reproducible Random Forest training pipeline and model persistence
├── water_model.joblib          # Serialized Random Forest model artifact
├── tests/
│   └── test_water_quality.py   # Automated pytest unit test suite
├── .env.example                # Template for optional ThingSpeak & MQTT credentials
└── README.md                   # Full documentation, setup guide, and ideathon script
```

---

## 🚀 Quickstart Installation & Execution

### Prerequisites
- Python 3.10 or 3.11 installed
- Git (optional)

### 1. Clone or Extract the Project
```bash
git clone <repository-url>
cd smart-water-quality-monitoring
```

### 2. Create and Activate a Virtual Environment
- **On Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
- **On Windows (Command Prompt):**
  ```cmd
  python -m venv venv
  venv\Scripts\activate
  ```
- **On Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Machine Learning Model (Optional - Pre-trained on start)
```bash
python train_model.py
```

### 5. Launch the Streamlit Web Dashboard
```bash
streamlit run app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

### 6. Run the Automated Unit Tests
```bash
pytest tests/ -v
```

---

## 💻 Running in VS Code on Windows

1. Open **VS Code**.
2. Go to **File → Open Folder...** and select the project directory.
3. Open a terminal inside VS Code (**Ctrl + `** or **Terminal → New Terminal**).
4. Verify your Python interpreter: Press **Ctrl + Shift + P**, type `Python: Select Interpreter`, and choose your virtual environment (`.\venv\Scripts\python.exe`).
5. In the terminal, execute:
   ```powershell
   pip install -r requirements.txt
   streamlit run app.py
   ```
6. Click the local URL in the terminal or navigate to `http://localhost:8501`.

---

## 🌐 Running in Google Colab

To run this application in Google Colab with localtunnel:

1. Create a new Colab Notebook and run:
   ```python
   # 1. Install dependencies
   !pip install streamlit pyngrok plotly pandas scikit-learn joblib

   # 2. Upload or clone project files
   # (Upload app.py, water_quality.py, sensor_reliability.py, trend_detection.py, alert_engine.py, data_service.py, train_model.py)

   # 3. Train the model
   !python train_model.py

   # 4. Run Streamlit in the background with localtunnel
   !npm install -g localtunnel
   !streamlit run app.py & npx localtunnel --port 8501
   ```
2. Click the public `localtunnel.me` URL generated in the output.

---

## 🎮 How to Test Demo Mode

Demo mode functions 100% offline without physical sensors or API keys:

1. **Step-by-Step Simulation:**  
   Click the **"➕ Next Sample"** button in the sidebar to advance the water stream with realistic Gaussian Brownian drift.
2. **Pollution Spike Injection:**  
   Click **"⚠️ Simulate Pollution Spike"**. Observe:
   - pH drops from ~7.3 to 4.85 (acidic breach).
   - Turbidity jumps to 38.6 NTU.
   - WQI drops to "Unsafe".
   - Early-Warning Intelligence flags a rapid rising trend.
   - The Alert Center creates a Critical alert with step-by-step containment instructions.
3. **Sensor Hardware Fault Simulation:**  
   Select a fault mode (`freeze`, `jump`, or `noise`) and click **"🛠️ Simulate Faulty Sensor"**:
   - The **Sensor Reliability Intelligence** tab flags the specific hardware issue.
   - The Alert Engine notifies operators: *"Hardware caution: Sensor reliability flagged; check probe before assuming water is contaminated."*
4. **Data Reset:**  
   Click **"🔄 Reset Data"** to restore nominal fresh water baseline readings.

---

## 📡 Optional Cloud & IoT Integration (ESP32 / ThingSpeak)

### ThingSpeak REST API
To stream from ThingSpeak:
1. Create a ThingSpeak channel with 5 fields:
   - Field 1: pH
   - Field 2: Turbidity (NTU)
   - Field 3: TDS (ppm)
   - Field 4: Electrical Conductivity (µS/cm)
   - Field 5: Temperature (°C)
2. Copy `.env.example` to `.env` and fill in:
   ```env
   THINGSPEAK_CHANNEL_ID="your_channel_id"
   THINGSPEAK_READ_API_KEY="your_read_api_key"
   ```

### ESP32 Microcontroller JSON Payload
The ESP32 publishes sensor telemetry to MQTT or HTTP in this standard format:
```json
{
  "device_id": "esp32_water_node_01",
  "timestamp": "2026-10-10 12:45:00",
  "ph": 7.28,
  "turbidity": 1.65,
  "tds": 235.0,
  "conductivity": 376.0,
  "temperature": 21.8,
  "battery_voltage": 3.92
}
```

---

## 🔧 Troubleshooting Guide

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `ModuleNotFoundError: No module named 'streamlit'` | Virtual environment not active or dependencies not installed | Run `pip install -r requirements.txt` within active virtualenv. |
| Streamlit port collision (`Port 8501 is already in use`) | Another instance is running | Run `streamlit run app.py --server.port 8502`. |
| Missing model file (`water_model.joblib`) | Model has not yet been generated | Run `python train_model.py` to produce a fresh artifact. |
| CSV Upload Error | Header columns do not match schema | Ensure CSV columns include: `ph`, `turbidity`, `tds`, `conductivity`, `temperature`. |
| ThingSpeak connection timeout | Invalid API Key or firewall blockage | Check API Key in `.env` or navigate to Demo Mode in sidebar. |

---

## 🎤 Ideathon Presentation Script (2-Minute Pitch)

> **[0:00 - 0:25] The Problem**  
> *"Good morning judges. Clean water is our most fundamental necessity, yet traditional municipal and industrial monitoring systems suffer from two fatal flaws: they rely on static threshold alarms that trigger false alarms on sensor noise, and they cannot distinguish between water that is genuinely contaminated and a physical probe that has simply flatlined or drifted."*

> **[0:25 - 0:55] Our Innovation**  
> *"We built the **Smart Water Quality Monitoring System** with our motto: 'Monitoring Today, Protecting Tomorrow.'  
> Our system introduces three distinct innovations:  
> First, **Sensor Reliability Intelligence** monitors ADC signals for frozen flatlines, excessive noise, and calibration drift.  
> Second, **Trend-Based Early Warning** uses rolling Z-scores and persistence filtering to catch deteriorating trends before water becomes hazardous.  
> Third, our **Explainable Alert Engine** tells the operator exactly which parameter failed and whether the probe is trustworthy before advising physical intervention."*

> **[0:55 - 1:30] Live Demonstration**  
> *(Show Dashboard)*  
> *"Here on the dashboard, we are monitoring all five core parameters: pH, Turbidity, TDS, Conductivity, and Temperature.  
> Watch what happens when I simulate a faulty sensor:* *(click 'Simulate Faulty Sensor')*  
> *Notice that our system does NOT panic and sound a chemical alarm. It flags a hardware warning: 'Frozen output detected on pH probe; verify wiring before treatment.'  
> Now watch what happens during a real industrial discharge:* *(click 'Simulate Pollution Spike')*  
> *The Random Forest AI model classifies the water as Unsafe, the early warning system notes a 380% surge in turbidity, and an explainable alert is dispatched."*

> **[1:30 - 2:00] Impact & Future Roadmap**  
> *"Our architecture integrates seamlessly with ESP32 edge nodes and ThingSpeak cloud telemetry. By preventing costly false alarms and catching real contamination early, we empower communities with affordable, intelligent water protection. Thank you!"*
