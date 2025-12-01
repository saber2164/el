# Battery Safety Optimizer - Deployment Tool

Deployable safety tool for Battery Management Systems (BMS) to monitor and enforce thermal runaway risk limits.

## Overview

This module provides:
- **Safety Optimizer:** Calculate maximum safe voltage limits
- **Risk Predictions:** Real-time thermal runaway severity estimation
- **Web Interface:** Streamlit-based BMS monitoring dashboard

## Quick Start

### 1. Train and Save Model

First, train the model and save it for deployment:

```bash
cd ../thermal_prediction
python3 main.py
```

This creates `models/rf_model.joblib` and `models/model_metadata.joblib`.

### 2. Test Safety Optimizer

```bash
cd ../safety_optimizer_algo
python3 safety_optimizer.py
```

### 3. Launch Web App

```bash
streamlit run app.py
```

Access at: http://localhost:8501

## Components

### safety_optimizer.py

**PhaseSafetyOptimizer** class for calculating safe operating limits.

**Key Methods:**
- `predict_risk(voltage_v, capacity_ah)` - Predict thermal runaway severity
- `get_max_safe_voltage(capacity_ah, max_risk_kj)` - Calculate voltage limit
- `generate_safety_boundary()` - Create voltage vs capacity boundary
- `get_risk_level(risk_kj)` - Categorize risk (Safe/Warning/Critical)

**Fixed Parameters (Phase 0/1):**
- Cell Format: 18650
- Casing Thickness: 30µm
- Trigger: Nail (worst-case)

**Example:**
```python
from safety_optimizer import PhaseSafetyOptimizer

optimizer = PhaseSafetyOptimizer()

# Predict risk
risk_kj = optimizer.predict_risk(voltage_v=4.2, capacity_ah=3.0)
print(f"Predicted risk: {risk_kj:.2f} kJ")

# Get max safe voltage
max_v = optimizer.get_max_safe_voltage(capacity_ah=3.0, max_risk_kj=40.0)
print(f"Max safe voltage: {max_v:.2f} V")
```

### app.py

Streamlit web application providing BMS safety interface.

**Enhanced Features:**
- **Dual Input Controls:** Each parameter has both slider and number input
  - Sliders for quick adjustments
  - Number inputs for precise manual entry
  - Both controls sync automatically
- Live voltage and capacity inputs with 0.01 precision
- Real-time risk prediction
- Risk gauge (Green/Yellow/Red zones)
- Critical voltage alerts with recommended actions
- Safety boundary visualization
- Current state tracking on voltage vs capacity plot

**Inputs (Sidebar):**
- **Live Voltage:** 2.5-4.5V (slider + number input, 0.01V precision)
- **SOH Capacity:** 1.0-5.0 Ah (slider + number input, 0.01 Ah precision)
- **Battery Phase:** Dropdown (Phase 0 Prototype / Phase 1 Production)
- **Max Risk Threshold:** 20-60 kJ (slider + number input, 1.0 kJ steps)

**Outputs:**
- **Current State Metrics** (with icons ⚡🔋⚙️):
  - Voltage display with safety margin indicator
  - Capacity display
  - Stored Energy calculation
- **Risk Gauge:**
  - Color-coded zones (Green/Yellow/Red)
  - Current risk value
  - Threshold indicator
- **Safety Alerts:**
  - "SAFE OPERATION" when within limits
  - "CRITICAL: DERATE CHARGE" with voltage reduction recommendation
- **Safety Boundary Map:**
  - Voltage vs Capacity plot
  - Safe operating area (green zone)
  - Unsafe zone (red zone)
  - Current battery state marker
- **Phase Specifications Panel** (📋):
  - Cell format, casing thickness, trigger mode

## Directory Structure

```
safety_optimizer_algo/
├── models/                    # Trained models (generated)
│   ├── rf_model.joblib       # Random Forest model
│   └── model_metadata.joblib # Feature names & config
├── safety_optimizer.py        # Core optimizer class
├── app.py                    # Streamlit web interface
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

```bash
pip install -r requirements.txt
```

## Deployment Considerations

### For Production BMS:

1. **Model Updates:** Retrain periodically with new failure data
2. **Real-time Integration:** Connect to BMS data feed via API
3. **Alert System:** Integrate with BMS warning/shutdown logic
4. **Logging:** Record all voltage limit violations
5. **Backup:** Fail-safe to lowest safe voltage if model fails

### Risk Thresholds:

- **Safe:** < 30 kJ (Green)
- **Warning:** 30-50 kJ (Yellow)
- **Critical:** 50-70 kJ (Orange)
- **Extreme:** > 70 kJ (Red)

## API Example

For programmatic access:

```python
from safety_optimizer import PhaseSafetyOptimizer

# Initialize
optimizer = PhaseSafetyOptimizer()

# BMS data feed (example)
bms_data = {
    'voltage_v': 4.15,
    'capacity_ah': 2.8
}

# Calculate safety metrics
risk = optimizer.predict_risk(bms_data['voltage_v'], bms_data['capacity_ah'])
max_safe_v = optimizer.get_max_safe_voltage(bms_data['capacity_ah'], max_risk_kj=40)
risk_level, color = optimizer.get_risk_level(risk)

# Decision logic
if bms_data['voltage_v'] > max_safe_v:
    print(f"ALERT: Derate voltage from {bms_data['voltage_v']}V to {max_safe_v}V")
else:
    print(f"Safe operation - Risk: {risk:.1f} kJ ({risk_level})")
```

## Testing

Run optimizer tests:
```bash
python3 safety_optimizer.py
```

Expected output:
- Risk predictions for various scenarios
- Max safe voltage calculations (2-5 Ah)
- Verification that higher capacity → lower safe voltage

## Safety Notes

- **Worst-case assumption:** Nail trigger (most severe failure mode)
- **Conservative limits:** Binary search ensures predictions stay below threshold
- **Hardware fixed:** Cannot adjust cell format or casing thickness for Phase 0/1
- **State of Charge:** Primary controllable parameter for risk management

## Performance

Model Metrics (from training):
- R² = 90.7% (high accuracy)
- RMSE = 7.4 kJ (average error)
- Trained on 365 destructive test samples

## License

Research project for battery safety analysis.
