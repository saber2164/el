# Battery Thermal Runaway Severity Prediction

Machine Learning model to predict thermal runaway severity (explosion risk) in lithium-ion batteries using the NREL Battery Failure Databank.

## 🎯 Objective

Predict the total heat energy released during battery thermal runaway based on:
- Physical design (capacity, format, casing thickness)
- Operating state (voltage/state of charge)
- Failure mode (trigger mechanism)

## 📊 Results

- **Model Performance:** R² = 90.7%, RMSE = 7.4 kJ
- **Key Finding:** Stored Energy (Capacity × Voltage) explains 69.2% of risk variation
- **Critical Insight:** Each 1V increase in voltage → ~63% increase in explosion severity

## 🚀 Quick Start

```bash
# Run the ML pipeline
python3 battery_thermal_runaway_prediction.py
```

**Note:** Requires the NREL Battery Failure Databank dataset (`battery-failure-databank-revision2-feb24.xlsx`)

## 📁 Project Files

- `battery_thermal_runaway_prediction.py` - Complete ML pipeline
- `battery_thermal_runaway_walkthrough.md` - Detailed technical documentation
- `feature_importance.png` - Risk driver visualization

## 🔬 Key Findings

1. **Stored Energy** (69.2% importance) - Primary risk driver
2. **State of Charge** (18.6% importance) - 10× more important than trigger mechanism
3. **Trigger Mechanism** (1.7% importance) - Nail penetration > Heater
4. **Casing Thickness** (0.14% importance) - Minimal protective effect

## 🛡️ Safety Recommendations

- Limit storage/transport to 50-70% State of Charge
- Monitor Stored Energy (Capacity × Voltage) in real-time
- Enhanced physical protection against mechanical damage
- Focus on prevention rather than containment

## 🔧 Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- openpyxl

## 📄 License

Research project for battery safety analysis.
