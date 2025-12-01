# Battery Thermal Runaway Severity Prediction

Machine Learning model to predict thermal runaway severity (explosion risk) in lithium-ion batteries using the NREL Battery Failure Databank.

## Objective

Predict the total heat energy released during battery thermal runaway based on:
- Physical design (capacity, format, casing thickness)
- Operating state (voltage/state of charge)
- Failure mode (trigger mechanism)

## Results

- **Model Performance:** R² = 90.7%, RMSE = 7.4 kJ
- **Key Finding:** Stored Energy (Capacity × Voltage) explains 69.2% of risk variation
- **Critical Insight:** Each 1V increase in voltage → ~63% increase in explosion severity

## Quick Start

```bash
# Run the complete ML pipeline with visualizations
python3 main.py
```

**Note:** Requires the NREL Battery Failure Databank dataset (`battery-failure-databank-revision2-feb24.xlsx`)

## Modular Architecture

The pipeline is split into separate modules for easier debugging and maintenance:

### Core Modules

1. **data_cleaning.py** - Data loading, type conversion, filtering, missing value handling
   - Visualizations: Data quality overview, target distributions, Q-Q plots
   
2. **feature_engineering.py** - Feature creation and encoding
   - Visualizations: Correlation matrix, feature distributions, scatter plots
   
3. **model_training.py** - Train/test split and Random Forest training
   - Configuration with overfitting prevention
   
4. **evaluation.py** - Model evaluation and performance analysis
   - Visualizations: Prediction plots, residual analysis, metrics tables
   
5. **main.py** - Orchestrates the complete pipeline

### Legacy Files

- `battery_thermal_runaway_prediction.py` - Original monolithic script (legacy)
- `battery_thermal_runaway_walkthrough.md` - Detailed technical documentation

## Output Structure

All visualizations and analysis outputs are organized in the `output/` directory:

```
output/
├── data_cleaning/          # Data quality analysis (3 PNG, 1 CSV)
├── feature_engineering/    # Feature analysis (3 PNG, 1 CSV)
└── evaluation/            # Model performance (4 PNG, 2 CSV)
```

**Total:** 15 files including visualizations, tables, and documentation

See `output/README.md` for detailed documentation of all outputs.

## Key Findings

1. **Stored Energy** (69.2% importance) - Primary risk driver
2. **State of Charge** (18.6% importance) - 10× more important than trigger mechanism
3. **Trigger Mechanism** (1.7% importance) - Nail penetration > Heater
4. **Casing Thickness** (0.14% importance) - Minimal protective effect

## Safety Recommendations

- Limit storage/transport to 50-70% State of Charge
- Monitor Stored Energy (Capacity × Voltage) in real-time
- Enhanced physical protection against mechanical damage
- Focus on prevention rather than containment

## Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl scipy
```

Or install from requirements:
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- openpyxl
- scipy

## Documentation

- `README.md` - This file (project overview)
- `output/README.md` - Detailed output documentation
- `VISUALIZATION_SUMMARY.md` - Quick visualization reference
- `battery_thermal_runaway_walkthrough.md` - Complete technical walkthrough

## License

Research project for battery safety analysis.
