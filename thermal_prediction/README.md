# Thermal Prediction Module

This directory contains the machine learning pipeline for predicting battery thermal runaway severity.

## Module Files

**Main Pipeline:**
- `main.py` - Orchestrates the complete ML pipeline

**Core Modules:**
- `data_cleaning.py` - Data loading, preprocessing, and quality visualizations
- `feature_engineering.py` - Feature creation, encoding, and analysis
- `model_training.py` - Random Forest model training
- `evaluation.py` - Performance evaluation and visualizations

**Legacy:**
- `battery_thermal_runaway_prediction.py` - Original monolithic script (for reference)

## Usage

Run from this directory:

```bash
cd thermal_prediction
python3 main.py
```

## Data Requirements

The pipeline expects the dataset in `../data/battery-failure-databank-revision2-feb24.xlsx`

## Outputs

All visualizations and results are saved to:
- `../output/data_cleaning/` - Data quality analysis
- `../output/feature_engineering/` - Feature analysis  
- `../output/evaluation/` - Model performance metrics
