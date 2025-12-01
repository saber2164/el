# Data Directory

This directory contains the NREL Battery Failure Databank dataset used for training the thermal runaway severity prediction model.

## Files

- `battery-failure-databank-revision2-feb24.xlsx` - Original dataset (Excel format)
- `battery-failure-databank.csv` - Converted CSV version
- `battery-failure-databank-revision2-feb24.csv` - Additional CSV version

## Dataset Information

**Source:** NREL (National Renewable Energy Laboratory) Battery Failure Databank  
**Version:** Revision 2, February 2024  
**Samples:** 365 battery cells subjected to destructive thermal runaway testing  

**Key Columns:**
- `Corrected-Total-Energy-Yield-kJ` - Target variable (heat energy released)
- `Cell-Capacity-Ah` - Battery capacity
- `Pre-Test-Cell-Open-Circuit-Voltage-V` - Voltage before test
- `Cell-Casing-Thickness-µm` - Physical containment
- `Cell-Format` - Cell type (18650, 21700, D-Cell)
- `Trigger-Mechanism` - Failure method (Heater, Nail)

## Usage

The main pipeline (`thermal_prediction/main.py`) automatically loads:
```python
DATASET_PATH = '../data/battery-failure-databank-revision2-feb24.xlsx'
```

## Citation

If using this dataset, please cite the NREL Battery Failure Databank.
