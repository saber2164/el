# Battery Thermal Runaway Prediction - Output Directory Structure

## Overview

This directory contains all visualizations and analysis outputs generated during the ML pipeline execution. Each subdirectory corresponds to a specific step in the pipeline.

## Directory Structure

```
output/
├── data_cleaning/          # Step 1: Data Quality & Cleaning
├── feature_engineering/    # Step 2: Feature Analysis
├── model_training/         # Step 3: Training Process (reserved)
└── evaluation/             # Step 4: Model Performance & Results
```

---

## 1. Data Cleaning (`output/data_cleaning/`)

Visual and statistical analysis of raw data quality and target variable.

### Visualizations

- **01_data_quality_overview.png**
  - Missing values analysis (bar chart)
  - Data types distribution (pie chart)
  
- **02_target_distribution.png**
  - Histogram of thermal runaway energy yield
  - Box plot for outlier detection
  - Q-Q plot for normality assessment

- **03_summary_table.png**
  - Visual table of target variable statistics

### Tables (CSV)

- **target_statistics.csv**
  - Count, Mean, Std Dev, Min, 25%, Median, 75%, Max
  - Summary statistics for Corrected-Total-Energy-Yield-kJ

---

## 2. Feature Engineering (`output/feature_engineering/`)

Analysis of engineered features and their relationships.

### Visualizations

- **01_correlation_matrix.png**
  - Heatmap showing correlations between all numeric features and target
  - Identifies multicollinearity and strong predictors

- **02_feature_distributions.png**
  - Histograms for each numeric feature
  - Red/orange lines show mean and median values
  - Grid layout for easy comparison

- **03_feature_vs_target.png**
  - Scatter plots of each feature vs target variable
  - Correlation coefficients displayed on each plot
  - Identifies linear and non-linear relationships

### Tables (CSV)

- **feature_summary.csv**
  - Feature, Count, Mean, Std, Min, Max, Missing
  - Comprehensive statistics for all numeric features

---

## 3. Model Evaluation (`output/evaluation/`)

Comprehensive model performance analysis and diagnostics.

### Visualizations

- **01_prediction_plots.png**
  - Training set: Actual vs Predicted (with R² and RMSE)
  - Testing set: Actual vs Predicted (with R² and RMSE)
  - Perfect fit reference line (red dashed)

- **02_residual_analysis.png**
  - Training residuals scatter plot (check for patterns)
  - Testing residuals scatter plot
  - Training residuals distribution histogram
  - Testing residuals distribution histogram
  - Validates model assumptions

- **03_metrics_table.png**
  - Visual comparison table of RMSE, MAE, R² for train vs test
  - Quick performance overview

- **04_feature_importance.png**
  - Horizontal bar chart of top 15 features
  - Quantifies contribution of each feature to predictions
  - Identifies key risk drivers

### Tables (CSV)

- **metrics_comparison.csv**
  - RMSE, MAE, R² for both training and testing sets
  - Numerical values for reporting

- **feature_importance.csv**
  - Feature names and importance scores
  - Sorted by descending importance
  - All features included

---

## Key Insights from Visualizations

### Data Quality
- Dataset has 365 samples with 43 columns
- Target variable range: 1.37 - 110.83 kJ
- Some features have 21.6% - 42.5% missing values (handled via median imputation)

### Feature Relationships
- **Stored_Energy_Wh** shows strongest correlation with target (0.749)
- High correlation between Stored Energy and Voltage (expected, as Stored Energy = Capacity × Voltage)
- Trigger mechanisms show categorical differences in risk levels

### Model Performance
- Testing R² = 0.907 (explains 90.7% of variance)
- Testing RMSE = 7.4 kJ (average prediction error)
- Minimal overfitting (R² gap < 0.03)
- Residuals approximately normally distributed (validates model assumptions)

### Risk Drivers
1. **Stored_Energy_Wh**: 69.2% importance
2. **Pre-Test-Voltage**: 18.6% importance
3. **Cell-Capacity-Ah**: 9.8% importance
4. **Trigger-Mechanism_Nail**: 1.7% importance

---

## Usage

### View All Visualizations
```bash
# Linux/Mac
xdg-open output/*/*.png

# Or navigate and open individually
eog output/data_cleaning/01_data_quality_overview.png
```

### Access CSV Data
```bash
# View in terminal
cat output/*/*.csv

# Or open in spreadsheet software
libreoffice output/evaluation/metrics_comparison.csv
```

### Regenerate Outputs
```bash
# Run complete pipeline
python3 main.py

# Outputs will be overwritten in output/ directories
```

---

## Notes

- All images are saved at 300 DPI for publication quality
- PNG format chosen for lossless quality with transparency support
- CSV files can be imported into Excel, Python, R, or other analysis tools
- Directory structure mirrors the ML pipeline flow for intuitive navigation
