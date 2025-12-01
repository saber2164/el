# Visualization Summary

## Pipeline Outputs

All visualizations are saved in the `output/` directory, organized by pipeline stage.

### Total Files Generated: 15

#### 1. Data Cleaning (4 files)
- `01_data_quality_overview.png` - Missing values + data types
- `02_target_distribution.png` - Histogram, box plot, Q-Q plot
- `03_summary_table.png` - Statistics table visualization
- `target_statistics.csv` - Numerical statistics

#### 2. Feature Engineering (4 files)
- `01_correlation_matrix.png` - Feature correlation heatmap
- `02_feature_distributions.png` - All feature histograms
- `03_feature_vs_target.png` - Scatter plots with correlations
- `feature_summary.csv` - Feature statistics table

#### 3. Model Evaluation (6 files)
- `01_prediction_plots.png` - Actual vs predicted (train & test)
- `02_residual_analysis.png` - Residual plots and distributions
- `03_metrics_table.png` - Performance metrics visualization
- `04_feature_importance.png` - Top 15 features bar chart
- `metrics_comparison.csv` - RMSE, MAE, R² values
- `feature_importance.csv` - All features with importance scores

#### 4. Documentation (1 file)
- `README.md` - Complete documentation of all outputs

## Quick Access

View all images:
```bash
find output -name "*.png" | xargs eog &  # Linux
```

View all CSVs:
```bash
cat output/*/*.csv
```
