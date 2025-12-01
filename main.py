#!/usr/bin/env python3
"""
Battery Thermal Runaway Severity Prediction - Main Pipeline
============================================================
Orchestrates the complete ML pipeline for predicting thermal runaway severity.

Usage:
    python3 main.py

Author: Battery Safety ML Pipeline
Date: December 2025
"""

import warnings
warnings.filterwarnings('ignore')

from data_cleaning import clean_data
from feature_engineering import engineer_features
from model_training import split_data, train_random_forest
from evaluation import (
    evaluate_model, 
    analyze_feature_importance, 
    plot_feature_importance,
    generate_technical_summary
)


def main():
    """
    Main execution pipeline.
    """
    print("="*80)
    print("BATTERY THERMAL RUNAWAY SEVERITY PREDICTION MODEL")
    print("="*80)
    print("\nLoading NREL Battery Failure Databank dataset...")
    
    # Configuration
    DATASET_PATH = 'battery-failure-databank-revision2-feb24.xlsx'
    TARGET_COL = 'Corrected-Total-Energy-Yield-kJ'
    
    # Step 1: Data Cleaning
    df = clean_data(DATASET_PATH)
    
    # Step 2: Feature Engineering
    X, feature_cols = engineer_features(df)
    y = df[TARGET_COL]
    
    # Step 3: Train/Test Split
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    
    # Step 4: Model Training
    model = train_random_forest(
        X_train, y_train,
        n_estimators=100,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42
    )
    
    # Step 5: Model Evaluation
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test, output_dir='output/evaluation')
    
    # Step 6: Feature Importance Analysis
    feature_importance = analyze_feature_importance(model, X.columns, output_dir='output/evaluation')
    
    # Step 7: Generate Visualizations
    plot_feature_importance(feature_importance, output_path='output/evaluation/04_feature_importance.png', top_n=15)
    
    # Also save to root for backward compatibility
    plot_feature_importance(feature_importance, output_path='feature_importance.png', top_n=15)
    
    # Step 8: Technical Summary
    numeric_features = ['Cell-Capacity-Ah', 'Pre-Test-Cell-Open-Circuit-Voltage-V', 
                       'Cell-Casing-Thickness-µm', 'Stored_Energy_Wh']
    generate_technical_summary(feature_importance, metrics, df, TARGET_COL, numeric_features)
    
    # Summary
    print("\n" + "="*80)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\nOutputs generated:")
    print(f"  1. Data Cleaning: output/data_cleaning/")
    print(f"     - Data quality overview, target distributions, summary statistics")
    print(f"  2. Feature Engineering: output/feature_engineering/")
    print(f"     - Correlation matrix, feature distributions, feature vs target plots")
    print(f"  3. Model Evaluation: output/evaluation/")
    print(f"     - Prediction plots, residual analysis, metrics comparison, feature importance")
    print(f"\n  Model Performance: RMSE = {metrics['test_rmse']:.3f} kJ, R² = {metrics['test_r2']:.3f}")
    print("="*80)


if __name__ == "__main__":
    main()
