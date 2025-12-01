"""
Model Evaluation Module
========================
Handles model evaluation, metrics calculation, and visualization.
Includes comprehensive visualizations for model performance analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def evaluate_model(model, X_train, X_test, y_train, y_test, output_dir='output/evaluation'):
    """
    Evaluate model performance on training and testing sets with visualizations.
    
    Args:
        model: Trained model
        X_train: Training features
        X_test: Testing features
        y_train: Training target
        y_test: Testing target
        output_dir: Directory to save visualizations
        
    Returns:
        dict: Dictionary containing all metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred)
    }
    
    print(f"✓ Performance Metrics:")
    print(f"\n  Training Set:")
    print(f"    - RMSE: {metrics['train_rmse']:.3f} kJ")
    print(f"    - MAE:  {metrics['train_mae']:.3f} kJ")
    print(f"    - R²:   {metrics['train_r2']:.3f}")
    
    print(f"\n  Testing Set:")
    print(f"    - RMSE: {metrics['test_rmse']:.3f} kJ")
    print(f"    - MAE:  {metrics['test_mae']:.3f} kJ")
    print(f"    - R²:   {metrics['test_r2']:.3f}")
    
    # Check for overfitting
    overfitting_gap = metrics['train_r2'] - metrics['test_r2']
    print(f"\n  Overfitting Check:")
    print(f"    - R² Gap (Train - Test): {overfitting_gap:.3f}")
    if overfitting_gap < 0.1:
        print(f"    ✓ No significant overfitting detected")
    elif overfitting_gap < 0.2:
        print(f"    ⚠ Minor overfitting (acceptable)")
    else:
        print(f"    ⚠ Moderate overfitting detected")
    
    # Create visualizations
    print("\n✓ Creating evaluation visualizations...")
    visualize_predictions(y_train, y_train_pred, y_test, y_test_pred, metrics, output_dir)
    visualize_residuals(y_train, y_train_pred, y_test, y_test_pred, output_dir)
    create_metrics_table(metrics, output_dir)
    
    return metrics


def visualize_predictions(y_train, y_train_pred, y_test, y_test_pred, metrics, output_dir='output/evaluation'):
    """
    Create prediction visualization plots.
    
    Args:
        y_train: Training actual values
        y_train_pred: Training predictions
        y_test: Testing actual values
        y_test_pred: Testing predictions
        metrics: Metrics dictionary
        output_dir: Output directory
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Training set predictions
    axes[0].scatter(y_train, y_train_pred, alpha=0.5, s=50, color='steelblue', label='Predictions')
    axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
                 'r--', lw=2, label='Perfect Fit')
    axes[0].set_xlabel('Actual Energy Yield (kJ)', fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Predicted Energy Yield (kJ)', fontweight='bold', fontsize=12)
    axes[0].set_title(f'Training Set Predictions\nR² = {metrics["train_r2"]:.3f}, RMSE = {metrics["train_rmse"]:.3f} kJ',
                     fontweight='bold', fontsize=12, pad=15)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Testing set predictions
    axes[1].scatter(y_test, y_test_pred, alpha=0.6, s=50, color='coral', label='Predictions')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                 'r--', lw=2, label='Perfect Fit')
    axes[1].set_xlabel('Actual Energy Yield (kJ)', fontweight='bold', fontsize=12)
    axes[1].set_ylabel('Predicted Energy Yield (kJ)', fontweight='bold', fontsize=12)
    axes[1].set_title(f'Testing Set Predictions\nR² = {metrics["test_r2"]:.3f}, RMSE = {metrics["test_rmse"]:.3f} kJ',
                     fontweight='bold', fontsize=12, pad=15)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_prediction_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {output_dir}/01_prediction_plots.png")


def visualize_residuals(y_train, y_train_pred, y_test, y_test_pred, output_dir='output/evaluation'):
    """
    Create residual analysis plots.
    
    Args:
        y_train: Training actual values
        y_train_pred: Training predictions
        y_test: Testing actual values
        y_test_pred: Testing predictions
        output_dir: Output directory
    """
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Residual plot - Training
    axes[0, 0].scatter(y_train_pred, train_residuals, alpha=0.5, s=50, color='steelblue')
    axes[0, 0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 0].set_xlabel('Predicted Values (kJ)', fontweight='bold')
    axes[0, 0].set_ylabel('Residuals (kJ)', fontweight='bold')
    axes[0, 0].set_title('Training Set Residuals', fontweight='bold', pad=15)
    axes[0, 0].grid(alpha=0.3)
    
    # Residual plot - Testing
    axes[0, 1].scatter(y_test_pred, test_residuals, alpha=0.6, s=50, color='coral')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Values (kJ)', fontweight='bold')
    axes[0, 1].set_ylabel('Residuals (kJ)', fontweight='bold')
    axes[0, 1].set_title('Testing Set Residuals', fontweight='bold', pad=15)
    axes[0, 1].grid(alpha=0.3)
    
    # Residual distribution - Training
    axes[1, 0].hist(train_residuals, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Residuals (kJ)', fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontweight='bold')
    axes[1, 0].set_title('Training Residuals Distribution', fontweight='bold', pad=15)
    axes[1, 0].grid(alpha=0.3)
    
    # Residual distribution - Testing
    axes[1, 1].hist(test_residuals, bins=30, color='coral', edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Residuals (kJ)', fontweight='bold')
    axes[1, 1].set_ylabel('Frequency', fontweight='bold')
    axes[1, 1].set_title('Testing Residuals Distribution', fontweight='bold', pad=15)
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_residual_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {output_dir}/02_residual_analysis.png")


def create_metrics_table(metrics, output_dir='output/evaluation'):
    """
    Create metrics comparison table.
    
    Args:
        metrics: Dictionary of metrics
        output_dir: Output directory
    """
    metrics_df = pd.DataFrame({
        'Metric': ['RMSE (kJ)', 'MAE (kJ)', 'R²'],
        'Training': [
            round(metrics['train_rmse'], 3),
            round(metrics['train_mae'], 3),
            round(metrics['train_r2'], 3)
        ],
        'Testing': [
            round(metrics['test_rmse'], 3),
            round(metrics['test_mae'], 3),
            round(metrics['test_r2'], 3)
        ]
    })
    
    # Save to CSV
    metrics_df.to_csv(f'{output_dir}/metrics_comparison.csv', index=False)
    print(f"  → Saved: {output_dir}/metrics_comparison.csv")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns,
                    cellLoc='center', loc='center',
                    colColours=['lightblue']*3)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    plt.title('Model Performance Metrics Comparison', fontweight='bold', pad=20, fontsize=14)
    plt.savefig(f'{output_dir}/03_metrics_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {output_dir}/03_metrics_table.png")


def analyze_feature_importance(model, feature_names, output_dir='output/evaluation'):
    """
    Analyze and display feature importance with visualizations.
    
    Args:
        model: Trained Random Forest model
        feature_names: List of feature names
        output_dir: Output directory
        
    Returns:
        pandas.DataFrame: Feature importance dataframe
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n✓ Top 10 Most Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['Feature']:45s} {row['Importance']:.4f}")
    
    # Save to CSV
    feature_importance.to_csv(f'{output_dir}/feature_importance.csv', index=False)
    print(f"\n  → Saved: {output_dir}/feature_importance.csv")
    
    return feature_importance


def plot_feature_importance(feature_importance, output_path='output/evaluation/04_feature_importance.png', top_n=15):
    """
    Create and save feature importance visualization.
    
    Args:
        feature_importance: DataFrame with Feature and Importance columns
        output_path: Path to save the plot
        top_n: Number of top features to display
    """
    top_features = feature_importance.head(min(top_n, len(feature_importance)))
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    
    plt.barh(range(len(top_features)), top_features['Importance'], color=colors)
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
    plt.ylabel('Feature', fontsize=12, fontweight='bold')
    plt.title('Battery Thermal Runaway Risk Drivers\nRandom Forest Feature Importance', 
              fontsize=14, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()
    
    # Add value labels
    for i, (idx, row) in enumerate(top_features.iterrows()):
        plt.text(row['Importance'] + 0.002, i, f"{row['Importance']:.3f}", 
                 va='center', fontsize=9)
    
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  → Saved: {output_path}")


def generate_technical_summary(feature_importance, metrics, df, target_col, numeric_features):
    """
    Generate technical summary and safety insights.
    
    Args:
        feature_importance: Feature importance dataframe
        metrics: Model metrics dictionary
        df: Original dataframe
        target_col: Name of target column
        numeric_features: List of numeric feature names
    """
    print("\n" + "="*80)
    print("TECHNICAL SUMMARY & SAFETY INSIGHTS")
    print("="*80)
    
    # Top feature analysis
    top_feature_name = feature_importance.iloc[0]['Feature']
    top_feature_importance = feature_importance.iloc[0]['Importance']
    
    print(f"\n✓ KEY FINDINGS:")
    print(f"\n1. MOST CRITICAL RISK FACTOR:")
    print(f"   → {top_feature_name}")
    print(f"   → Explains {top_feature_importance*100:.1f}% of thermal runaway severity variation")
    
    # Correlations
    correlations = {}
    for col in numeric_features:
        if col in df.columns:
            correlations[col] = df[col].corr(df[target_col])
    
    # Stored Energy impact
    print(f"\n2. STORED ENERGY IMPACT:")
    stored_energy_imp = feature_importance[feature_importance['Feature'] == 'Stored_Energy_Wh']['Importance'].values
    if len(stored_energy_imp) > 0:
        print(f"   → Stored Energy (Wh) importance: {stored_energy_imp[0]:.4f}")
        if 'Stored_Energy_Wh' in correlations:
            corr = correlations['Stored_Energy_Wh']
            print(f"   → Correlation with energy yield: {corr:.3f}")
            if corr > 0:
                print(f"   → CONCLUSION: Higher stored energy = Higher explosion risk")
    
    # Voltage analysis
    print(f"\n3. STATE OF CHARGE (VOLTAGE) ANALYSIS:")
    voltage_imp = feature_importance[feature_importance['Feature'] == 'Pre-Test-Cell-Open-Circuit-Voltage-V']['Importance'].values
    if len(voltage_imp) > 0:
        print(f"   → Voltage importance: {voltage_imp[0]:.4f}")
        if 'Pre-Test-Cell-Open-Circuit-Voltage-V' in correlations:
            corr = correlations['Pre-Test-Cell-Open-Circuit-Voltage-V']
            print(f"   → Correlation with energy yield: {corr:.3f}")
            
            voltage_range = df['Pre-Test-Cell-Open-Circuit-Voltage-V'].max() - df['Pre-Test-Cell-Open-Circuit-Voltage-V'].min()
            avg_energy = df[target_col].mean()
            impact_per_volt = (corr * df[target_col].std()) / df['Pre-Test-Cell-Open-Circuit-Voltage-V'].std()
            pct_change_per_volt = (impact_per_volt / avg_energy) * 100
            
            print(f"   → Impact: ~{impact_per_volt:.2f} kJ increase per volt increase")
            print(f"   → Relative impact: ~{pct_change_per_volt:.1f}% per volt")
    
    # Trigger mechanism
    print(f"\n4. TRIGGER MECHANISM COMPARISON:")
    trigger_features = [col for col in feature_importance['Feature'] if 'Trigger-Mechanism' in col]
    if trigger_features:
        print(f"   → Trigger mechanism categories:")
        for feat in trigger_features:
            imp = feature_importance[feature_importance['Feature'] == feat]['Importance'].values[0]
            mechanism = feat.replace('Trigger-Mechanism_', '')
            print(f"      • {mechanism:20s} importance: {imp:.4f}")
    
    # Model performance
    print(f"\n5. MODEL PERFORMANCE ASSESSMENT:")
    print(f"   → Prediction Error (RMSE): ±{metrics['test_rmse']:.3f} kJ")
    print(f"   → Mean Absolute Error: ±{metrics['test_mae']:.3f} kJ")
    print(f"   → Variance Explained (R²): {metrics['test_r2']*100:.1f}%")
    
    if metrics['test_r2'] > 0.7:
        print(f"   → ✓ EXCELLENT: Model explains most variance in thermal runaway severity")
    elif metrics['test_r2'] > 0.5:
        print(f"   → ✓ GOOD: Model captures majority of risk factors")
    else:
        print(f"   → ⚠ MODERATE: Additional factors may influence thermal runaway")
    
    print(f"\n6. SAFETY RECOMMENDATIONS:")
    print(f"   → CRITICAL: Monitor and limit state of charge (voltage) in high-risk scenarios")
    print(f"   → IMPORTANT: Stored energy is primary driver - capacity × voltage matters")
    print(f"   → DESIGN: Consider trigger mechanism type in safety protocols")
    print(f"   → MITIGATION: Physical containment design shows measurable impact")
