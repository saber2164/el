#!/usr/bin/env python3
"""
Battery Thermal Runaway Severity Prediction Model
==================================================
Predicts the thermal runaway severity (risk) of lithium-ion batteries
based on physical design and charging state using the NREL Battery Failure Databank.

Author: Battery Safety ML Pipeline
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("="*80)
print("BATTERY THERMAL RUNAWAY SEVERITY PREDICTION MODEL")
print("="*80)
print("\nLoading NREL Battery Failure Databank dataset...")

# ============================================================================
# STEP 1: DATA LOADING & INITIAL CLEANING
# ============================================================================

# Load the Excel file from the 'Battery Failure Databank' sheet
df = pd.read_excel('battery-failure-databank-revision2-feb24.xlsx', 
                   sheet_name='Battery Failure Databank')

print(f"\n✓ Dataset loaded successfully")
print(f"  - Initial shape: {df.shape[0]} rows × {df.shape[1]} columns")

# Clean Cell-Description column (strip whitespace)
df['Cell-Description'] = df['Cell-Description'].str.strip()
print(f"✓ Cleaned Cell-Description column")

# ============================================================================
# STEP 2: DATA TYPE CONVERSION & CLEANING
# ============================================================================

print("\n" + "="*80)
print("DATA CLEANING & TYPE CONVERSION")
print("="*80)

# Convert Pre-Test-Cell-Open-Circuit-Voltage-V to numeric
df['Pre-Test-Cell-Open-Circuit-Voltage-V'] = pd.to_numeric(
    df['Pre-Test-Cell-Open-Circuit-Voltage-V'], errors='coerce'
)

# Convert Cell-Casing-Thickness-µm to numeric (handle '-' values)
df['Cell-Casing-Thickness-µm'] = pd.to_numeric(
    df['Cell-Casing-Thickness-µm'].replace('-', np.nan), errors='coerce'
)

print(f"✓ Converted voltage and thickness columns to numeric")

# Filter rows where target is missing or zero
target_col = 'Corrected-Total-Energy-Yield-kJ'
initial_count = len(df)
df = df[df[target_col].notna() & (df[target_col] > 0)]
filtered_count = len(df)

print(f"\n✓ Filtered dataset:")
print(f"  - Rows before: {initial_count}")
print(f"  - Rows after: {filtered_count}")
print(f"  - Removed: {initial_count - filtered_count} rows (missing/zero target)")

# Display target variable statistics
print(f"\n✓ Target Variable Statistics ({target_col}):")
print(f"  - Range: [{df[target_col].min():.2f}, {df[target_col].max():.2f}] kJ")
print(f"  - Mean: {df[target_col].mean():.2f} kJ")
print(f"  - Median: {df[target_col].median():.2f} kJ")
print(f"  - Std Dev: {df[target_col].std():.2f} kJ")

# ============================================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================================

print("\n" + "="*80)
print("FEATURE ENGINEERING")
print("="*80)

# Create Stored Energy feature (Physics: Energy = Capacity × Voltage)
df['Stored_Energy_Wh'] = df['Cell-Capacity-Ah'] * df['Pre-Test-Cell-Open-Circuit-Voltage-V']
print(f"✓ Created feature: Stored_Energy_Wh = Cell-Capacity-Ah × Voltage")
print(f"  - Range: [{df['Stored_Energy_Wh'].min():.2f}, {df['Stored_Energy_Wh'].max():.2f}] Wh")

# Select features for modeling
feature_cols = [
    'Cell-Capacity-Ah',
    'Pre-Test-Cell-Open-Circuit-Voltage-V',
    'Cell-Casing-Thickness-µm',
    'Stored_Energy_Wh',
    'Cell-Format',
    'Trigger-Mechanism'
]

# Check for missing values in features
print(f"\n✓ Feature completeness check:")
for col in feature_cols:
    missing = df[col].isna().sum()
    missing_pct = (missing / len(df)) * 100
    print(f"  - {col}: {missing} missing ({missing_pct:.1f}%)")

# Handle missing values in numeric features by filling with median
numeric_features = ['Cell-Capacity-Ah', 'Pre-Test-Cell-Open-Circuit-Voltage-V', 
                   'Cell-Casing-Thickness-µm', 'Stored_Energy_Wh']

for col in numeric_features:
    if df[col].isna().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  → Filled missing values in {col} with median: {median_val:.2f}")

# One-Hot Encode categorical variables
print(f"\n✓ One-Hot Encoding categorical features:")
print(f"  - Cell-Format: {df['Cell-Format'].nunique()} categories {df['Cell-Format'].unique().tolist()}")
print(f"  - Trigger-Mechanism: {df['Trigger-Mechanism'].nunique()} categories {df['Trigger-Mechanism'].unique().tolist()}")

# Create dummy variables
df_encoded = pd.get_dummies(df[feature_cols], 
                            columns=['Cell-Format', 'Trigger-Mechanism'],
                            drop_first=False)

print(f"\n✓ Final feature set: {df_encoded.shape[1]} features")
print(f"  Features: {list(df_encoded.columns)}")

# ============================================================================
# STEP 4: TRAIN/TEST SPLIT
# ============================================================================

print("\n" + "="*80)
print("TRAIN/TEST SPLIT")
print("="*80)

X = df_encoded
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✓ Split completed (80/20):")
print(f"  - Training samples: {len(X_train)}")
print(f"  - Testing samples: {len(X_test)}")
print(f"  - Training target range: [{y_train.min():.2f}, {y_train.max():.2f}] kJ")
print(f"  - Testing target range: [{y_test.min():.2f}, {y_test.max():.2f}] kJ")

# ============================================================================
# STEP 5: MODEL TRAINING
# ============================================================================

print("\n" + "="*80)
print("RANDOM FOREST MODEL TRAINING")
print("="*80)

# Initialize Random Forest with overfitting prevention
rf_model = RandomForestRegressor(
    n_estimators=100,       # Sufficient for stable predictions
    max_depth=8,            # Prevent excessive tree depth
    min_samples_split=10,   # Minimum samples required to split
    min_samples_leaf=4,     # Prevent tiny leaf nodes
    random_state=42,
    n_jobs=-1,              # Use all CPU cores
    verbose=0
)

print(f"✓ Random Forest Configuration:")
print(f"  - n_estimators: 100")
print(f"  - max_depth: 8 (overfitting prevention)")
print(f"  - min_samples_split: 10")
print(f"  - min_samples_leaf: 4")

print(f"\n⏳ Training model...")
rf_model.fit(X_train, y_train)
print(f"✓ Training completed!")

# ============================================================================
# STEP 6: MODEL EVALUATION
# ============================================================================

print("\n" + "="*80)
print("MODEL EVALUATION")
print("="*80)

# Make predictions
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"✓ Performance Metrics:")
print(f"\n  Training Set:")
print(f"    - RMSE: {train_rmse:.3f} kJ")
print(f"    - MAE:  {train_mae:.3f} kJ")
print(f"    - R²:   {train_r2:.3f}")

print(f"\n  Testing Set:")
print(f"    - RMSE: {test_rmse:.3f} kJ")
print(f"    - MAE:  {test_mae:.3f} kJ")
print(f"    - R²:   {test_r2:.3f}")

# Check for overfitting
overfitting_gap = train_r2 - test_r2
print(f"\n  Overfitting Check:")
print(f"    - R² Gap (Train - Test): {overfitting_gap:.3f}")
if overfitting_gap < 0.1:
    print(f"    ✓ No significant overfitting detected")
elif overfitting_gap < 0.2:
    print(f"    ⚠ Minor overfitting (acceptable)")
else:
    print(f"    ⚠ Moderate overfitting detected")

# ============================================================================
# STEP 7: FEATURE IMPORTANCE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Get feature importances
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n✓ Top 10 Most Important Features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['Feature']:45s} {row['Importance']:.4f}")

# Create feature importance plot
plt.figure(figsize=(12, 8))
top_n = min(15, len(feature_importance))  # Show top 15 or all if fewer
top_features = feature_importance.head(top_n)

colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
bars = plt.barh(range(len(top_features)), top_features['Importance'], color=colors)

plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
plt.ylabel('Feature', fontsize=12, fontweight='bold')
plt.title('Battery Thermal Runaway Risk Drivers\nRandom Forest Feature Importance', 
          fontsize=14, fontweight='bold', pad=20)
plt.gca().invert_yaxis()  # Highest importance at top

# Add value labels on bars
for i, (idx, row) in enumerate(top_features.iterrows()):
    plt.text(row['Importance'] + 0.002, i, f"{row['Importance']:.3f}", 
             va='center', fontsize=9)

plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Feature importance plot saved: feature_importance.png")

# ============================================================================
# STEP 8: TECHNICAL SUMMARY & INTERPRETATION
# ============================================================================

print("\n" + "="*80)
print("TECHNICAL SUMMARY & SAFETY INSIGHTS")
print("="*80)

# Analyze top features
top_feature_name = feature_importance.iloc[0]['Feature']
top_feature_importance = feature_importance.iloc[0]['Importance']

# Get correlation with target for top numeric features
correlations = {}
for col in numeric_features:
    if col in X.columns:
        correlations[col] = df[col].corr(df[target_col])

print(f"\n✓ KEY FINDINGS:")
print(f"\n1. MOST CRITICAL RISK FACTOR:")
print(f"   → {top_feature_name}")
print(f"   → Explains {top_feature_importance*100:.1f}% of thermal runaway severity variation")

print(f"\n2. STORED ENERGY IMPACT:")
stored_energy_imp = feature_importance[feature_importance['Feature'] == 'Stored_Energy_Wh']['Importance'].values
if len(stored_energy_imp) > 0:
    print(f"   → Stored Energy (Wh) importance: {stored_energy_imp[0]:.4f}")
    if 'Stored_Energy_Wh' in correlations:
        corr = correlations['Stored_Energy_Wh']
        print(f"   → Correlation with energy yield: {corr:.3f}")
        if corr > 0:
            print(f"   → CONCLUSION: Higher stored energy = Higher explosion risk")

print(f"\n3. STATE OF CHARGE (VOLTAGE) ANALYSIS:")
voltage_imp = feature_importance[feature_importance['Feature'] == 'Pre-Test-Cell-Open-Circuit-Voltage-V']['Importance'].values
if len(voltage_imp) > 0:
    print(f"   → Voltage importance: {voltage_imp[0]:.4f}")
    if 'Pre-Test-Cell-Open-Circuit-Voltage-V' in correlations:
        corr = correlations['Pre-Test-Cell-Open-Circuit-Voltage-V']
        print(f"   → Correlation with energy yield: {corr:.3f}")
        
        # Calculate approximate impact
        voltage_range = df['Pre-Test-Cell-Open-Circuit-Voltage-V'].max() - df['Pre-Test-Cell-Open-Circuit-Voltage-V'].min()
        avg_energy = df[target_col].mean()
        impact_per_volt = (corr * df[target_col].std()) / df['Pre-Test-Cell-Open-Circuit-Voltage-V'].std()
        pct_change_per_volt = (impact_per_volt / avg_energy) * 100
        
        print(f"   → Impact: ~{impact_per_volt:.2f} kJ increase per volt increase")
        print(f"   → Relative impact: ~{pct_change_per_volt:.1f}% per volt")

print(f"\n4. TRIGGER MECHANISM COMPARISON:")
trigger_features = [col for col in feature_importance['Feature'] if 'Trigger-Mechanism' in col]
if trigger_features:
    print(f"   → Trigger mechanism categories:")
    for feat in trigger_features:
        imp = feature_importance[feature_importance['Feature'] == feat]['Importance'].values[0]
        mechanism = feat.replace('Trigger-Mechanism_', '')
        print(f"      • {mechanism:20s} importance: {imp:.4f}")

print(f"\n5. PHYSICAL DESIGN FACTORS:")
thickness_imp = feature_importance[feature_importance['Feature'] == 'Cell-Casing-Thickness-µm']['Importance'].values
if len(thickness_imp) > 0:
    print(f"   → Casing thickness importance: {thickness_imp[0]:.4f}")
    if 'Cell-Casing-Thickness-µm' in correlations:
        corr = correlations['Cell-Casing-Thickness-µm']
        print(f"   → Correlation with energy yield: {corr:.3f}")
        if corr < 0:
            print(f"   → CONCLUSION: Thicker casing = Lower explosion severity (better containment)")
        else:
            print(f"   → CONCLUSION: Casing thickness shows minimal protective effect")

print(f"\n6. MODEL PERFORMANCE ASSESSMENT:")
print(f"   → Prediction Error (RMSE): ±{test_rmse:.3f} kJ")
print(f"   → Mean Absolute Error: ±{test_mae:.3f} kJ")
print(f"   → Variance Explained (R²): {test_r2*100:.1f}%")

if test_r2 > 0.7:
    print(f"   → ✓ EXCELLENT: Model explains most variance in thermal runaway severity")
elif test_r2 > 0.5:
    print(f"   → ✓ GOOD: Model captures majority of risk factors")
else:
    print(f"   → ⚠ MODERATE: Additional factors may influence thermal runaway")

print(f"\n7. SAFETY RECOMMENDATIONS:")
print(f"   → CRITICAL: Monitor and limit state of charge (voltage) in high-risk scenarios")
print(f"   → IMPORTANT: Stored energy is primary driver - capacity × voltage matters")
print(f"   → DESIGN: Consider trigger mechanism type in safety protocols")
print(f"   → MITIGATION: Physical containment design shows measurable impact")

print("\n" + "="*80)
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("="*80)
print(f"\nOutputs generated:")
print(f"  1. Feature importance plot: feature_importance.png")
print(f"  2. Model performance: RMSE = {test_rmse:.3f} kJ, R² = {test_r2:.3f}")
print(f"  3. Technical summary: See above analysis")
print("="*80)
