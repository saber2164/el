"""
Feature Engineering Module
===========================
Handles feature creation and encoding for the battery thermal runaway model.
Includes visualizations for feature analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def create_stored_energy_feature(df):
    """
    Create Stored Energy feature based on physics (E = Capacity × Voltage).
    
    Args:
        df: Input dataframe
        
    Returns:
        pandas.DataFrame: DataFrame with new Stored_Energy_Wh feature
    """
    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)
    
    df['Stored_Energy_Wh'] = df['Cell-Capacity-Ah'] * df['Pre-Test-Cell-Open-Circuit-Voltage-V']
    
    print(f"✓ Created feature: Stored_Energy_Wh = Cell-Capacity-Ah × Voltage")
    print(f"  - Range: [{df['Stored_Energy_Wh'].min():.2f}, {df['Stored_Energy_Wh'].max():.2f}] Wh")
    
    return df


def check_feature_completeness(df, feature_cols):
    """
    Check for missing values in feature columns.
    
    Args:
        df: Input dataframe
        feature_cols: List of feature column names
    """
    print(f"\n✓ Feature completeness check:")
    for col in feature_cols:
        missing = df[col].isna().sum()
        missing_pct = (missing / len(df)) * 100
        print(f"  - {col}: {missing} missing ({missing_pct:.1f}%)")


def encode_categorical_features(df, feature_cols):
    """
    One-hot encode categorical features.
    
    Args:
        df: Input dataframe
        feature_cols: List of all feature column names (including categorical)
        
    Returns:
        pandas.DataFrame: Encoded feature dataframe
    """
    print(f"\n✓ One-Hot Encoding categorical features:")
    
    # Identify categorical columns
    categorical_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols:
        print(f"  - {col}: {df[col].nunique()} categories {df[col].unique().tolist()}")
    
    # Create dummy variables
    df_encoded = pd.get_dummies(df[feature_cols], 
                                columns=categorical_cols,
                                drop_first=False)
    
    print(f"\n✓ Final feature set: {df_encoded.shape[1]} features")
    print(f"  Features: {list(df_encoded.columns)}")
    
    return df_encoded


def visualize_features(df, feature_cols, target_col='Corrected-Total-Energy-Yield-kJ', output_dir='output/feature_engineering'):
    """
    Create visualizations for feature analysis.
    
    Args:
        df: Input dataframe with features
        feature_cols: List of feature column names
        target_col: Target column name
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get numeric features only
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    # 1. Correlation matrix
    if len(numeric_features) > 0:
        corr_df = df[numeric_features + [target_col]].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_df, dtype=bool))
        sns.heatmap(corr_df, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix', fontweight='bold', pad=20, fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/01_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  → Saved: {output_dir}/01_correlation_matrix.png")
    
    # 2. Feature distributions
    n_features = len(numeric_features)
    if n_features > 0:
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, feat in enumerate(numeric_features):
            axes[idx].hist(df[feat].dropna(), bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            axes[idx].set_xlabel(feat, fontsize=9, fontweight='bold')
            axes[idx].set_ylabel('Frequency', fontsize=9, fontweight='bold')
            axes[idx].set_title(f'Distribution of {feat}', fontsize=10, fontweight='bold')
            axes[idx].grid(alpha=0.3)
            
            # Add statistics
            mean_val = df[feat].mean()
            median_val = df[feat].median()
            axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            axes[idx].axvline(median_val, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/02_feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  → Saved: {output_dir}/02_feature_distributions.png")
    
    # 3. Feature vs Target scatter plots
    if len(numeric_features) > 0:
        n_cols = 2
        n_rows = (len(numeric_features) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
        axes = axes.flatten() if len(numeric_features) > 1 else [axes]
        
        for idx, feat in enumerate(numeric_features):
            axes[idx].scatter(df[feat], df[target_col], alpha=0.5, s=30, color='steelblue')
            axes[idx].set_xlabel(feat, fontsize=9, fontweight='bold')
            axes[idx].set_ylabel(target_col, fontsize=9, fontweight='bold')
            axes[idx].set_title(f'{feat} vs Target', fontsize=10, fontweight='bold')
            axes[idx].grid(alpha=0.3)
            
            # Add correlation value
            corr = df[[feat, target_col]].corr().iloc[0, 1]
            axes[idx].text(0.05, 0.95, f'Corr: {corr:.3f}', 
                          transform=axes[idx].transAxes, 
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                          verticalalignment='top')
        
        # Hide unused subplots
        for idx in range(len(numeric_features), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/03_feature_vs_target.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  → Saved: {output_dir}/03_feature_vs_target.png")


def create_feature_summary_table(df, feature_cols, output_dir='output/feature_engineering'):
    """
    Create summary table for features.
    
    Args:
        df: Input dataframe
        feature_cols: List of feature columns
        output_dir: Directory to save table
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get numeric features
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    # Create summary
    summary_data = []
    for feat in numeric_features:
        summary_data.append({
            'Feature': feat,
            'Count': df[feat].count(),
            'Mean': round(df[feat].mean(), 3),
            'Std': round(df[feat].std(), 3),
            'Min': round(df[feat].min(), 3),
            'Max': round(df[feat].max(), 3),
            'Missing': df[feat].isna().sum()
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'{output_dir}/feature_summary.csv', index=False)
    print(f"  → Saved: {output_dir}/feature_summary.csv")


def engineer_features(df, visualize=True):
    """
    Complete feature engineering pipeline with optional visualizations.
    
    Args:
        df: Input dataframe
        
    Returns:
        tuple: (encoded_features_df, feature_column_list)
    """
    # Define feature columns
    feature_cols = [
        'Cell-Capacity-Ah',
        'Pre-Test-Cell-Open-Circuit-Voltage-V',
        'Cell-Casing-Thickness-µm',
        'Stored_Energy_Wh',
        'Cell-Format',
        'Trigger-Mechanism'
    ]
    
    # Create stored energy feature
    df = create_stored_energy_feature(df)
    
    # Check completeness
    check_feature_completeness(df, feature_cols)
    
    # Handle missing values in numeric features
    numeric_features = ['Cell-Capacity-Ah', 'Pre-Test-Cell-Open-Circuit-Voltage-V', 
                       'Cell-Casing-Thickness-µm', 'Stored_Energy_Wh']
    
    from data_cleaning import handle_missing_values
    df = handle_missing_values(df, numeric_features)
    
    # Encode categorical features
    df_encoded = encode_categorical_features(df, feature_cols)
    
    # Create visualizations
    if visualize:
        print("\n✓ Creating feature engineering visualizations...")
        visualize_features(df, feature_cols)
        create_feature_summary_table(df, feature_cols)
    
    return df_encoded, feature_cols
