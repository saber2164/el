"""
Data Loading and Cleaning Module
==================================
Handles loading the NREL Battery Failure Databank and cleaning operations.
Includes visualizations for data quality assessment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def load_dataset(filepath, sheet_name='Battery Failure Databank'):
    """
    Load the NREL Battery Failure Databank from Excel file.
    
    Args:
        filepath: Path to the Excel file
        sheet_name: Name of the sheet to read
        
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    print("="*80)
    print("DATA LOADING")
    print("="*80)
    
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    print(f"✓ Dataset loaded successfully")
    print(f"  - Initial shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    return df


def clean_cell_description(df):
    """
    Clean the Cell-Description column by stripping whitespace.
    
    Args:
        df: Input dataframe
        
    Returns:
        pandas.DataFrame: DataFrame with cleaned Cell-Description
    """
    df['Cell-Description'] = df['Cell-Description'].str.strip()
    print(f"✓ Cleaned Cell-Description column")
    return df


def convert_data_types(df):
    """
    Convert object columns to appropriate numeric types.
    
    Args:
        df: Input dataframe
        
    Returns:
        pandas.DataFrame: DataFrame with converted types
    """
    print("\n" + "="*80)
    print("DATA TYPE CONVERSION")
    print("="*80)
    
    # Convert voltage to numeric
    df['Pre-Test-Cell-Open-Circuit-Voltage-V'] = pd.to_numeric(
        df['Pre-Test-Cell-Open-Circuit-Voltage-V'], errors='coerce'
    )
    
    # Convert casing thickness to numeric (handle '-' placeholder)
    df['Cell-Casing-Thickness-µm'] = pd.to_numeric(
        df['Cell-Casing-Thickness-µm'].replace('-', np.nan), errors='coerce'
    )
    
    print(f"✓ Converted voltage and thickness columns to numeric")
    
    return df


def filter_target_values(df, target_col='Corrected-Total-Energy-Yield-kJ'):
    """
    Filter out rows with missing or zero target values.
    
    Args:
        df: Input dataframe
        target_col: Name of the target column
        
    Returns:
        pandas.DataFrame: Filtered dataframe
    """
    initial_count = len(df)
    df = df[df[target_col].notna() & (df[target_col] > 0)]
    filtered_count = len(df)
    
    print(f"\n✓ Filtered dataset:")
    print(f"  - Rows before: {initial_count}")
    print(f"  - Rows after: {filtered_count}")
    print(f"  - Removed: {initial_count - filtered_count} rows (missing/zero target)")
    
    return df


def display_target_statistics(df, target_col='Corrected-Total-Energy-Yield-kJ'):
    """
    Display statistics for the target variable.
    
    Args:
        df: Input dataframe
        target_col: Name of the target column
    """
    print(f"\n✓ Target Variable Statistics ({target_col}):")
    print(f"  - Range: [{df[target_col].min():.2f}, {df[target_col].max():.2f}] kJ")
    print(f"  - Mean: {df[target_col].mean():.2f} kJ")
    print(f"  - Median: {df[target_col].median():.2f} kJ")
    print(f"  - Std Dev: {df[target_col].std():.2f} kJ")


def handle_missing_values(df, numeric_features):
    """
    Fill missing values in numeric features with median.
    
    Args:
        df: Input dataframe
        numeric_features: List of numeric feature column names
        
    Returns:
        pandas.DataFrame: DataFrame with imputed values
    """
    print(f"\n✓ Handling missing values:")
    
    for col in numeric_features:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  → Filled missing values in {col} with median: {median_val:.2f}")
    
    return df


def visualize_data_quality(df, output_dir='output/data_cleaning'):
    """
    Create visualizations for data quality assessment.
    
    Args:
        df: Input dataframe
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Missing values heatmap
    plt.figure(figsize=(14, 8))
    missing_data = df.isnull().sum().sort_values(ascending=False)
    missing_pct = (missing_data / len(df)) * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Missing values bar chart
    top_missing = missing_pct[missing_pct > 0].head(20)
    if len(top_missing) > 0:
        axes[0].barh(range(len(top_missing)), top_missing.values, color='coral')
        axes[0].set_yticks(range(len(top_missing)))
        axes[0].set_yticklabels(top_missing.index, fontsize=9)
        axes[0].set_xlabel('Missing Percentage (%)', fontweight='bold')
        axes[0].set_title('Top 20 Columns with Missing Values', fontweight='bold', pad=15)
        axes[0].grid(axis='x', alpha=0.3)
        axes[0].invert_yaxis()
    else:
        axes[0].text(0.5, 0.5, 'No Missing Values Found', 
                    ha='center', va='center', fontsize=14)
        axes[0].set_title('Missing Values Analysis', fontweight='bold')
    
    # Data types distribution
    dtype_counts = df.dtypes.value_counts()
    colors = plt.cm.Set3(range(len(dtype_counts)))
    axes[1].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
    axes[1].set_title('Data Types Distribution', fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_data_quality_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {output_dir}/01_data_quality_overview.png")
    
    # 2. Target variable distribution
    target_col = 'Corrected-Total-Energy-Yield-kJ'
    if target_col in df.columns:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Histogram
        axes[0].hist(df[target_col].dropna(), bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Energy Yield (kJ)', fontweight='bold')
        axes[0].set_ylabel('Frequency', fontweight='bold')
        axes[0].set_title('Target Variable Distribution', fontweight='bold', pad=15)
        axes[0].axvline(df[target_col].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df[target_col].mean():.2f} kJ', linewidth=2)
        axes[0].axvline(df[target_col].median(), color='orange', linestyle='--', 
                       label=f'Median: {df[target_col].median():.2f} kJ', linewidth=2)
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Box plot
        axes[1].boxplot(df[target_col].dropna(), vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7))
        axes[1].set_ylabel('Energy Yield (kJ)', fontweight='bold')
        axes[1].set_title('Target Variable Box Plot', fontweight='bold', pad=15)
        axes[1].grid(alpha=0.3)
        
        # Q-Q plot for normality
        from scipy import stats
        stats.probplot(df[target_col].dropna(), dist="norm", plot=axes[2])
        axes[2].set_title('Q-Q Plot (Normality Check)', fontweight='bold', pad=15)
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/02_target_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  → Saved: {output_dir}/02_target_distribution.png")


def create_data_summary_table(df, target_col='Corrected-Total-Energy-Yield-kJ', output_dir='output/data_cleaning'):
    """
    Create summary statistics table.
    
    Args:
        df: Input dataframe
        target_col: Name of target column
        output_dir: Directory to save table
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Summary statistics
    summary_stats = pd.DataFrame({
        'Metric': ['Count', 'Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max'],
        'Value': [
            df[target_col].count(),
            df[target_col].mean(),
            df[target_col].std(),
            df[target_col].min(),
            df[target_col].quantile(0.25),
            df[target_col].median(),
            df[target_col].quantile(0.75),
            df[target_col].max()
        ]
    })
    summary_stats['Value'] = summary_stats['Value'].round(3)
    
    # Save to CSV
    summary_stats.to_csv(f'{output_dir}/target_statistics.csv', index=False)
    print(f"  → Saved: {output_dir}/target_statistics.csv")
    
    # Create visualization of summary table
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary_stats.values, colLabels=summary_stats.columns,
                    cellLoc='center', loc='center',
                    colColours=['lightblue']*2)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    plt.title(f'{target_col} - Summary Statistics', fontweight='bold', pad=20, fontsize=14)
    plt.savefig(f'{output_dir}/03_summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {output_dir}/03_summary_table.png")


def clean_data(filepath, sheet_name='Battery Failure Databank', visualize=True):
    """
    Complete data cleaning pipeline with optional visualizations.
    
    Args:
        filepath: Path to the Excel file
        sheet_name: Name of the sheet to read
        visualize: Whether to create visualizations
        
    Returns:
        pandas.DataFrame: Cleaned dataset
    """
    # Load data
    df = load_dataset(filepath, sheet_name)
    
    # Visualize initial data quality
    if visualize:
        print("\n✓ Creating data quality visualizations...")
        visualize_data_quality(df)
    
    # Clean cell description
    df = clean_cell_description(df)
    
    # Convert data types
    df = convert_data_types(df)
    
    # Filter target values
    df = filter_target_values(df)
    
    # Display statistics
    display_target_statistics(df)
    
    # Create summary table
    if visualize:
        create_data_summary_table(df)
    
    return df
