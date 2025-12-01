"""
Model Training Module
=====================
Handles train/test split and Random Forest model training.
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("\n" + "="*80)
    print("TRAIN/TEST SPLIT")
    print("="*80)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"✓ Split completed ({int((1-test_size)*100)}/{int(test_size*100)}):")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Testing samples: {len(X_test)}")
    print(f"  - Training target range: [{y_train.min():.2f}, {y_train.max():.2f}] kJ")
    print(f"  - Testing target range: [{y_test.min():.2f}, {y_test.max():.2f}] kJ")
    
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=8, 
                       min_samples_split=10, min_samples_leaf=4, random_state=42):
    """
    Train Random Forest model with overfitting prevention.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples to split node
        min_samples_leaf: Minimum samples per leaf
        random_state: Random seed
        
    Returns:
        RandomForestRegressor: Trained model
    """
    print("\n" + "="*80)
    print("RANDOM FOREST MODEL TRAINING")
    print("="*80)
    
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    
    print(f"✓ Random Forest Configuration:")
    print(f"  - n_estimators: {n_estimators}")
    print(f"  - max_depth: {max_depth} (overfitting prevention)")
    print(f"  - min_samples_split: {min_samples_split}")
    print(f"  - min_samples_leaf: {min_samples_leaf}")
    
    print(f"\n⏳ Training model...")
    rf_model.fit(X_train, y_train)
    print(f"✓ Training completed!")
    
    return rf_model
