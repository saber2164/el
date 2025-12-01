"""
Battery Safety Optimizer
=========================
Calculates maximum safe operating voltage for Phase 0/1 batteries
to keep thermal runaway risk below specified thresholds.

For BMS integration and safety protocol enforcement.
"""

import joblib
import numpy as np
import pandas as pd
import os
from pathlib import Path


class PhaseSafetyOptimizer:
    """
    Safety optimizer for Phase 0 and Phase 1 batteries.
    
    Calculates maximum safe voltage limits based on:
    - Fixed hardware parameters (18650, 30µm casing, Nail trigger)
    - Battery capacity (SOH-adjusted)
    - Risk threshold (default: 40 kJ)
    """
    
    # Phase 0/1 Fixed Hardware Parameters (Cannot be changed)
    CELL_FORMAT = '18650'
    CASING_THICKNESS_UM = 30  # µm
    TRIGGER_MECHANISM = 'Nail'  # Worst-case scenario
    
    # Risk thresholds (kJ)
    RISK_THRESHOLDS = {
        'safe': 30.0,      # Green zone
        'warning': 50.0,   # Yellow zone
        'critical': 70.0   # Red zone
    }
    
    def __init__(self, model_dir='models'):
        """
        Initialize optimizer by loading trained model.
        
        Args:
            model_dir: Directory containing saved model files
        """
        self.model_dir = Path(model_dir)
        self._load_model()
        
    def _load_model(self):
        """Load trained Random Forest model and metadata."""
        model_path = self.model_dir / 'rf_model.joblib'
        metadata_path = self.model_dir / 'model_metadata.joblib'
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Please run thermal_prediction/main.py first to train and save the model."
            )
        
        self.model = joblib.load(model_path)
        self.metadata = joblib.load(metadata_path)
        self.feature_names = self.metadata['feature_names']
        
        print(f"✓ Loaded model from {model_path}")
        print(f"  - Features: {len(self.feature_names)}")
        print(f"  - Test R²: {self.metadata['test_r2']:.3f}")
        print(f"  - Test RMSE: {self.metadata['test_rmse']:.3f} kJ")
    
    def _create_feature_vector(self, voltage_v, capacity_ah):
        """
        Create feature vector for model prediction.
        
        Args:
            voltage_v: Battery voltage (V)
            capacity_ah: Battery capacity (Ah)
            
        Returns:
            pandas.DataFrame: Feature vector matching model's expected input
        """
        # Calculate stored energy
        stored_energy_wh = capacity_ah * voltage_v
        
        # Create base features
        features = {
            'Cell-Capacity-Ah': capacity_ah,
            'Pre-Test-Cell-Open-Circuit-Voltage-V': voltage_v,
            'Cell-Casing-Thickness-µm': self.CASING_THICKNESS_UM,
            'Stored_Energy_Wh': stored_energy_wh,
        }
        
        # One-hot encoding for Cell-Format
        features['Cell-Format_18650'] = 1 if self.CELL_FORMAT == '18650' else 0
        features['Cell-Format_21700'] = 1 if self.CELL_FORMAT == '21700' else 0
        features['Cell-Format_D-Cell'] = 1 if self.CELL_FORMAT == 'D-Cell' else 0
        
        # One-hot encoding for Trigger-Mechanism
        features['Trigger-Mechanism_Heater (ISC)'] = 1 if self.TRIGGER_MECHANISM == 'Heater (ISC)' else 0
        features['Trigger-Mechanism_Heater (Non-ISC)'] = 1 if self.TRIGGER_MECHANISM == 'Heater (Non-ISC)' else 0
        features['Trigger-Mechanism_Nail'] = 1 if self.TRIGGER_MECHANISM == 'Nail' else 0
        
        # Create DataFrame with correct feature order
        df = pd.DataFrame([features])
        df = df[self.feature_names]  # Ensure correct order
        
        return df
    
    def predict_risk(self, voltage_v, capacity_ah):
        """
        Predict thermal runaway severity for given battery state.
        
        Args:
            voltage_v: Battery voltage (V)
            capacity_ah: Battery capacity (Ah)
            
        Returns:
            float: Predicted energy yield (kJ)
        """
        features = self._create_feature_vector(voltage_v, capacity_ah)
        prediction = self.model.predict(features)[0]
        return max(0, prediction)  # Ensure non-negative
    
    def get_max_safe_voltage(self, capacity_ah, max_risk_kj=40.0, 
                            voltage_min=2.5, voltage_max=4.5, tolerance=0.01):
        """
        Calculate maximum safe operating voltage to stay below risk threshold.
        
        Uses binary search to find the voltage that keeps predicted risk
        just below the maximum allowable threshold.
        
        Args:
            capacity_ah: Battery capacity (Ah)
            max_risk_kj: Maximum allowable risk threshold (kJ)
            voltage_min: Minimum search voltage (V)
            voltage_max: Maximum search voltage (V)
            tolerance: Search tolerance (V)
            
        Returns:
            float: Maximum safe voltage (V)
        """
        # Binary search for max safe voltage
        low, high = voltage_min, voltage_max
        max_safe_v = voltage_min
        
        while (high - low) > tolerance:
            mid = (low + high) / 2
            risk = self.predict_risk(mid, capacity_ah)
            
            if risk <= max_risk_kj:
                max_safe_v = mid
                low = mid  # Can go higher
            else:
                high = mid  # Too risky, go lower
        
        return round(max_safe_v, 2)
    
    def get_risk_level(self, predicted_risk_kj):
        """
        Categorize risk level based on thresholds.
        
        Args:
            predicted_risk_kj: Predicted thermal runaway energy (kJ)
            
        Returns:
            tuple: (level_name, color_code)
        """
        if predicted_risk_kj < self.RISK_THRESHOLDS['safe']:
            return 'SAFE', 'green'
        elif predicted_risk_kj < self.RISK_THRESHOLDS['warning']:
            return 'WARNING', 'yellow'
        elif predicted_risk_kj < self.RISK_THRESHOLDS['critical']:
            return 'CRITICAL', 'orange'
        else:
            return 'EXTREME', 'red'
    
    def generate_safety_boundary(self, capacity_range=(1.0, 5.0), 
                                 max_risk_kj=40.0, num_points=50):
        """
        Generate voltage vs capacity safety boundary curve.
        
        Args:
            capacity_range: (min, max) capacity range (Ah)
            max_risk_kj: Risk threshold for boundary (kJ)
            num_points: Number of points in curve
            
        Returns:
            tuple: (capacities, max_voltages) arrays for plotting
        """
        capacities = np.linspace(capacity_range[0], capacity_range[1], num_points)
        max_voltages = np.array([
            self.get_max_safe_voltage(cap, max_risk_kj) 
            for cap in capacities
        ])
        
        return capacities, max_voltages
    
    def get_phase_info(self, phase):
        """
        Get information about battery phase.
        
        Args:
            phase: 'Phase 0 Prototype' or 'Phase 1 Production'
            
        Returns:
            dict: Phase specifications
        """
        return {
            'Cell Format': self.CELL_FORMAT,
            'Casing Thickness': f'{self.CASING_THICKNESS_UM} µm',
            'Trigger Mode': f'{self.TRIGGER_MECHANISM} (Worst-case)',
            'Phase': phase,
            'Status': 'Prototype' if 'Phase 0' in phase else 'Production'
        }


# Example usage and testing
if __name__ == '__main__':
    print("="*80)
    print("BATTERY SAFETY OPTIMIZER - TEST")
    print("="*80)
    
    # Initialize optimizer
    optimizer = PhaseSafetyOptimizer()
    
    # Test scenarios
    test_cases = [
        {'voltage': 4.2, 'capacity': 3.0, 'name': 'Fully Charged 3Ah'},
        {'voltage': 3.7, 'capacity': 3.0, 'name': 'Nominal 3Ah'},
        {'voltage': 4.2, 'capacity': 2.0, 'name': 'Fully Charged 2Ah (Degraded)'},
    ]
    
    print(f"\n{'Scenario':<25} {'Voltage':<10} {'Capacity':<12} {'Risk':<12} {'Level':<10}")
    print("-"*80)
    
    for case in test_cases:
        risk = optimizer.predict_risk(case['voltage'], case['capacity'])
        level, color = optimizer.get_risk_level(risk)
        print(f"{case['name']:<25} {case['voltage']:<10.2f} {case['capacity']:<12.2f} {risk:<12.2f} {level:<10}")
    
    # Test max safe voltage calculation
    print(f"\n{'Capacity (Ah)':<15} {'Max Safe V':<15} {'Risk @ Max V':<15}")
    print("-"*50)
    
    for capacity in [2.0, 3.0, 4.0, 5.0]:
        max_v = optimizer.get_max_safe_voltage(capacity, max_risk_kj=40.0)
        risk_at_max = optimizer.predict_risk(max_v, capacity)
        print(f"{capacity:<15.1f} {max_v:<15.2f} {risk_at_max:<15.2f}")
    
    print("\n" + "="*80)
