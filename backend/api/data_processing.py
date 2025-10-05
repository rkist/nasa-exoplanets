import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DataProcessor:
    def __init__(self):
        self.feature_columns = [
            'eff_temp', 'surface_gravity', 'metallicity', 'radius',
            'reddening', 'extinction', 'gkcolor', 'grcolor', 'jkcolor'
        ]
        self.label_encoder = LabelEncoder()
        
    def preprocess_single(self, features):
        """Preprocess a single data point"""
        # Create dataframe
        df = pd.DataFrame([features])
        
        # Fill missing values with median (or use 0 for now)
        df = df.fillna(0)
        
        # Ensure all required columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Select only feature columns
        df = df[self.feature_columns]
        
        return df.values
    
    def preprocess_batch(self, df):
        """Preprocess a batch of data"""
        # Fill missing values
        df = df.fillna(0)
        
        # Ensure all required columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Select only feature columns
        df_processed = df[self.feature_columns]
        
        return df_processed

    def preprocess_records(self, records):
        """Preprocess list of JSON-like records"""
        if not records:
            return pd.DataFrame(columns=self.feature_columns).values

        df = pd.DataFrame(records)
        processed = self.preprocess_batch(df)
        return processed.values

    def build_training_dataframe(self, dataset_name='combined'):
        """Return a dataframe with features and binary labels for training."""
        X, y = self.load_dataset(dataset_name)

        if X is None or y is None:
            return None

        df = pd.DataFrame(X, columns=self.feature_columns)
        labels = np.where(y == 0, 'Likely False Positive', 'Candidate')
        df['label'] = labels
        return df
    
    def load_dataset(self, dataset_name):
        """Load and preprocess dataset"""
        try:
            if dataset_name == 'kepler':
                return self._load_kepler_data()
            elif dataset_name == 'k2':
                return self._load_k2_data()
            elif dataset_name == 'tess':
                return self._load_tess_data()
            elif dataset_name == 'combined':
                return self._load_combined_data()
            else:
                # Generate synthetic data for demo
                return self._generate_synthetic_data()
        except Exception as e:
            print(f"Error loading dataset: {e}")
            # Fallback to synthetic data
            return self._generate_synthetic_data()
    
    def _load_kepler_data(self):
        """Load Kepler dataset"""
        # In production, you would load from NASA Exoplanet Archive
        # For now, generate synthetic data
        print("Loading Kepler dataset...")
        return self._generate_synthetic_data(5000, 'kepler')
    
    def _load_k2_data(self):
        """Load K2 dataset"""
        print("Loading K2 dataset...")
        return self._generate_synthetic_data(3000, 'k2')
    
    def _load_tess_data(self):
        """Load TESS dataset"""
        print("Loading TESS dataset...")
        return self._generate_synthetic_data(4000, 'tess')
    
    def _load_combined_data(self):
        """Load combined dataset"""
        print("Loading combined dataset...")
        return self._generate_synthetic_data(10000, 'combined')
    
    def _generate_synthetic_data(self, n_samples=5000, source='kepler'):
        """Generate synthetic exoplanet data for demonstration"""
        np.random.seed(42)
        
        # Generate features based on realistic exoplanet characteristics
        data = {
            'eff_temp': np.random.normal(5500, 900, n_samples),
            'surface_gravity': np.random.normal(4.3, 0.3, n_samples),
            'metallicity': np.random.normal(0.0, 0.2, n_samples),
            'radius': np.random.lognormal(np.log(1.8), 0.5, n_samples),
            'reddening': np.abs(np.random.normal(0.04, 0.02, n_samples)),
            'extinction': np.abs(np.random.normal(0.05, 0.03, n_samples)),
            'gkcolor': np.random.normal(1.5, 0.3, n_samples),
            'grcolor': np.random.normal(0.7, 0.2, n_samples),
            'jkcolor': np.random.normal(0.5, 0.1, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Ensure positive values
        for col in df.columns:
            if col in ['metallicity', 'gkcolor', 'grcolor', 'jkcolor', 'surface_gravity']:
                continue
            df[col] = df[col].abs()
        
        # Clip values to plausible astrophysical ranges
        df['metallicity'] = df['metallicity'].clip(-0.6, 0.6)
        df['eff_temp'] = df['eff_temp'].clip(3200, 7800)
        df['gkcolor'] = df['gkcolor'].clip(0.5, 2.5)
        df['extinction'] = df['extinction'].clip(0, 0.4)
        df['grcolor'] = df['grcolor'].clip(0.2, 1.2)
        df['radius'] = df['radius'].clip(0.5, 12)
        df['jkcolor'] = df['jkcolor'].clip(0.1, 1.0)
        df['surface_gravity'] = df['surface_gravity'].clip(3.0, 5.0)
        df['reddening'] = df['reddening'].clip(0, 0.3)
        
        # Generate labels based on some heuristic rules
        labels = []
        for _, row in df.iterrows():
            # Confirmed: reasonable planetary characteristics
            if (-0.2 <= row['metallicity'] <= 0.4 and
                4500 <= row['eff_temp'] <= 6500 and
                0.6 <= row['gkcolor'] <= 2.0 and
                row['extinction'] < 0.15 and
                0.4 <= row['grcolor'] <= 1.0 and
                0.8 <= row['radius'] <= 6 and
                0.3 <= row['jkcolor'] <= 0.8 and
                3.8 <= row['surface_gravity'] <= 4.6 and
                row['reddening'] < 0.12):
                label = 2  # Confirmed
            # Candidate: some unusual but possible characteristics
            elif (-0.4 <= row['metallicity'] <= 0.5 and
                  3800 <= row['eff_temp'] <= 7200 and
                  0.4 <= row['gkcolor'] <= 2.2 and
                  row['extinction'] < 0.25 and
                  0.3 <= row['grcolor'] <= 1.1 and
                  0.5 <= row['radius'] <= 9 and
                  0.2 <= row['jkcolor'] <= 0.9 and
                  3.5 <= row['surface_gravity'] <= 4.9 and
                  row['reddening'] < 0.2):
                label = 1  # Candidate
            # False positive: anomalous characteristics
            else:
                label = 0  # False Positive
        
            labels.append(label)
        
        # Add some randomness to make it more realistic
        labels = np.array(labels)
        random_flip = np.random.rand(n_samples) < 0.1
        labels[random_flip] = np.random.choice([0, 1, 2], size=random_flip.sum())
        
        X = df.values
        y = labels
        
        print(f"Generated {n_samples} samples from {source}")
        print(f"Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def load_from_csv(self, filepath):
        """Load data from CSV file"""
        df = pd.read_csv(filepath)
        
        # Check if labels exist
        if 'label' in df.columns or 'disposition' in df.columns:
            label_col = 'label' if 'label' in df.columns else 'disposition'
            
            # Encode labels
            y = self.label_encoder.fit_transform(df[label_col])
            df = df.drop(columns=[label_col])
        else:
            y = None
        
        # Preprocess features
        X = self.preprocess_batch(df)
        
        return X.values, y
