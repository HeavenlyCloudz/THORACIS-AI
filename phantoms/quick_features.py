# quick_features.py
import numpy as np
import pandas as pd

def extract_features(s21_data, freqs):
    """Extract ML features from S21 data"""
    features = {}
    
    # Statistical features
    features['mean'] = np.mean(s21_data)
    features['std'] = np.std(s21_data)
    features['min'] = np.min(s21_data)
    features['max'] = np.max(s21_data)
    features['range'] = features['max'] - features['min']
    
    # Frequency domain features
    peak_idx = np.argmax(s21_data)
    features['peak_freq'] = freqs[peak_idx]
    features['peak_value'] = s21_data[peak_idx]
    
    # Slope (trend)
    slope = np.polyfit(freqs, s21_data, 1)[0]
    features['slope'] = slope
    
    # Energy
    features['energy'] = np.sum(s21_data**2)
    
    return features

# Load your data
print("=== FEATURE EXTRACTION ===")
baseline = pd.read_csv('baseline_data/path1_20260308_175728.csv')
freqs = baseline['Frequency_Hz'].values / 1e9
s21 = baseline['S21_dB'].values

features = extract_features(s21, freqs)
print("\nExtracted Features:")
for key, value in features.items():
    print(f"  {key}: {value:.4f}")
