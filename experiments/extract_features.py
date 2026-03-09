import pandas as pd
import numpy as np
from scipy import stats

def extract_features(filename, label):
    """Extract features from a single S21 scan CSV."""
    df = pd.read_csv(filename)
    s21 = df['S21_dB']
    
    features = {
        'label': label,
        'mean_db': s21.mean(),
        'std_db': s21.std(),
        'min_db': s21.min(),
        'max_db': s21.max(),
        'range_db': s21.max() - s21.min(),
    }
    
    # Calculate slope (linear regression on the curve)
    x = np.arange(len(s21))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, s21)
    features['slope'] = slope
    
    # Calculate quartiles
    features['q1_db'] = np.percentile(s21, 25)
    features['median_db'] = np.percentile(s21, 50)
    features['q3_db'] = np.percentile(s21, 75)
    
    return features

# --- Extract features from your three key scans ---
print("Extracting features from your experiment data...")
dataset = []

# Add your scans (adjust filenames and labels as needed)
dataset.append(extract_features('scan_phantom_healthy.csv', 'healthy'))
dataset.append(extract_features('scan_phantom_tumor_highcontrast.csv', 'tumor'))
# Add more scans if you have them

# Create DataFrame and save
features_df = pd.DataFrame(dataset)
print("\nExtracted Features:")
print(features_df.to_string(index=False))

# Save to CSV for your ML model
features_df.to_csv('training_dataset.csv', index=False)
print(f"\n✅ Features saved to 'training_dataset.csv'")
print(f"   You now have {len(dataset)} samples for ML training.")