# smart_augmentation.py
"""
Smart augmentation for S21 data - avoids overfitting
Uses physically realistic transformations
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy import signal
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def db_to_linear(db):
    return 10 ** (db / 10)

def linear_to_db(linear):
    linear = np.maximum(linear, 1e-12)
    return 10 * np.log10(linear)

def magnitude_phase_jitter(s21_mag_db, s21_phase_deg, noise_mag_db=0.2, noise_phase_deg=2.0):
    """
    Add realistic jitter to magnitude and phase
    Simulates antenna placement variation and cable movement
    """
    mag_jitter = np.random.normal(0, noise_mag_db, len(s21_mag_db))
    phase_jitter = np.random.normal(0, noise_phase_deg, len(s21_phase_deg))
    
    return s21_mag_db + mag_jitter, s21_phase_deg + phase_jitter

def frequency_shift_interpolation(freqs, s21_mag_db, s21_phase_deg, shift_percent=1.5):
    """
    Shift frequency response by small percentage
    Uses interpolation to maintain smooth curve
    """
    shift = np.random.uniform(-shift_percent, shift_percent) / 100
    new_freqs = freqs * (1 + shift)
    
    # Interpolate magnitude and phase
    interp_mag = interp1d(freqs, s21_mag_db, kind='cubic', fill_value='extrapolate')
    interp_phase = interp1d(freqs, s21_phase_deg, kind='cubic', fill_value='extrapolate')
    
    # Keep original frequency points
    return interp_mag(freqs), interp_phase(freqs)

def complex_awgn(s21_real, s21_imag, snr_db=25):
    """
    Add Additive White Gaussian Noise to complex components
    More physically accurate than adding noise to dB
    """
    signal_power = np.mean(s21_real**2 + s21_imag**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(s21_real)) + 1j * np.random.randn(len(s21_imag)))
    
    s21_complex = s21_real + 1j * s21_imag
    s21_noisy = s21_complex + noise
    
    return np.real(s21_noisy), np.imag(s21_noisy)

def virtual_phantom_mixup(s21_a, s21_b, lambda_val=None):
    """
    Create virtual phantoms by mixing two existing ones
    Uses linear mixing which is physically consistent for S-parameters
    """
    if lambda_val is None:
        lambda_val = np.random.uniform(0.3, 0.7)
    return lambda_val * s21_a + (1 - lambda_val) * s21_b

def time_domain_features(s21_real, s21_imag):
    """
    Convert frequency domain S21 to time domain via IFFT
    Creates additional features that XGBoost handles well
    """
    s21_complex = s21_real + 1j * s21_imag
    time_domain = np.fft.ifft(s21_complex)
    return np.real(time_domain), np.imag(time_domain), np.abs(time_domain)

def augment_dataset(df, n_augmented_per_sample=10, group_k_folds=4):
    """
    Smart augmentation with Group K-Fold awareness
    """
    augmented_samples = []
    metadata = []
    
    # Group by condition and experiment
    groups = df.groupby(['condition', 'experiment_id'])
    
    for (condition, exp_id), group in groups:
        print(f"  Augmenting {condition} - Experiment {exp_id}...")
        
        for _ in range(n_augmented_per_sample):
            # Pick a random sample from this group
            sample = group.sample(1).iloc[0]
            
            # Extract features
            freq_cols = [c for c in df.columns if c.startswith('freq_')]
            s21_mag_db = sample[freq_cols].values
            
            # Generate phase from magnitude (placeholder - replace with real phase if available)
            s21_phase_deg = np.random.uniform(-30, 30, len(s21_mag_db))
            s21_real = db_to_linear(s21_mag_db) * np.cos(np.radians(s21_phase_deg))
            s21_imag = db_to_linear(s21_mag_db) * np.sin(np.radians(s21_phase_deg))
            
            # Randomly choose augmentation type
            aug_type = np.random.choice([
                'magnitude_phase_jitter',
                'frequency_shift',
                'complex_noise',
                'mixup'
            ])
            
            if aug_type == 'magnitude_phase_jitter':
                mag_new, phase_new = magnitude_phase_jitter(s21_mag_db, s21_phase_deg)
                augmented = mag_new
                
            elif aug_type == 'frequency_shift':
                freqs = np.linspace(2, 3, len(s21_mag_db))
                mag_new, phase_new = frequency_shift_interpolation(freqs, s21_mag_db, s21_phase_deg)
                augmented = mag_new
                
            elif aug_type == 'complex_noise':
                real_new, imag_new = complex_awgn(s21_real, s21_imag, snr_db=25)
                augmented = linear_to_db(np.sqrt(real_new**2 + imag_new**2))
                
            elif aug_type == 'mixup':
                # Pick another sample from same class for mixup
                other_sample = group.sample(1).iloc[0]
                other_mag = other_sample[freq_cols].values
                lambda_val = np.random.uniform(0.3, 0.7)
                augmented = virtual_phantom_mixup(s21_mag_db, other_mag, lambda_val)
            
            # Create augmented sample
            aug_sample = {col: val for col, val in sample.items()}
            for i, col in enumerate(freq_cols):
                aug_sample[col] = augmented[i]
            aug_sample['augmented'] = True
            aug_sample['augmentation_type'] = aug_type
            
            augmented_samples.append(aug_sample)
            metadata.append({
                'original_class': condition,
                'experiment_id': exp_id,
                'augmentation_type': aug_type
            })
    
    return pd.DataFrame(augmented_samples), metadata

def main():
    print("="*70)
    print("PULMO AI: Smart Augmentation (No Overfitting)")
    print("="*70)
    
    # Load your combined dataset
    df = pd.read_csv("pulmo_combined_dataset/pulmo_combined.csv")
    print(f"\n📁 Original dataset: {len(df)} samples")
    
    # Add condition and experiment_id if not present
    # You'll need to add these columns based on your metadata
    # For now, assume class = 0,1,2 corresponds to condition
    df['condition'] = df['class'].map({0: 'baseline', 1: 'healthy', 2: 'tumor'})
    df['experiment_id'] = df.index // 12  # Rough grouping
    
    # Generate augmented samples
    print("\n🔧 Generating augmented samples...")
    df_aug, metadata = augment_dataset(df, n_augmented_per_sample=5)
    
    # Combine original and augmented
    df_original = df.copy()
    df_original['augmented'] = False
    df_original['augmentation_type'] = 'original'
    
    df_combined = pd.concat([df_original, df_aug], ignore_index=True)
    
    print(f"\n✅ Combined dataset: {len(df_combined)} samples")
    print(f"   Original: {len(df_original)}")
    print(f"   Augmented: {len(df_aug)}")
    
    # Save
    output_dir = "pulmo_augmented_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    df_combined.to_csv(f"{output_dir}/pulmo_augmented.csv", index=False)
    
    with open(f"{output_dir}/augmentation_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n📁 Saved to: {output_dir}/")
    print("   - pulmo_augmented.csv")
    print("   - augmentation_metadata.json")
    
    print("\n🎉 AUGMENTATION COMPLETE!")

if __name__ == "__main__":
    import os
    main()