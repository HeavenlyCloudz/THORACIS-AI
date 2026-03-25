# smart_augmentation_rotations_v2.py - WITH SYNTHETIC DATA
"""
Enhanced augmentation with synthetic data generation
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.neighbors import KernelDensity
import json
import os
import warnings
warnings.filterwarnings('ignore')

def generate_synthetic_data(X, y, exp_id, noise_level=0.05, n_synthetic=2):
    """
    Generate synthetic samples using kernel density estimation
    """
    synthetic_samples = []
    
    for class_label in np.unique(y):
        class_mask = y == class_label
        X_class = X[class_mask]
        exp_class = exp_id[class_mask]
        
        # Fit KDE to class distribution
        kde = KernelDensity(kernel='gaussian', bandwidth=0.1)
        kde.fit(X_class)
        
        # Generate synthetic samples
        for _ in range(n_synthetic * len(X_class)):
            # Sample from KDE
            random_idx = np.random.choice(len(X_class))
            base_sample = X_class[random_idx]
            
            # Add noise with correlation structure
            noise = np.random.normal(0, noise_level, base_sample.shape)
            synthetic_feature = base_sample + noise
            
            # Assign a new experiment ID for synthetic data
            synthetic_exp = -1  # Mark as synthetic
            
            synthetic_samples.append({
                'features': synthetic_feature,
                'class': class_label,
                'exp_id': synthetic_exp,
                'synthetic': True
            })
    
    return synthetic_samples

def mixup_cross_class(X_a, y_a, exp_a, X_b, y_b, exp_b, n_mixup=100):
    """
    Create cross-class mixup samples to improve decision boundary
    """
    mixup_samples = []
    
    for _ in range(n_mixup):
        # Randomly select two samples from different classes
        idx_a = np.random.choice(len(X_a))
        idx_b = np.random.choice(len(X_b))
        
        # Ensure different classes
        if y_a[idx_a] == y_b[idx_b]:
            continue
        
        # Random mixup weight
        lambda_val = np.random.uniform(0.3, 0.7)
        
        # Mix features
        mixed_features = lambda_val * X_a[idx_a] + (1 - lambda_val) * X_b[idx_b]
        
        # Assign class based on mixup weight (soft label)
        # For hard labels, use the class with higher weight
        if lambda_val > 0.5:
            hard_label = y_a[idx_a]
        else:
            hard_label = y_b[idx_b]
        
        mixup_samples.append({
            'features': mixed_features,
            'class': hard_label,
            'exp_id': -2,  # Mixup samples get special ID
            'synthetic': True,
            'mixup_alpha': lambda_val
        })
    
    return mixup_samples

def augment_dataset(df, n_augmented_per_sample=5, add_synthetic=True):
    """
    Enhanced augmentation with synthetic data
    """
    augmented_samples = []
    synthetic_samples_list = []
    mixup_samples_list = []
    
    # Get features
    feature_cols = [c for c in df.columns if c.startswith('freq_')]
    X = df[feature_cols].values
    y = df['class'].values
    exp_ids = df['exp_id'].values
    
    # Generate synthetic samples using KDE
    if add_synthetic:
        print("   Generating synthetic samples...")
        synthetic = generate_synthetic_data(X, y, exp_ids, noise_level=0.05, n_synthetic=2)
        for s in synthetic:
            synthetic_samples_list.append(s)
    
    # Generate cross-class mixup samples
    print("   Generating cross-class mixup samples...")
    for class1, class2 in [(0,1), (0,2), (1,2)]:
        mask1 = y == class1
        mask2 = y == class2
        mixup = mixup_cross_class(
            X[mask1], y[mask1], exp_ids[mask1],
            X[mask2], y[mask2], exp_ids[mask2],
            n_mixup=50
        )
        mixup_samples_list.extend(mixup)
    
    # Original augmentation (your existing code)
    freqs = np.linspace(2, 3, len(feature_cols))
    
    for idx, row in df.iterrows():
        original_features = row[feature_cols].values
        class_label = row['class']
        exp_id = row['exp_id']
        rotation = row['rotation']
        
        for aug_idx in range(n_augmented_per_sample):
            # Randomly choose augmentation type
            aug_type = np.random.choice([
                'magnitude_jitter',
                'frequency_shift',
                'mixup_same_class',
                'rotation_simulate'
            ], p=[0.25, 0.25, 0.25, 0.25])
            
            if aug_type == 'magnitude_jitter':
                mag_jitter = np.random.normal(0, 0.3, len(original_features))
                augmented = original_features + mag_jitter
                
            elif aug_type == 'frequency_shift':
                shift = np.random.uniform(-1, 1) / 100
                new_freqs = freqs * (1 + shift)
                interp_mag = interp1d(freqs, original_features, kind='cubic', fill_value='extrapolate')
                augmented = interp_mag(freqs)
                
            elif aug_type == 'mixup_same_class':
                same_class = df[df['class'] == class_label]
                if len(same_class) > 1:
                    other_idx = np.random.choice([i for i in same_class.index if i != idx])
                    other_features = same_class.loc[other_idx, feature_cols].values
                    lambda_val = np.random.uniform(0.3, 0.7)
                    augmented = lambda_val * original_features + (1 - lambda_val) * other_features
                else:
                    augmented = original_features + np.random.normal(0, 0.3, len(original_features))
                    aug_type = 'magnitude_jitter (fallback)'
                
            elif aug_type == 'rotation_simulate':
                new_rotation = np.random.choice([0, 120, 240])
                if new_rotation == 0:
                    augmented = original_features
                elif new_rotation == 120:
                    augmented = np.roll(original_features, shift=20)
                else:
                    augmented = original_features + np.random.normal(0, 0.2, len(original_features))
            
            # Create augmented sample
            new_row = row.to_dict()
            for i, col in enumerate(feature_cols):
                new_row[col] = augmented[i]
            new_row['augmented'] = True
            new_row['augmentation_type'] = aug_type
            new_row['original_rotation'] = rotation
            
            augmented_samples.append(new_row)
    
    # Convert to DataFrames
    df_augmented = pd.DataFrame(augmented_samples)
    
    df_synthetic = pd.DataFrame(synthetic_samples_list) if synthetic_samples_list else pd.DataFrame()
    if not df_synthetic.empty:
        for i, col in enumerate(feature_cols):
            df_synthetic[col] = [s['features'][i] for s in synthetic_samples_list]
        df_synthetic['class'] = [s['class'] for s in synthetic_samples_list]
        df_synthetic['exp_id'] = [s['exp_id'] for s in synthetic_samples_list]
        df_synthetic['augmented'] = True
        df_synthetic['synthetic'] = True
        df_synthetic['augmentation_type'] = 'kde_synthetic'
        df_synthetic['rotation'] = 0  # Default rotation for synthetic
    
    df_mixup = pd.DataFrame(mixup_samples_list) if mixup_samples_list else pd.DataFrame()
    if not df_mixup.empty:
        for i, col in enumerate(feature_cols):
            df_mixup[col] = [s['features'][i] for s in mixup_samples_list]
        df_mixup['class'] = [s['class'] for s in mixup_samples_list]
        df_mixup['exp_id'] = [s['exp_id'] for s in mixup_samples_list]
        df_mixup['augmented'] = True
        df_mixup['synthetic'] = True
        df_mixup['augmentation_type'] = 'cross_class_mixup'
        df_mixup['rotation'] = 0
    
    return df_augmented, df_synthetic, df_mixup

def main():
    print("="*70)
    print("PULMO AI: Enhanced Augmentation with Synthetic Data")
    print("="*70)
    
    # Load the rotation-based combined dataset
    try:
        df = pd.read_csv("pulmo_combined_rotations/pulmo_combined.csv")
        print(f"\n📁 Loaded rotation-based dataset: {len(df)} samples")
    except:
        print("❌ Please run combine_all_experiments_rotations.py first")
        return
    
    print(f"   Class distribution: baseline={(df['class']==0).sum()}, "
          f"healthy={(df['class']==1).sum()}, tumor={(df['class']==2).sum()}")
    
    # Generate augmented and synthetic samples
    print("\n🔧 Generating enhanced samples...")
    df_aug, df_synthetic, df_mixup = augment_dataset(df, n_augmented_per_sample=5, add_synthetic=True)
    
    # Combine all
    df_original = df.copy()
    df_original['augmented'] = False
    df_original['synthetic'] = False
    df_original['augmentation_type'] = 'original'
    
    df_combined = pd.concat([df_original, df_aug, df_synthetic, df_mixup], ignore_index=True)
    
    print(f"\n✅ Combined dataset: {len(df_combined)} samples")
    print(f"   Original: {len(df_original)}")
    print(f"   Rotational augmentations: {len(df_aug)}")
    print(f"   KDE synthetic: {len(df_synthetic)}")
    print(f"   Cross-class mixup: {len(df_mixup)}")
    
    # Show class distribution after augmentation
    print(f"\n📊 Final class distribution:")
    for class_label in [0, 1, 2]:
        class_name = ['baseline', 'healthy', 'tumor'][class_label]
        count = (df_combined['class'] == class_label).sum()
        print(f"   {class_name}: {count}")
    
    # Save
    output_dir = "pulmo_augmented_rotations_enhanced"
    os.makedirs(output_dir, exist_ok=True)
    
    df_combined.to_csv(f"{output_dir}/pulmo_augmented_enhanced.csv", index=False)
    
    metadata = {
        'num_original_samples': len(df_original),
        'num_rotational_augmentations': len(df_aug),
        'num_kde_synthetic': len(df_synthetic),
        'num_cross_class_mixup': len(df_mixup),
        'num_total_samples': len(df_combined),
        'class_distribution': {
            'baseline': int((df_combined['class']==0).sum()),
            'healthy': int((df_combined['class']==1).sum()),
            'tumor': int((df_combined['class']==2).sum())
        }
    }
    
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n📁 Saved to: {output_dir}/")
    print("   - pulmo_augmented_enhanced.csv")
    print("   - metadata.json")
    print("\n🎉 ENHANCED AUGMENTATION COMPLETE!")

if __name__ == "__main__":
    main()
