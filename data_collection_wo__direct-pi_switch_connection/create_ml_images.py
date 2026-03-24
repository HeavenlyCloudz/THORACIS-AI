# create_ml_images_final.py - FIXED for small dataset
"""
PULMO AI: Convert S21 Scans to ML-Ready Formats
UPDATED: Background subtraction + Only Paths 1 & 2
FIXED: Handles small datasets (1 sample per class)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import json
import random
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# CONFIGURATION - Only use these paths (aligned opposite pairs)
VALID_PATHS = [1, 2]  # Path 1: 1→3, Path 2: 2→4
NUM_PATHS = 2  # Only 2 paths now!

def db_to_linear(db):
    """Convert dB to linear magnitude"""
    return 10 ** (db / 10)

def linear_to_db(linear):
    """Convert linear magnitude to dB"""
    linear = np.maximum(linear, 1e-12)
    return 10 * np.log10(linear)

class S21DataProcessor:
    def __init__(self, freq_start=2.0, freq_stop=3.0, num_points=201):
        self.freq_start = freq_start
        self.freq_stop = freq_stop
        self.num_points = num_points
        self.frequencies = np.linspace(freq_start, freq_stop, num_points)
    
    def load_with_background_subtraction(self, folder_path, baseline_data_dict=None):
        """
        Load CSV files and apply background subtraction if baseline provided
        Returns: dict path_num -> averaged S21 array (after subtraction)
        """
        path_data = {p: [] for p in VALID_PATHS}
        csv_files = sorted(Path(folder_path).glob('*.csv'))
        
        for file in csv_files:
            df = pd.read_csv(file)
            s21_values = df['S21_dB'].values
            
            # Ensure correct length
            if len(s21_values) > self.num_points:
                s21_values = s21_values[:self.num_points]
            elif len(s21_values) < self.num_points:
                s21_values = np.pad(s21_values, (0, self.num_points - len(s21_values)), 'edge')
            
            # Extract path number
            filename = file.stem
            parts = filename.split('_')
            path_num = None
            for part in parts:
                if part.startswith('path'):
                    try:
                        path_num = int(part[4])
                    except:
                        pass
                    break
            
            # Only use valid paths
            if path_num in VALID_PATHS:
                # Apply background subtraction if baseline provided
                if baseline_data_dict and path_num in baseline_data_dict:
                    baseline_linear = db_to_linear(baseline_data_dict[path_num])
                    signal_linear = db_to_linear(s21_values)
                    corrected_linear = signal_linear - baseline_linear
                    s21_values = linear_to_db(corrected_linear)
                
                path_data[path_num].append(s21_values)
        
        # Average multiple runs
        averaged_data = {}
        for path_num in VALID_PATHS:
            if path_data[path_num]:
                averaged_data[path_num] = np.mean(path_data[path_num], axis=0)
            else:
                averaged_data[path_num] = np.full(self.num_points, np.nan)
        
        return averaged_data, len(csv_files)
    
    def create_raw_features_vector(self, path_data):
        """Create flattened feature vector (2 paths × 201 points = 402 features)"""
        features = []
        for path_num in VALID_PATHS:
            features.extend(path_data.get(path_num, np.full(self.num_points, np.nan)))
        return np.array(features)
    
    def create_2d_image(self, path_data):
        """Create 2D image matrix (2 paths × 201 frequency points)"""
        image_matrix = np.zeros((NUM_PATHS, self.num_points))
        for idx, path_num in enumerate(VALID_PATHS):
            image_matrix[idx, :] = path_data.get(path_num, np.full(self.num_points, np.nan))
        return image_matrix
    
    def normalize_image(self, image_matrix):
        """Normalize to [0, 1] range for CNN"""
        image_matrix = np.clip(image_matrix, -100, 0)
        img_min = -100
        img_max = 0
        return (image_matrix - img_min) / (img_max - img_min)
    
    def extract_statistical_features(self, path_data):
        """Extract statistical features from each path"""
        features = {}
        for path_num in VALID_PATHS:
            if path_num in path_data:
                s21 = path_data[path_num]
                features[f'path{path_num}_mean'] = np.mean(s21)
                features[f'path{path_num}_std'] = np.std(s21)
                features[f'path{path_num}_min'] = np.min(s21)
                features[f'path{path_num}_max'] = np.max(s21)
                features[f'path{path_num}_range'] = np.max(s21) - np.min(s21)
        return features
    
    def extract_frequency_features(self, path_data):
        """Extract frequency-domain features"""
        features = {}
        for path_num in VALID_PATHS:
            if path_num in path_data:
                s21 = path_data[path_num]
                
                slope = np.polyfit(self.frequencies, s21, 1)[0]
                curvature = np.polyfit(self.frequencies, s21, 2)[0]
                features[f'path{path_num}_slope'] = slope
                features[f'path{path_num}_curvature'] = curvature
                
                peak_idx = np.argmax(s21)
                features[f'path{path_num}_peak_freq'] = self.frequencies[peak_idx]
                features[f'path{path_num}_peak_value'] = s21[peak_idx]
                
                # Use trapezoid instead of trapz (deprecated)
                features[f'path{path_num}_energy'] = np.trapezoid(s21, self.frequencies)
        return features
    
    def extract_path_difference_features(self, path_data):
        """Extract path-to-path difference features (now between Path 1 and Path 2)"""
        features = {}
        
        if 1 in path_data and 2 in path_data:
            diff_1_2 = np.mean(path_data[1] - path_data[2])
            features['diff_path1_path2'] = diff_1_2
            features['asymmetry_index'] = diff_1_2
        
        return features
    
    def extract_all_features(self, path_data):
        """Extract ALL features into a single vector"""
        all_features = {}
        all_features.update(self.extract_statistical_features(path_data))
        all_features.update(self.extract_frequency_features(path_data))
        all_features.update(self.extract_path_difference_features(path_data))
        return all_features

class DataAugmenter:
    @staticmethod
    def add_gaussian_noise(image, sigma=0.01):
        noise = np.random.normal(0, sigma, image.shape)
        return np.clip(image + noise, 0, 1)
    
    @staticmethod
    def gaussian_blur(image, sigma=1.0):
        return gaussian_filter(image, sigma=sigma)
    
    @staticmethod
    def frequency_shift(image, shift_pixels=5):
        return np.roll(image, shift_pixels, axis=1)
    
    @staticmethod
    def random_combination(image):
        aug = image.copy()
        if random.random() > 0.5:
            aug = DataAugmenter.add_gaussian_noise(aug, sigma=random.uniform(0.005, 0.02))
        if random.random() > 0.5:
            aug = DataAugmenter.gaussian_blur(aug, sigma=random.uniform(0.5, 2.0))
        if random.random() > 0.3:
            shift = random.randint(-10, 10)
            aug = DataAugmenter.frequency_shift(aug, shift)
        return np.clip(aug, 0, 1)

def main():
    print("="*70)
    print("PULMO AI: Convert S21 Scans to ML-Ready Formats (FINAL)")
    print("="*70)
    print("\n📡 CONFIGURATION:")
    print(f"   Using only Paths {VALID_PATHS} (aligned opposite pairs)")
    print("   Applying background subtraction using Air Baseline")
    print("   Generating features for XGBoost and CNN")
    
    processor = S21DataProcessor()
    augmenter = DataAugmenter()
    
    # Find latest phantom data folder
    data_folders = sorted(Path('.').glob('phantom_data_*'))
    if not data_folders:
        print("❌ No phantom_data_* folders found!")
        return
    
    latest_folder = data_folders[-1]
    print(f"\n📁 Using data from: {latest_folder}")
    
    # Load baseline (air) first
    print("\n📂 Loading Air Baseline (for background subtraction)...")
    baseline_folder = latest_folder / '01_baseline_air'
    baseline_data, _ = processor.load_with_background_subtraction(baseline_folder)
    
    if not baseline_data:
        print("❌ No baseline data found!")
        return
    
    print(f"   Loaded baseline for paths: {list(baseline_data.keys())}")
    
    # Define conditions
    conditions = [
        ('01_baseline_air', 'baseline', 0, 0.0),
        ('02_healthy_phantom', 'healthy', 1, 0.0),
        ('03_tumor_phantom', 'tumor', 2, 20.0)
    ]
    
    # Create output directories
    timestamp = latest_folder.name.replace('phantom_data_', '')
    output_base = f"ml_dataset_final_{timestamp}"
    os.makedirs(output_base, exist_ok=True)
    
    dirs = {
        'raw_features': f"{output_base}/01_raw_features",
        'cnn_images': f"{output_base}/02_cnn_images",
        'cnn_augmented': f"{output_base}/03_cnn_augmented",
        'visualizations': f"{output_base}/04_visualizations",
        'models': f"{output_base}/05_models"
    }
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    all_raw_features = []
    all_feature_vectors = []
    all_labels = []
    all_tumor_sizes = []
    all_cnn_images = []
    all_sample_ids = []
    
    print("\n📥 Processing conditions with background subtraction...")
    
    for folder_name, class_name, class_label, tumor_size in conditions:
        folder_path = latest_folder / folder_name
        if not folder_path.exists():
            print(f"  ⚠️ Skipping {folder_name} - not found")
            continue
        
        print(f"\n  📍 {class_name.upper()}: {folder_name}")
        
        # Load with background subtraction (except baseline itself)
        if folder_name == '01_baseline_air':
            path_data, num_files = processor.load_with_background_subtraction(folder_path)
        else:
            path_data, num_files = processor.load_with_background_subtraction(folder_path, baseline_data)
        
        if not path_data or all(np.isnan(path_data[p]).all() for p in path_data):
            print(f"     No valid data found")
            continue
        
        print(f"     Found {num_files} CSV files")
        print(f"     Paths loaded: {list(path_data.keys())}")
        
        # Create features
        raw_features = processor.create_raw_features_vector(path_data)
        engineered_features = processor.extract_all_features(path_data)
        feature_vector = [engineered_features.get(k, 0) for k in sorted(engineered_features.keys())]
        cnn_image = processor.create_2d_image(path_data)
        cnn_image_norm = processor.normalize_image(cnn_image)
        
        all_raw_features.append(raw_features)
        all_feature_vectors.append(feature_vector)
        all_labels.append(class_label)
        all_tumor_sizes.append(tumor_size)
        all_cnn_images.append(cnn_image_norm)
        all_sample_ids.append(folder_name)
        
        # Save individual files
        features_df = pd.DataFrame([raw_features])
        features_df.columns = [f'freq_{i}' for i in range(len(raw_features))]
        features_df['class'] = class_label
        features_df['tumor_size_mm'] = tumor_size
        features_df.to_csv(f"{dirs['raw_features']}/{folder_name}_features.csv", index=False)
        
        engineered_df = pd.DataFrame([feature_vector])
        engineered_df.columns = sorted(engineered_features.keys())
        engineered_df['class'] = class_label
        engineered_df.to_csv(f"{dirs['raw_features']}/{folder_name}_engineered.csv", index=False)
        
        np.save(f"{dirs['cnn_images']}/{folder_name}_image.npy", cnn_image_norm)
        
        # Visualization
        plt.figure(figsize=(8, 4))
        plt.imshow(cnn_image_norm, aspect='auto', cmap='viridis',
                  extent=[processor.freq_start, processor.freq_stop, NUM_PATHS + 0.5, 0.5])
        plt.colorbar(label='Normalized S21')
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Path Number')
        plt.yticks([1, 2], ['Path 1 (1→3)', 'Path 2 (2→4)'])
        plt.title(f"{class_name.upper()} (class={class_label}, tumor={tumor_size}mm)")
        plt.tight_layout()
        plt.savefig(f"{dirs['visualizations']}/{folder_name}.png", dpi=150)
        plt.close()
        
        print(f"     ✅ Saved")
        print(f"        • Raw features: {len(raw_features)}-dim")
        print(f"        • Engineered: {len(feature_vector)} features")
        print(f"        • CNN image: {NUM_PATHS}×{processor.num_points}")
        
        # Generate augmentations for tumor samples
        if 'tumor' in folder_name:
            print(f"     🔄 Generating augmentations...")
            for aug_name, aug_func in [
                ('noise', lambda x: augmenter.add_gaussian_noise(x, 0.01)),
                ('blur', lambda x: augmenter.gaussian_blur(x, 1.0)),
                ('shift', lambda x: augmenter.frequency_shift(x, 5)),
            ]:
                aug_image = aug_func(cnn_image_norm.copy())
                np.save(f"{dirs['cnn_augmented']}/{folder_name}_{aug_name}.npy", aug_image)
    
    # Create master datasets
    print("\n" + "="*70)
    print("📊 CREATING MASTER DATASETS")
    print("="*70)
    
    if len(all_raw_features) == 0:
        print("❌ No data to process!")
        return
    
    # XGBoost datasets
    xgb_df = pd.DataFrame(all_raw_features)
    xgb_df.columns = [f'freq_{i}' for i in range(len(all_raw_features[0]))]
    xgb_df['class'] = all_labels
    xgb_df['tumor_size_mm'] = all_tumor_sizes
    xgb_df['sample_id'] = all_sample_ids
    xgb_df.to_csv(f"{output_base}/pulmo_xgboost_data.csv", index=False)
    
    if all_feature_vectors:
        # Get feature names dynamically
        temp_features = processor.extract_all_features({1: np.zeros(processor.num_points), 2: np.zeros(processor.num_points)})
        feature_names = sorted(temp_features.keys())
        xgb_engineered_df = pd.DataFrame(all_feature_vectors, columns=feature_names)
        xgb_engineered_df['class'] = all_labels
        xgb_engineered_df['tumor_size_mm'] = all_tumor_sizes
        xgb_engineered_df['sample_id'] = all_sample_ids
        xgb_engineered_df.to_csv(f"{output_base}/pulmo_xgboost_engineered.csv", index=False)
    
    # CNN dataset
    cnn_images_array = np.array(all_cnn_images).reshape(-1, NUM_PATHS, processor.num_points, 1)
    np.save(f"{dirs['cnn_images']}/cnn_images.npy", cnn_images_array)
    np.save(f"{dirs['cnn_images']}/cnn_labels.npy", np.array(all_labels))
    
    # Train/val splits - handle small dataset gracefully
    X = np.array(all_raw_features)
    y = np.array(all_labels)
    
    # Check if we have enough samples for stratification
    unique, counts = np.unique(y, return_counts=True)
    min_samples = min(counts)
    
    if min_samples >= 2:
        # Can do stratified split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"✅ Stratified split: {len(X_train)} train, {len(X_val)} val")
    else:
        # Simple split without stratification
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"✅ Simple split (stratification not possible): {len(X_train)} train, {len(X_val)} val")
    
    np.save(f"{dirs['models']}/X_train.npy", X_train)
    np.save(f"{dirs['models']}/X_val.npy", X_val)
    np.save(f"{dirs['models']}/y_train.npy", y_train)
    np.save(f"{dirs['models']}/y_val.npy", y_val)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    with open(f"{dirs['models']}/scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\n✅ XGBoost Dataset: {len(all_raw_features)} samples, {len(all_raw_features[0])} features")
    print(f"   Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"✅ CNN Dataset: {cnn_images_array.shape}")
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'paths_used': VALID_PATHS,
        'num_samples': len(all_raw_features),
        'num_augmented': len(list(Path(dirs['cnn_augmented']).glob('*.npy'))),
        'feature_dims': {
            'raw': len(all_raw_features[0]),
            'engineered': len(all_feature_vectors[0]) if all_feature_vectors else 0
        },
        'class_distribution': {int(k): int(v) for k, v in zip(unique, counts)},
        'samples': all_sample_ids
    }
    
    with open(f"{output_base}/dataset_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n📁 Output folder: {output_base}/")
    print("\n🚀 DONE! Now you can train your models!")

if __name__ == "__main__":
    main()
