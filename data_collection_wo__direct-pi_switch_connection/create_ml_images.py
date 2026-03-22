# create_ml_images_updated.py
"""
PULMO AI: Convert S21 Scans to ML-Ready Formats
Generates:
1. Raw features for XGBoost (804-dim + path differences)
2. 2D images for CNN (4 paths × 201 frequency points)
3. Augmented dataset for CNN training
4. Train/validation splits
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

class S21DataProcessor:
    def __init__(self, freq_start=2.0, freq_stop=3.0, num_points=201):
        self.freq_start = freq_start
        self.freq_stop = freq_stop
        self.num_points = num_points
        self.frequencies = np.linspace(freq_start, freq_stop, num_points)
        
    def load_all_scans_from_folder(self, folder_path):
        """Load ALL CSV files from a condition folder (multiple runs/rotations)"""
        all_path_data = []
        csv_files = sorted(Path(folder_path).glob('*.csv'))
        
        for file in csv_files:
            df = pd.read_csv(file)
            s21_values = df['S21_dB'].values
            
            # Ensure correct length
            if len(s21_values) > self.num_points:
                s21_values = s21_values[:self.num_points]
            elif len(s21_values) < self.num_points:
                s21_values = np.pad(s21_values, (0, self.num_points - len(s21_values)), 'edge')
            
            # Extract path number from filename
            # Filename format: condition_pathX_rotX_runX_timestamp.csv
            parts = file.stem.split('_')
            path_num = 1
            for part in parts:
                if part.startswith('path'):
                    try:
                        path_num = int(part[4])
                    except:
                        pass
                    break
            
            all_path_data.append({
                'path_num': path_num,
                's21': s21_values,
                'filename': file.name,
                'rotation': self._extract_rotation(file.name),
                'run': self._extract_run(file.name)
            })
        
        # Group by path
        grouped = {i: [] for i in range(1, 5)}
        for data in all_path_data:
            grouped[data['path_num']].append(data['s21'])
        
        # Average multiple runs for each path
        path_data = {}
        for path_num in range(1, 5):
            if grouped[path_num]:
                # Average across all runs for this path
                path_data[path_num] = np.mean(grouped[path_num], axis=0)
            else:
                path_data[path_num] = np.full(self.num_points, np.nan)
        
        return path_data, len(csv_files)
    
    def _extract_rotation(self, filename):
        """Extract rotation from filename"""
        if 'rot' in filename:
            try:
                parts = filename.split('_')
                for part in parts:
                    if part.startswith('rot'):
                        return int(part[3:])
            except:
                pass
        return 0
    
    def _extract_run(self, filename):
        """Extract run number from filename"""
        if 'run' in filename:
            try:
                parts = filename.split('_')
                for part in parts:
                    if part.startswith('run'):
                        return int(part[3:])
            except:
                pass
        return 1
    
    def create_raw_features_vector(self, path_data):
        """
        Create flattened feature vector for XGBoost
        Shape: (4 paths × 201 frequency points) = 804 features
        """
        features = []
        for path_num in range(1, 5):
            if path_num in path_data:
                features.extend(path_data[path_num])
            else:
                features.extend([np.nan] * self.num_points)
        return np.array(features)
    
    def create_2d_image(self, path_data):
        """
        Create 2D image matrix for CNN:
        Rows = 4 paths
        Columns = 201 frequency points
        """
        image_matrix = np.zeros((4, self.num_points))
        for path_num in range(1, 5):
            if path_num in path_data:
                image_matrix[path_num-1, :] = path_data[path_num]
            else:
                image_matrix[path_num-1, :] = np.nan
        return image_matrix
    
    def normalize_image(self, image_matrix):
        """Normalize to [0, 1] range for CNN"""
        image_matrix = np.clip(image_matrix, -60, 0)
        img_min = -60
        img_max = 0
        return (image_matrix - img_min) / (img_max - img_min)
    
    def extract_statistical_features(self, path_data):
        """Extract statistical features from each path"""
        features = {}
        for path_num in range(1, 5):
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
        for path_num in range(1, 5):
            if path_num in path_data:
                s21 = path_data[path_num]
                
                # Slope and curvature
                slope = np.polyfit(self.frequencies, s21, 1)[0]
                curvature = np.polyfit(self.frequencies, s21, 2)[0]
                features[f'path{path_num}_slope'] = slope
                features[f'path{path_num}_curvature'] = curvature
                
                # Peak location and value
                peak_idx = np.argmax(s21)
                features[f'path{path_num}_peak_freq'] = self.frequencies[peak_idx]
                features[f'path{path_num}_peak_value'] = s21[peak_idx]
                
                # Energy (area under curve)
                features[f'path{path_num}_energy'] = np.trapz(s21, self.frequencies)
        return features
    
    def extract_path_difference_features(self, path_data):
        """Extract path-to-path difference features for spatial information"""
        features = {}
        
        if 1 in path_data and 3 in path_data:
            # Opposite paths
            diff_1_3 = np.mean(path_data[1] - path_data[3])
            features['diff_path1_path3'] = diff_1_3
        
        if 2 in path_data and 4 in path_data:
            diff_2_4 = np.mean(path_data[2] - path_data[4])
            features['diff_path2_path4'] = diff_2_4
        
        if 1 in path_data and 4 in path_data:
            diff_1_4 = np.mean(path_data[1] - path_data[4])
            features['diff_path1_path4'] = diff_1_4
        
        if 2 in path_data and 3 in path_data:
            diff_2_3 = np.mean(path_data[2] - path_data[3])
            features['diff_path2_path3'] = diff_2_3
        
        # Asymmetry index (how different are the two sides)
        if 1 in path_data and 2 in path_data and 3 in path_data and 4 in path_data:
            left_side = path_data[1] + path_data[2]
            right_side = path_data[3] + path_data[4]
            asymmetry = np.mean(left_side - right_side)
            features['asymmetry_index'] = asymmetry
            
            # Cross-path ratio
            cross_ratio = (path_data[1] + path_data[4]) / (path_data[2] + path_data[3] + 1e-6)
            features['cross_ratio'] = np.mean(cross_ratio)
        
        return features
    
    def extract_all_features(self, path_data):
        """Extract ALL features into a single vector"""
        all_features = {}
        
        # Statistical features
        all_features.update(self.extract_statistical_features(path_data))
        
        # Frequency features
        all_features.update(self.extract_frequency_features(path_data))
        
        # Path difference features
        all_features.update(self.extract_path_difference_features(path_data))
        
        return all_features

class DataAugmenter:
    """Augment 2D image data for CNN training"""
    
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
    def horizontal_flip(image):
        """Flip horizontally (swap paths)"""
        flipped = image.copy()
        flipped[0:2] = image[1::-1]
        flipped[2:4] = image[3:1:-1]
        return flipped
    
    @staticmethod
    def add_baseline_drift(image, drift_strength=0.05):
        drift = np.linspace(0, drift_strength, image.shape[1])
        return image + drift[np.newaxis, :]
    
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
        if random.random() > 0.7:
            aug = DataAugmenter.horizontal_flip(aug)
        return np.clip(aug, 0, 1)

def create_tumor_size_label(condition_folder):
    """Create tumor size label based on condition"""
    if 'tumor' in condition_folder.lower():
        return 20.0  # mm - adjust to your actual tumor size
    else:
        return 0.0

def main():
    print("="*70)
    print("PULMO AI: Convert S21 Scans to ML-Ready Formats")
    print("="*70)
    
    processor = S21DataProcessor()
    augmenter = DataAugmenter()
    
    # Find latest phantom data folder
    data_folders = sorted(Path('.').glob('phantom_data_*'))
    if not data_folders:
        print("❌ No phantom_data_* folders found! Run full_phantom_scan.py first")
        return
    
    latest_folder = data_folders[-1]
    print(f"\n📁 Using data from: {latest_folder}")
    
    # Create output directories
    timestamp = latest_folder.name.replace('phantom_data_', '')
    output_base = f"ml_dataset_{timestamp}"
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
    
    # Define conditions
    conditions = ['01_baseline_air', '02_healthy_phantom_1', '03_healthy_phantom_2', '04_tumor_phantom']
    
    # Data containers
    all_raw_features = []
    all_feature_vectors = []  # For XGBoost with engineered features
    all_labels = []
    all_tumor_sizes = []
    all_cnn_images = []
    all_metadata = []
    
    print("\n📥 Processing conditions:")
    
    for condition in conditions:
        folder_path = latest_folder / condition
        if not folder_path.exists():
            print(f"  ⚠️  Skipping {condition} - not found")
            continue
        
        print(f"\n  📍 {condition}")
        
        # Load ALL scans from this condition (multiple runs)
        path_data, num_files = processor.load_all_scans_from_folder(folder_path)
        
        if not path_data:
            print(f"     No data found")
            continue
        
        print(f"     Found {num_files} CSV files")
        
        # Create sample ID
        sample_id = condition
        
        # 1. Create raw features (804-dim) for XGBoost
        raw_features = processor.create_raw_features_vector(path_data)
        all_raw_features.append(raw_features)
        
        # 2. Create engineered feature vector (statistical + frequency + path differences)
        engineered_features = processor.extract_all_features(path_data)
        feature_vector = [engineered_features.get(k, 0) for k in sorted(engineered_features.keys())]
        all_feature_vectors.append(feature_vector)
        
        # 3. Create 2D image for CNN
        cnn_image = processor.create_2d_image(path_data)
        cnn_image_norm = processor.normalize_image(cnn_image)
        all_cnn_images.append(cnn_image_norm)
        
        # 4. Labels
        class_label = 2 if 'tumor' in condition else (1 if 'healthy' in condition else 0)
        tumor_size = 20.0 if 'tumor' in condition else 0.0
        
        all_labels.append(class_label)
        all_tumor_sizes.append(tumor_size)
        
        # Save raw features as CSV
        features_df = pd.DataFrame([raw_features])
        features_df.columns = [f'freq_{i}' for i in range(len(raw_features))]
        features_df['class'] = class_label
        features_df['tumor_size_mm'] = tumor_size
        features_df.to_csv(f"{dirs['raw_features']}/{sample_id}_features.csv", index=False)
        
        # Save engineered features
        engineered_df = pd.DataFrame([feature_vector])
        engineered_df.columns = sorted(engineered_features.keys())
        engineered_df['class'] = class_label
        engineered_df['tumor_size_mm'] = tumor_size
        engineered_df.to_csv(f"{dirs['raw_features']}/{sample_id}_engineered_features.csv", index=False)
        
        # Save CNN image
        np.save(f"{dirs['cnn_images']}/{sample_id}_image.npy", cnn_image_norm)
        
        # Save visualization
        plt.figure(figsize=(10, 3))
        plt.imshow(cnn_image_norm, aspect='auto', cmap='viridis',
                  extent=[processor.freq_start, processor.freq_stop, 4.5, 0.5])
        plt.colorbar(label='Normalized S21')
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Path Number')
        plt.yticks([1, 2, 3, 4])
        plt.title(f"{condition} (class={class_label}, tumor={tumor_size}mm)")
        plt.tight_layout()
        plt.savefig(f"{dirs['visualizations']}/{sample_id}.png", dpi=150)
        plt.close()
        
        # Save metadata
        metadata = {
            'sample_id': sample_id,
            'class': class_label,
            'class_name': 'tumor' if class_label == 2 else ('healthy' if class_label == 1 else 'baseline'),
            'tumor_size_mm': tumor_size,
            'num_files': num_files,
            'engineered_features': list(engineered_features.keys()),
            'mean_s21_db': float(np.nanmean(cnn_image)),
            'std_s21_db': float(np.nanstd(cnn_image))
        }
        all_metadata.append(metadata)
        
        with open(f"{dirs['cnn_images']}/{sample_id}_meta.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"     ✅ Saved: {sample_id}")
        print(f"        • Raw features: {len(raw_features)}-dim")
        print(f"        • Engineered: {len(feature_vector)} features")
        print(f"        • CNN image: 4×201")
        
        # Generate augmentations for CNN
        print(f"     🔄 Generating augmentations...")
        
        augmentations = {
            'noise_01': lambda x: augmenter.add_gaussian_noise(x, 0.01),
            'noise_02': lambda x: augmenter.add_gaussian_noise(x, 0.02),
            'blur_1': lambda x: augmenter.gaussian_blur(x, 1.0),
            'blur_2': lambda x: augmenter.gaussian_blur(x, 2.0),
            'shift_pos_5': lambda x: augmenter.frequency_shift(x, 5),
            'shift_neg_5': lambda x: augmenter.frequency_shift(x, -5),
            'flip': lambda x: augmenter.horizontal_flip(x),
            'drift': lambda x: augmenter.add_baseline_drift(x, 0.05),
            'random': lambda x: augmenter.random_combination(x)
        }
        
        for aug_name, aug_func in augmentations.items():
            aug_image = aug_func(cnn_image_norm.copy())
            np.save(f"{dirs['cnn_augmented']}/{sample_id}_{aug_name}.npy", aug_image)
    
    # Create master datasets
    print("\n" + "="*70)
    print("📊 CREATING MASTER DATASETS")
    print("="*70)
    
    # 1. XGBoost with raw features
    xgb_df = pd.DataFrame(all_raw_features)
    xgb_df.columns = [f'freq_{i}' for i in range(len(all_raw_features[0]))]
    xgb_df['class'] = all_labels
    xgb_df['tumor_size_mm'] = all_tumor_sizes
    xgb_df.to_csv(f"{output_base}/pulmo_xgboost_raw.csv", index=False)
    
    # 2. XGBoost with engineered features
    if all_feature_vectors:
        feature_names = sorted(processor.extract_all_features({1: np.zeros(201)}).keys())
        xgb_engineered_df = pd.DataFrame(all_feature_vectors, columns=feature_names)
        xgb_engineered_df['class'] = all_labels
        xgb_engineered_df['tumor_size_mm'] = all_tumor_sizes
        xgb_engineered_df.to_csv(f"{output_base}/pulmo_xgboost_engineered.csv", index=False)
    
    # 3. CNN dataset
    cnn_images_array = np.array(all_cnn_images).reshape(-1, 4, 201, 1)
    np.save(f"{dirs['cnn_images']}/cnn_images.npy", cnn_images_array)
    np.save(f"{dirs['cnn_images']}/cnn_labels.npy", np.array(all_labels))
    np.save(f"{dirs['cnn_images']}/cnn_tumor_sizes.npy", np.array(all_tumor_sizes))
    
    # Create train/validation splits
    X_raw = np.array(all_raw_features)
    y_class = np.array(all_labels)
    y_size = np.array(all_tumor_sizes)
    
    X_train, X_val, y_class_train, y_class_val, y_size_train, y_size_val = train_test_split(
        X_raw, y_class, y_size, test_size=0.2, random_state=42, stratify=y_class
    )
    
    # Save splits
    np.save(f"{dirs['models']}/X_train.npy", X_train)
    np.save(f"{dirs['models']}/X_val.npy", X_val)
    np.save(f"{dirs['models']}/y_class_train.npy", y_class_train)
    np.save(f"{dirs['models']}/y_class_val.npy", y_class_val)
    np.save(f"{dirs['models']}/y_size_train.npy", y_size_train)
    np.save(f"{dirs['models']}/y_size_val.npy", y_size_val)
    
    # Save scaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    with open(f"{dirs['models']}/xgboost_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save metadata
    with open(f"{output_base}/dataset_metadata.json", 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'num_original_samples': len(all_raw_features),
            'num_augmented': len(list(Path(dirs['cnn_augmented']).glob('*.npy'))),
            'feature_dims': {
                'raw': len(all_raw_features[0]),
                'engineered': len(all_feature_vectors[0]) if all_feature_vectors else 0
            },
            'class_distribution': {0: sum(1 for l in all_labels if l == 0),
                                   1: sum(1 for l in all_labels if l == 1),
                                   2: sum(1 for l in all_labels if l == 2)},
            'metadata': all_metadata
        }, f, indent=2)
    
    print(f"\n✅ XGBoost Raw Dataset: {xgb_df.shape}")
    if all_feature_vectors:
        print(f"✅ XGBoost Engineered Dataset: {len(all_feature_vectors)} samples, {len(all_feature_vectors[0])} features")
    print(f"✅ CNN Dataset: {cnn_images_array.shape}")
    print(f"✅ Train/Val splits saved")
    print(f"✅ Augmentations: {len(list(Path(dirs['cnn_augmented']).glob('*.npy')))} files")
    
    print(f"\n📁 Output folder: {output_base}/")
    print("\n🚀 NEXT: Upload to Kaggle and train models!")

if __name__ == "__main__":
    main()
