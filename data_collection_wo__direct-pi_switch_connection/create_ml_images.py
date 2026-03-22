# create_ml_images_updated.py
"""
PULMO AI: Convert S21 Scans to ML-Ready Formats
MULTIPLE RUN AGGREGATION for statistical robustness
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
import glob

class S21DataProcessor:
    def __init__(self, freq_start=2.0, freq_stop=3.0, num_points=201):
        self.freq_start = freq_start
        self.freq_stop = freq_stop
        self.num_points = num_points
        self.frequencies = np.linspace(freq_start, freq_stop, num_points)
        
    def load_all_runs_from_folder(self, folder_path):
        """
        Load ALL runs from a condition folder and aggregate them
        Returns: mean S21, std S21, and list of all runs
        """
        # Find all CSV files in the folder
        csv_files = sorted(glob.glob(f"{folder_path}/*.csv"))
        
        if not csv_files:
            print(f"     ⚠️ No CSV files found in {folder_path}")
            return None, None, None
        
        # Group by path number
        path_data = {1: [], 2: [], 3: [], 4: []}
        
        for file in csv_files:
            # Extract path number from filename
            if 'path1' in file or 'Path 1' in file:
                path_num = 1
            elif 'path2' in file or 'Path 2' in file:
                path_num = 2
            elif 'path3' in file or 'Path 3' in file:
                path_num = 3
            elif 'path4' in file or 'Path 4' in file:
                path_num = 4
            else:
                continue
                
            try:
                df = pd.read_csv(file)
                s21_values = df['S21_dB'].values
                
                # Ensure correct length
                if len(s21_values) > self.num_points:
                    s21_values = s21_values[:self.num_points]
                elif len(s21_values) < self.num_points:
                    s21_values = np.pad(s21_values, (0, self.num_points - len(s21_values)), 'edge')
                
                path_data[path_num].append(s21_values)
            except Exception as e:
                print(f"     ⚠️ Error loading {file}: {e}")
        
        # Check if we have data for all paths
        for path_num in range(1, 5):
            if not path_data[path_num]:
                print(f"     ⚠️ No data for Path {path_num}")
                return None, None, None
        
        # Calculate statistics
        mean_data = {}
        std_data = {}
        all_runs = {}
        
        for path_num in range(1, 5):
            runs_array = np.array(path_data[path_num])
            mean_data[path_num] = np.mean(runs_array, axis=0)
            std_data[path_num] = np.std(runs_array, axis=0)
            all_runs[path_num] = runs_array
        
        return mean_data, std_data, all_runs
    
    def create_raw_features_vector(self, mean_data):
        """Create flattened feature vector for XGBoost from mean data"""
        features = []
        for path_num in range(1, 5):
            if path_num in mean_data:
                features.extend(mean_data[path_num])
            else:
                features.extend([np.nan] * self.num_points)
        return np.array(features)
    
    def create_2d_image(self, mean_data):
        """Create 2D image matrix for CNN from mean data"""
        image_matrix = np.zeros((4, self.num_points))
        
        for path_num in range(1, 5):
            if path_num in mean_data:
                image_matrix[path_num-1, :] = mean_data[path_num]
            else:
                image_matrix[path_num-1, :] = np.nan
                
        return image_matrix
    
    def normalize_image(self, image_matrix):
        """Normalize to [0, 1] range for CNN"""
        image_matrix = np.clip(image_matrix, -60, 0)
        img_min = -60
        img_max = 0
        normalized = (image_matrix - img_min) / (img_max - img_min)
        return normalized

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

def main():
    print("="*70)
    print("PULMO AI: Convert S21 Scans to ML-Ready Formats")
    print("(Now with MULTIPLE RUN AGGREGATION)")
    print("="*70)
    
    # Initialize processor
    processor = S21DataProcessor()
    augmenter = DataAugmenter()
    
    # Find the latest phantom data folder
    data_folders = sorted(Path('.').glob('phantom_data_*'))
    if not data_folders:
        print("❌ No phantom_data_* folders found!")
        return
    
    latest_folder = data_folders[-1]
    print(f"\n📁 Using data from: {latest_folder}")
    
    # Create output directories
    timestamp = latest_folder.name.replace('phantom_data_', '')
    output_base = f"ml_dataset_{timestamp}"
    os.makedirs(output_base, exist_ok=True)
    
    # Create subdirectories
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
    conditions = [
        {'folder': '01_baseline_air', 'class': 'baseline', 'tumor_size': 0},
        {'folder': '02_healthy_phantom_1', 'class': 'healthy', 'tumor_size': 0},
        {'folder': '03_healthy_phantom_2', 'class': 'healthy', 'tumor_size': 0},
        {'folder': '04_tumor_phantom', 'class': 'tumor', 'tumor_size': 20}
    ]
    
    # Data containers
    raw_features_list = []
    raw_labels_list = []
    raw_tumor_sizes_list = []
    cnn_images_list = []
    cnn_labels_list = []
    cnn_tumor_sizes_list = []
    metadata_records = []
    
    print("\n📥 Processing conditions (aggregating multiple runs):")
    
    # Process each condition
    for cond in conditions:
        folder_path = latest_folder / cond['folder']
        if not folder_path.exists():
            print(f"  ⚠️ Skipping {cond['folder']} - not found")
            continue
        
        print(f"\n  📍 {cond['folder']} → class: {cond['class']}")
        
        # Load ALL runs and get statistics
        mean_data, std_data, all_runs = processor.load_all_runs_from_folder(folder_path)
        
        if mean_data is None:
            print(f"     ❌ No usable data")
            continue
        
        # Count number of runs
        num_runs = len(all_runs[1]) if all_runs[1] else 0
        print(f"     📊 Aggregated {num_runs} runs per path")
        
        # Calculate run-to-run variation
        path_variations = []
        for path_num in range(1, 5):
            if std_data[path_num] is not None:
                path_variations.append(np.mean(std_data[path_num]))
        avg_variation = np.mean(path_variations) if path_variations else 0
        print(f"     📊 Average run-to-run std dev: {avg_variation:.2f} dB")
        
        # Create unique ID
        sample_id = f"{cond['class']}_{cond['folder']}"
        
        # 1. Create raw features for XGBoost (using MEAN data)
        raw_features = processor.create_raw_features_vector(mean_data)
        raw_features_list.append(raw_features)
        raw_labels_list.append(cond['class'])
        raw_tumor_sizes_list.append(cond['tumor_size'])
        
        # 2. Create 2D image for CNN (using MEAN data)
        cnn_image = processor.create_2d_image(mean_data)
        cnn_image_norm = processor.normalize_image(cnn_image)
        cnn_images_list.append(cnn_image_norm)
        cnn_labels_list.append(cond['class'])
        cnn_tumor_sizes_list.append(cond['tumor_size'])
        
        # Save raw features
        features_df = pd.DataFrame([raw_features])
        features_df.columns = [f'freq_{i}' for i in range(len(raw_features))]
        features_df['class'] = cond['class']
        features_df['tumor_size_mm'] = cond['tumor_size']
        features_df['num_runs'] = num_runs
        features_df['avg_variation_db'] = avg_variation
        features_df.to_csv(f"{dirs['raw_features']}/{sample_id}_features.csv", index=False)
        
        # Save CNN image
        np.save(f"{dirs['cnn_images']}/{sample_id}_image.npy", cnn_image_norm)
        
        # Save visualization with error bars
        plt.figure(figsize=(12, 8))
        
        # Plot mean with std as shaded region
        for path_num in range(1, 5):
            mean_vals = mean_data[path_num]
            std_vals = std_data[path_num] if std_data[path_num] is not None else np.zeros_like(mean_vals)
            
            plt.subplot(2, 2, path_num)
            plt.plot(processor.frequencies, mean_vals, 'b-', linewidth=2, label='Mean')
            plt.fill_between(processor.frequencies, 
                            mean_vals - std_vals, 
                            mean_vals + std_vals, 
                            alpha=0.3, color='blue', label='±1σ')
            plt.xlabel('Frequency (GHz)')
            plt.ylabel('S21 (dB)')
            plt.title(f'Path {path_num} (Mean of {num_runs} runs)')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        plt.suptitle(f"{cond['class'].upper()}: {cond['folder']}\nRun-to-run variation: {avg_variation:.2f} dB", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{dirs['visualizations']}/{sample_id}_with_stats.png", dpi=150)
        plt.close()
        
        # Save metadata
        metadata = {
            'sample_id': sample_id,
            'class': cond['class'],
            'tumor_size_mm': cond['tumor_size'],
            'condition_folder': cond['folder'],
            'num_runs': num_runs,
            'avg_variation_db': float(avg_variation),
            'freq_start_ghz': processor.freq_start,
            'freq_stop_ghz': processor.freq_stop,
            'num_freq_points': processor.num_points,
            'mean_s21_db': float(np.nanmean(cnn_image)),
            'std_s21_db': float(np.nanstd(cnn_image))
        }
        metadata_records.append(metadata)
        
        with open(f"{dirs['cnn_images']}/{sample_id}_meta.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"     ✅ Saved aggregated data: {num_runs} runs → 1 sample")
        
        # Generate augmentations (now with more confidence since we have statistical validation)
        print(f"     🔄 Generating augmentations...")
        
        augmentations = {
            'noise_01': lambda x: augmenter.add_gaussian_noise(x, sigma=0.01),
            'noise_02': lambda x: augmenter.add_gaussian_noise(x, sigma=0.02),
            'blur_1': lambda x: augmenter.gaussian_blur(x, sigma=1.0),
            'blur_2': lambda x: augmenter.gaussian_blur(x, sigma=2.0),
            'shift_pos_5': lambda x: augmenter.frequency_shift(x, 5),
            'shift_neg_5': lambda x: augmenter.frequency_shift(x, -5),
            'shift_pos_10': lambda x: augmenter.frequency_shift(x, 10),
            'shift_neg_10': lambda x: augmenter.frequency_shift(x, -10),
            'flip_horizontal': lambda x: augmenter.horizontal_flip(x),
            'drift_small': lambda x: augmenter.add_baseline_drift(x, 0.03),
            'drift_medium': lambda x: augmenter.add_baseline_drift(x, 0.05),
            'random_combo_1': lambda x: augmenter.random_combination(x),
            'random_combo_2': lambda x: augmenter.random_combination(x)
        }
        
        for aug_name, aug_func in augmentations.items():
            aug_image = aug_func(cnn_image_norm.copy())
            aug_image = np.clip(aug_image, 0, 1)
            
            aug_filename = f"{dirs['cnn_augmented']}/{sample_id}_{aug_name}.npy"
            np.save(aug_filename, aug_image)
        
        print(f"     ✅ Generated {len(augmentations)} augmentations")
    
    # Create master datasets
    print("\n" + "="*70)
    print("📊 CREATING MASTER DATASETS")
    print("="*70)
    
    if not raw_features_list:
        print("❌ No data processed!")
        return
    
    # 1. XGBoost Dataset
    xgb_df = pd.DataFrame(raw_features_list)
    xgb_df.columns = [f'freq_{i}' for i in range(xgb_df.shape[1])]
    xgb_df['class'] = raw_labels_list
    xgb_df['tumor_size_mm'] = raw_tumor_sizes_list
    xgb_df['class_label'] = xgb_df['class'].map({'baseline': 0, 'healthy': 1, 'tumor': 2})
    
    xgb_df.to_csv(f"{output_base}/pulmo_xgboost_dataset.csv", index=False)
    print(f"\n✅ XGBoost Dataset: {xgb_df.shape}")
    print(f"   • Samples: {xgb_df.shape[0]}")
    print(f"   • Each sample = average of {metadata_records[0]['num_runs']} runs")
    print(f"   • Classes: {xgb_df['class'].value_counts().to_dict()}")
    
    # Create train/validation splits
    X = xgb_df[[c for c in xgb_df.columns if c.startswith('freq_')]].values
    y_class = xgb_df['class_label'].values
    y_size = xgb_df['tumor_size_mm'].values
    
    X_train, X_val, y_class_train, y_class_val, y_size_train, y_size_val = train_test_split(
        X, y_class, y_size, test_size=0.2, random_state=42, stratify=y_class
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
    X_train_scaled = scaler.fit_transform(X_train)
    with open(f"{dirs['models']}/xgboost_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\n✅ XGBoost splits saved:")
    print(f"   • Train: {X_train.shape[0]} samples")
    print(f"   • Validation: {X_val.shape[0]} samples")
    
    # 2. CNN Dataset
    if cnn_images_list:
        cnn_images_array = np.array(cnn_images_list).reshape(-1, 4, 201, 1)
        class_to_idx = {'baseline': 0, 'healthy': 1, 'tumor': 2}
        cnn_class_idx = np.array([class_to_idx[label] for label in cnn_labels_list])
        cnn_sizes_array = np.array(cnn_tumor_sizes_list)
        
        np.save(f"{dirs['cnn_images']}/cnn_images.npy", cnn_images_array)
        np.save(f"{dirs['cnn_images']}/cnn_labels.npy", cnn_class_idx)
        np.save(f"{dirs['cnn_images']}/cnn_tumor_sizes.npy", cnn_sizes_array)
        
        print(f"\n✅ CNN Dataset: {cnn_images_array.shape}")
        print(f"   • Each image = average of {metadata_records[0]['num_runs']} runs")
    
    # Save final metadata
    with open(f"{output_base}/dataset_metadata.json", 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'source_folder': str(latest_folder),
            'scans_per_condition': metadata_records[0]['num_runs'] if metadata_records else 0,
            'xgboost_features': {
                'samples': len(xgb_df),
                'feature_dim': X.shape[1],
                'classes': xgb_df['class'].value_counts().to_dict()
            },
            'cnn_dataset': {
                'samples': len(cnn_images_list),
                'image_shape': [4, 201, 1]
            },
            'metadata_records': metadata_records
        }, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("✅ DATASET CREATION COMPLETE!")
    print("="*70)
    print(f"\n📊 SCIENTIFIC VALIDATION:")
    print(f"   • Each sample = average of {metadata_records[0]['num_runs']} repeat measurements")
    print(f"   • Run-to-run variation documented: {metadata_records[0]['avg_variation_db']:.2f} dB")
    print(f"   • Statistical uncertainty quantified")
    print(f"\n📁 Output folder: {output_base}/")
    
    print("\n🚀 NEXT STEPS:")
    print("  1. Review the quality plots with error bars")
    print("  2. Upload to Kaggle for training")
    print("  3. Present your reproducibility metrics to judges")

if __name__ == "__main__":
    main()
