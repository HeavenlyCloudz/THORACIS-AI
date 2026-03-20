# create_ml_images_updated.py
"""
Convert S21 Scans to ML-Ready Formats
Generates:
1. Raw features CSV for XGBoost (flattened S21 data)
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
from scipy.ndimage import gaussian_filter, rotate, zoom
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

class S21DataProcessor:
    def __init__(self, freq_start=2.0, freq_stop=3.0, num_points=201):
        self.freq_start = freq_start
        self.freq_stop = freq_stop
        self.num_points = num_points
        self.frequencies = np.linspace(freq_start, freq_stop, num_points)
        
    def load_path_data(self, folder_path):
        """Load all 4 paths from a condition folder"""
        path_data = {}
        csv_files = sorted(Path(folder_path).glob('*.csv'))
        
        for i, file in enumerate(csv_files[:4]):  # Take first 4 paths
            df = pd.read_csv(file)
            path_num = i + 1
            # Extract S21 values
            s21_values = df['S21_dB'].values
            # Ensure correct length
            if len(s21_values) > self.num_points:
                s21_values = s21_values[:self.num_points]
            elif len(s21_values) < self.num_points:
                # Pad if necessary (shouldn't happen with your setup)
                s21_values = np.pad(s21_values, (0, self.num_points - len(s21_values)), 'edge')
            
            path_data[path_num] = s21_values
            
        return path_data
    
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
        Values = S21 in dB
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
        # Clip extreme values
        image_matrix = np.clip(image_matrix, -60, 0)
        
        # Normalize to [0, 1]
        img_min = -60
        img_max = 0
        normalized = (image_matrix - img_min) / (img_max - img_min)
        
        return normalized
    
    def normalize_features(self, features_matrix):
        """Normalize raw features (will be used with StandardScaler later)"""
        return features_matrix  # Return as-is, scaling done per-feature in XGBoost

class DataAugmenter:
    """Augment 2D image data for CNN training"""
    
    @staticmethod
    def add_gaussian_noise(image, sigma=0.01):
        """Add Gaussian noise"""
        noise = np.random.normal(0, sigma, image.shape)
        return np.clip(image + noise, 0, 1)
    
    @staticmethod
    def gaussian_blur(image, sigma=1.0):
        """Apply Gaussian blur"""
        return gaussian_filter(image, sigma=sigma)
    
    @staticmethod
    def frequency_shift(image, shift_pixels=5):
        """Shift along frequency axis"""
        return np.roll(image, shift_pixels, axis=1)
    
    @staticmethod
    def horizontal_flip(image):
        """Flip horizontally (swap paths 1-2 or 3-4 symmetrically)"""
        flipped = image.copy()
        flipped[0:2] = image[1::-1]  # Swap path 1 and 2
        flipped[2:4] = image[3:1:-1]  # Swap path 3 and 4
        return flipped
    
    @staticmethod
    def add_baseline_drift(image, drift_strength=0.05):
        """Add slow varying baseline drift"""
        drift = np.linspace(0, drift_strength, image.shape[1])
        return image + drift[np.newaxis, :]
    
    @staticmethod
    def random_combination(image):
        """Apply random combination of augmentations"""
        aug = image.copy()
        
        # Randomly apply augmentations (50% chance each)
        if random.random() > 0.5:
            aug = DataAugmenter.add_gaussian_noise(aug, sigma=random.uniform(0.005, 0.02))
        if random.random() > 0.5:
            aug = DataAugmenter.gaussian_blur(aug, sigma=random.uniform(0.5, 2.0))
        if random.random() > 0.3:  # 70% chance
            shift = random.randint(-10, 10)
            aug = DataAugmenter.frequency_shift(aug, shift)
        if random.random() > 0.7:  # 30% chance
            aug = DataAugmenter.horizontal_flip(aug)
            
        return np.clip(aug, 0, 1)

def create_tumor_size_label(condition_folder, tumor_sizes=None):
    """
    Create tumor size label based on condition
    For healthy/baseline: size = 0 mm
    For tumor: assign size based on your phantom (e.g., 20mm)
    """
    if 'tumor' in condition_folder.lower():
        # You can modify this based on your actual tumor phantom size
        return 20.0  # mm - change to your actual tumor size
    else:
        return 0.0  # mm - no tumor

def main():
    print("="*70)
    print("PULMO AI: Convert S21 Scans to ML-Ready Formats")
    print("="*70)
    print("\n📊 Generating:")
    print("   • Raw features for XGBoost (804-dim vectors)")
    print("   • 2D images for CNN (4×201 matrices)")
    print("   • Augmented dataset for robust training")
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
    
    # Define conditions and their classes
    conditions = [
        {'folder': '01_baseline_air', 'class': 'baseline', 'tumor_size': 0},
        {'folder': '02_healthy_phantom_1', 'class': 'healthy', 'tumor_size': 0},
        {'folder': '03_healthy_phantom_2', 'class': 'healthy', 'tumor_size': 0},
        {'folder': '04_tumor_phantom', 'class': 'tumor', 'tumor_size': 20}  # Adjust size!
    ]
    
    # Data containers
    raw_features_list = []
    raw_labels_list = []
    raw_tumor_sizes_list = []
    cnn_images_list = []
    cnn_labels_list = []
    cnn_tumor_sizes_list = []
    metadata_records = []
    
    print("\n📥 Processing conditions:")
    
    # Process each condition
    for cond in conditions:
        folder_path = latest_folder / cond['folder']
        if not folder_path.exists():
            print(f"  ⚠️  Skipping {cond['folder']} - not found")
            continue
        
        print(f"\n  📍 {cond['folder']} → class: {cond['class']}")
        
        # Load path data
        path_data = processor.load_path_data(folder_path)
        
        if not path_data:
            print(f"     No data found")
            continue
        
        # Create unique ID
        sample_id = f"{cond['class']}_{cond['folder']}"
        
        # 1. Create raw features for XGBoost
        raw_features = processor.create_raw_features_vector(path_data)
        raw_features_list.append(raw_features)
        raw_labels_list.append(cond['class'])
        raw_tumor_sizes_list.append(cond['tumor_size'])
        
        # 2. Create 2D image for CNN
        cnn_image = processor.create_2d_image(path_data)
        cnn_image_norm = processor.normalize_image(cnn_image)
        cnn_images_list.append(cnn_image_norm)
        cnn_labels_list.append(cond['class'])
        cnn_tumor_sizes_list.append(cond['tumor_size'])
        
        # Save raw features as CSV
        features_df = pd.DataFrame([raw_features])
        features_df.columns = [f'freq_{i}' for i in range(len(raw_features))]
        features_df['class'] = cond['class']
        features_df['tumor_size_mm'] = cond['tumor_size']
        features_df.to_csv(f"{dirs['raw_features']}/{sample_id}_features.csv", index=False)
        
        # Save CNN image as numpy array
        np.save(f"{dirs['cnn_images']}/{sample_id}_image.npy", cnn_image_norm)
        
        # Save visualization
        plt.figure(figsize=(10, 3))
        plt.imshow(cnn_image_norm, aspect='auto', cmap='viridis', 
                  extent=[processor.freq_start, processor.freq_stop, 4.5, 0.5])
        plt.colorbar(label='Normalized S21')
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Path Number')
        plt.yticks([1, 2, 3, 4])
        plt.title(f"{cond['class'].upper()}: {cond['folder']}")
        plt.tight_layout()
        plt.savefig(f"{dirs['visualizations']}/{sample_id}.png", dpi=150)
        plt.close()
        
        # Save metadata
        metadata = {
            'sample_id': sample_id,
            'class': cond['class'],
            'tumor_size_mm': cond['tumor_size'],
            'condition_folder': cond['folder'],
            'num_paths': len(path_data),
            'freq_start_ghz': processor.freq_start,
            'freq_stop_ghz': processor.freq_stop,
            'num_freq_points': processor.num_points,
            'mean_s21_db': float(np.nanmean(cnn_image)),
            'std_s21_db': float(np.nanstd(cnn_image))
        }
        metadata_records.append(metadata)
        
        with open(f"{dirs['cnn_images']}/{sample_id}_meta.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"     ✅ Saved: {sample_id}")
        print(f"        • Raw features: 804-dim")
        print(f"        • CNN image: 4×201")
        
        # Generate CNN augmentations
        print(f"     🔄 Generating augmentations...")
        
        # Define augmentation techniques
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
    
    # Create master datasets
    print("\n" + "="*70)
    print("📊 CREATING MASTER DATASETS")
    print("="*70)
    
    # 1. XGBoost Dataset (Raw Features)
    xgb_df = pd.DataFrame(raw_features_list)
    xgb_df.columns = [f'freq_{i}' for i in range(xgb_df.shape[1])]
    xgb_df['class'] = raw_labels_list
    xgb_df['tumor_size_mm'] = raw_tumor_sizes_list
    xgb_df['class_label'] = xgb_df['class'].map({'baseline': 0, 'healthy': 1, 'tumor': 2})
    
    # Save full XGBoost dataset
    xgb_df.to_csv(f"{output_base}/pulmo_xgboost_dataset.csv", index=False)
    print(f"\n✅ XGBoost Dataset: {xgb_df.shape}")
    print(f"   • Samples: {xgb_df.shape[0]}")
    print(f"   • Features: {xgb_df.shape[1]-3} raw S21 values")
    print(f"   • Classes: {xgb_df['class'].value_counts().to_dict()}")
    
    # Create train/validation split for XGBoost
    X = xgb_df[[c for c in xgb_df.columns if c.startswith('freq_')]].values
    y_class = xgb_df['class_label'].values
    y_size = xgb_df['tumor_size_mm'].values
    
    X_train, X_val, y_class_train, y_class_val, y_size_train, y_size_val = train_test_split(
        X, y_class, y_size, test_size=0.2, random_state=42, stratify=y_class
    )
    
    # Save XGBoost splits
    np.save(f"{dirs['models']}/X_train.npy", X_train)
    np.save(f"{dirs['models']}/X_val.npy", X_val)
    np.save(f"{dirs['models']}/y_class_train.npy", y_class_train)
    np.save(f"{dirs['models']}/y_class_val.npy", y_class_val)
    np.save(f"{dirs['models']}/y_size_train.npy", y_size_train)
    np.save(f"{dirs['models']}/y_size_val.npy", y_size_val)
    
    # Save feature scaler (for XGBoost preprocessing if needed)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    with open(f"{dirs['models']}/xgboost_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\n✅ XGBoost splits saved:")
    print(f"   • Train: {X_train.shape[0]} samples")
    print(f"   • Validation: {X_val.shape[0]} samples")
    
    # 2. CNN Dataset (2D Images)
    cnn_images_array = np.array(cnn_images_list)
    cnn_labels_array = np.array(cnn_labels_list)
    cnn_sizes_array = np.array(cnn_tumor_sizes_list)
    
    # Add channel dimension (4, 201, 1)
    cnn_images_array = cnn_images_array.reshape(-1, 4, 201, 1)
    
    # Create class labels
    class_to_idx = {'baseline': 0, 'healthy': 1, 'tumor': 2}
    cnn_class_idx = np.array([class_to_idx[label] for label in cnn_labels_array])
    
    # Save full CNN dataset
    np.save(f"{dirs['cnn_images']}/cnn_images.npy", cnn_images_array)
    np.save(f"{dirs['cnn_images']}/cnn_labels.npy", cnn_class_idx)
    np.save(f"{dirs['cnn_images']}/cnn_tumor_sizes.npy", cnn_sizes_array)
    
    print(f"\n✅ CNN Dataset (original): {cnn_images_array.shape}")
    print(f"   • Samples: {cnn_images_array.shape[0]}")
    print(f"   • Image shape: 4×201×1")
    print(f"   • Classes: {dict(zip(*np.unique(cnn_labels_array, return_counts=True)))}")
    
    # Load and count augmentations
    aug_files = list(Path(dirs['cnn_augmented']).glob('*.npy'))
    print(f"\n✅ CNN Augmentations: {len(aug_files)} files")
    
    # Create augmented dataset array (load a few to check)
    if aug_files:
        sample_aug = np.load(aug_files[0])
        print(f"   • Sample augmented shape: {sample_aug.shape}")
        
        # Calculate total augmented samples
        unique_originals = set()
        for f in aug_files:
            parts = f.stem.split('_')
            unique_originals.add('_'.join(parts[:3]))  # class_condition
        print(f"   • Original samples augmented: {len(unique_originals)}")
        print(f"   • Total augmented samples: {len(aug_files)}")
    
    # Save metadata
    with open(f"{output_base}/dataset_metadata.json", 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'source_folder': str(latest_folder),
            'num_conditions': len(conditions),
            'xgboost_features': {
                'samples': len(xgb_df),
                'feature_dim': X.shape[1],
                'classes': xgb_df['class'].value_counts().to_dict()
            },
            'cnn_dataset': {
                'samples': len(cnn_images_array),
                'image_shape': [4, 201, 1],
                'classes': dict(zip(*np.unique(cnn_labels_array, return_counts=True)))
            },
            'augmentations': {
                'total_files': len(aug_files),
                'techniques': list(augmentations.keys())
            },
            'metadata_records': metadata_records
        }, f, indent=2)
    
    # Create Kaggle upload script
    create_kaggle_upload_script(output_base, dirs)
    
    # Create README
    create_readme(output_base, conditions)
    
    print("\n" + "="*70)
    print("✅ DATASET CREATION COMPLETE!")
    print("="*70)
    print(f"\n📁 Output folder: {output_base}/")
    print("\nFolder structure:")
    for dir_name, dir_path in dirs.items():
        print(f"  ├── {os.path.basename(dir_path)}/")
    
    print(f"\n📊 Key files:")
    print(f"  • XGBoost dataset: pulmo_xgboost_dataset.csv")
    print(f"  • CNN images: 02_cnn_images/cnn_images.npy")
    print(f"  • Train/val splits: 05_models/")
    print(f"  • Metadata: dataset_metadata.json")
    print(f"  • Kaggle uploader: upload_to_kaggle.py")
    
    print("\n🚀 NEXT STEPS:")
    print("  1. Review the dataset in the output folder")
    print("  2. Run the Kaggle uploader script")
    print("  3. Train on Kaggle using the provided notebooks")
    print("="*70)

def create_kaggle_upload_script(output_base, dirs):
    """Create a script to help upload to Kaggle"""
    
    kaggle_script = f'''
# upload_to_kaggle.py
"""
Script to prepare and upload PULMO AI dataset to Kaggle
Run: python upload_to_kaggle.py
"""

import os
import zipfile
from pathlib import Path

def create_kaggle_dataset():
    """Create a zip file ready for Kaggle upload"""
    
    output_base = "{output_base}"
    
    # Files to include
    files_to_zip = [
        "pulmo_xgboost_dataset.csv",
        "dataset_metadata.json",
        "02_cnn_images/cnn_images.npy",
        "02_cnn_images/cnn_labels.npy",
        "02_cnn_images/cnn_tumor_sizes.npy",
        "05_models/X_train.npy",
        "05_models/X_val.npy",
        "05_models/y_class_train.npy",
        "05_models/y_class_val.npy",
        "05_models/y_size_train.npy",
        "05_models/y_size_val.npy"
    ]
    
    # Create README for Kaggle
    readme_content = \"\"\"# PULMO AI Dataset for Kaggle

## Dataset Overview
This dataset contains microwave scattering parameter (S21) measurements from lung phantoms for AI-driven tumor detection.

## Contents
- **pulmo_xgboost_dataset.csv**: Raw S21 features (804-dim) for XGBoost
- **cnn_images.npy**: 2D images (4×201×1) for CNN training
- **cnn_labels.npy**: Class labels (0=baseline, 1=healthy, 2=tumor)
- **cnn_tumor_sizes.npy**: Tumor size in mm (0 for no tumor)
- **Train/validation splits**: Pre-split data for reproducible results

## Classes
- Baseline (air): 0
- Healthy phantom: 1  
- Tumor phantom (20mm): 2

## Usage
See the accompanying notebooks for XGBoost and CNN training examples.
\"\"\"
    
    with open(f"{output_base}/README.md", 'w') as f:
        f.write(readme_content)
    
    # Create zip file
    zip_filename = f"{output_base}_kaggle.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for file_pattern in files_to_zip:
            file_path = Path(output_base) / file_pattern
            if file_path.exists():
                zipf.write(file_path, arcname=file_pattern)
                print(f"✅ Added: {file_pattern}")
        
        # Add README
        zipf.write(f"{output_base}/README.md", arcname="README.md")
    
    print(f"\n✅ Kaggle dataset created: {zip_filename}")
    print(f"   Size: {os.path.getsize(zip_filename) / 1e6:.1f} MB")
    print("\n📤 Upload to Kaggle:")
    print("   1. Go to kaggle.com/datasets")
    print("   2. Click 'New Dataset'")
    print("   3. Upload this zip file")
    print("   4. Add description and make public/private")

if __name__ == "__main__":
    create_kaggle_dataset()
'''
    
    with open(f"{output_base}/upload_to_kaggle.py", 'w') as f:
        f.write(kaggle_script)

def create_readme(output_base, conditions):
    """Create comprehensive README"""
    
    readme = f"""# PULMO AI - Microwave Imaging Dataset

## 📊 Dataset Description
This dataset contains S21 parameter measurements from a 4-antenna microwave imaging system designed for lung tumor detection. The data is formatted for two machine learning approaches:
1. **XGBoost**: Raw S21 features (804-dimensional vectors)
2. **CNN**: 2D images (4 paths × 201 frequency points)

## 🎯 Classes
"""
    for cond in conditions:
        readme += f"- **{cond['class']}**: {cond['folder']} (tumor size: {cond['tumor_size']}mm)\n"
    
    readme += f"""
## 📁 File Structure
