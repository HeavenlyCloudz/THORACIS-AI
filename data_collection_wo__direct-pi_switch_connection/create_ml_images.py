# create_ml_images.py
"""
Converts S21 scans into 2D image-like matrices for CNN training
Each "image" = 4 paths × 201 frequency points
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import json
from scipy.ndimage import gaussian_filter, rotate, zoom
import random

class S21ToImageConverter:
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
            # Extract path number from filename
            path_num = i + 1
            path_data[path_num] = df['S21_dB'].values
            
        return path_data
    
    def create_2d_image(self, path_data):
        """
        Create 2D image matrix:
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
        # Clip extreme values (optional)
        image_matrix = np.clip(image_matrix, -60, 0)
        
        # Normalize to [0, 1]
        img_min = -60
        img_max = 0
        normalized = (image_matrix - img_min) / (img_max - img_min)
        
        return normalized
    
    def save_as_image_file(self, image_matrix, filename, title=None):
        """Save the 2D matrix as a visual image"""
        plt.figure(figsize=(12, 4))
        
        plt.imshow(image_matrix, aspect='auto', cmap='viridis', 
                   extent=[self.freq_start, self.freq_stop, 4.5, 0.5])
        plt.colorbar(label='S21 (dB)')
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Path Number')
        plt.yticks([1, 2, 3, 4])
        
        if title:
            plt.title(title)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
    def save_as_numpy(self, image_matrix, filename):
        """Save as numpy array for ML training"""
        np.save(filename, image_matrix)
        
    def save_metadata(self, metadata, filename):
        """Save metadata about the scan"""
        with open(filename, 'w') as f:
            json.dump(metadata, f, indent=2)

class DataAugmenter:
    """Augment the 2D image data for ML training"""
    
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
        """Shift along frequency axis (simulates slight frequency drift)"""
        return np.roll(image, shift_pixels, axis=1)
    
    @staticmethod
    def amplitude_scale(image, scale_factor=0.9):
        """Scale all amplitudes"""
        return image * scale_factor
    
    @staticmethod
    def path_mix(image, alpha=0.1):
        """Mix adjacent paths (simulates cross-coupling)"""
        augmented = image.copy()
        for i in range(3):
            mix = alpha * image[i+1] + (1-alpha) * image[i]
            augmented[i] = mix
        return augmented
    
    @staticmethod
    def add_baseline_drift(image, drift_strength=0.05):
        """Add slow varying baseline drift"""
        drift = np.linspace(0, drift_strength, image.shape[1])
        return image + drift[np.newaxis, :]
    
    @staticmethod
    def random_combination(image):
        """Apply random combination of augmentations"""
        aug = image.copy()
        
        # Randomly apply augmentations
        if random.random() > 0.5:
            aug = DataAugmenter.add_gaussian_noise(aug, sigma=random.uniform(0.005, 0.02))
        if random.random() > 0.5:
            aug = DataAugmenter.gaussian_blur(aug, sigma=random.uniform(0.5, 2.0))
        if random.random() > 0.5:
            shift = random.randint(-10, 10)
            aug = DataAugmenter.frequency_shift(aug, shift)
        if random.random() > 0.5:
            scale = random.uniform(0.85, 1.15)
            aug = DataAugmenter.amplitude_scale(aug, scale)
        if random.random() > 0.7:  # Less common
            aug = DataAugmenter.path_mix(aug, alpha=random.uniform(0.05, 0.2))
            
        return np.clip(aug, 0, 1)

def main():
    print("=== PULMO AI: Convert S21 Scans to ML Images ===\n")
    
    # Initialize converter
    converter = S21ToImageConverter()
    augmenter = DataAugmenter()
    
    # Find the latest phantom data folder
    data_folders = sorted(Path('.').glob('phantom_data_*'))
    if not data_folders:
        print("❌ No phantom_data_* folders found!")
        return
    
    latest_folder = data_folders[-1]
    print(f"📁 Using data from: {latest_folder}")
    
    # Create output directories
    output_base = f"ml_dataset_{latest_folder.name.replace('phantom_data_', '')}"
    os.makedirs(output_base, exist_ok=True)
    
    # Subfolders for each class
    classes = ['baseline', 'healthy', 'tumor']
    for class_name in classes:
        os.makedirs(f"{output_base}/{class_name}/images", exist_ok=True)
        os.makedirs(f"{output_base}/{class_name}/numpy", exist_ok=True)
        os.makedirs(f"{output_base}/{class_name}/augmented", exist_ok=True)
    
    print("\n📊 Processing conditions:")
    
    # Process each condition
    conditions = [
        ('01_baseline_air', 'baseline'),
        ('02_healthy_phantom_1', 'healthy'),
        ('03_healthy_phantom_2', 'healthy'),
        ('04_tumor_phantom', 'tumor')
    ]
    
    dataset_stats = {}
    
    for condition_folder, class_name in conditions:
        folder_path = latest_folder / condition_folder
        if not folder_path.exists():
            print(f"⚠️  Skipping {condition_folder} - not found")
            continue
            
        print(f"\n  📍 {condition_folder} → class: {class_name}")
        
        # Load path data
        path_data = converter.load_path_data(folder_path)
        
        if not path_data:
            print(f"     No data found")
            continue
        
        # Create 2D image
        image = converter.create_2d_image(path_data)
        
        # Normalize
        image_norm = converter.normalize_image(image)
        
        # Create unique ID
        sample_id = f"{class_name}_{condition_folder}"
        
        # Save visual image
        img_file = f"{output_base}/{class_name}/images/{sample_id}.png"
        converter.save_as_image_file(image_norm, img_file, 
                                     title=f"{class_name.upper()}: {condition_folder}")
        
        # Save numpy array
        numpy_file = f"{output_base}/{class_name}/numpy/{sample_id}.npy"
        converter.save_as_numpy(image_norm, numpy_file)
        
        # Save metadata
        metadata = {
            'class': class_name,
            'condition': condition_folder,
            'freq_start': converter.freq_start,
            'freq_stop': converter.freq_stop,
            'num_points': converter.num_points,
            'paths_available': list(path_data.keys()),
            'mean_s21': float(np.nanmean(image)),
            'std_s21': float(np.nanstd(image)),
            'min_s21': float(np.nanmin(image)),
            'max_s21': float(np.nanmax(image))
        }
        
        with open(f"{output_base}/{class_name}/numpy/{sample_id}_meta.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"     ✅ Saved: {sample_id}")
        
        # Generate augmented versions
        print(f"     🔄 Generating augmentations...")
        
        augmentations = {
            'noise': lambda x: augmenter.add_gaussian_noise(x, sigma=0.01),
            'blur': lambda x: augmenter.gaussian_blur(x, sigma=1.0),
            'shift_pos': lambda x: augmenter.frequency_shift(x, 5),
            'shift_neg': lambda x: augmenter.frequency_shift(x, -5),
            'scale_up': lambda x: augmenter.amplitude_scale(x, 1.1),
            'scale_down': lambda x: augmenter.amplitude_scale(x, 0.9),
            'mix_paths': lambda x: augmenter.path_mix(x, alpha=0.1),
            'drift': lambda x: augmenter.add_baseline_drift(x, 0.03),
        }
        
        for aug_name, aug_func in augmentations.items():
            aug_image = aug_func(image_norm.copy())
            aug_image = np.clip(aug_image, 0, 1)
            
            aug_file = f"{output_base}/{class_name}/augmented/{sample_id}_{aug_name}.npy"
            np.save(aug_file, aug_image)
            
            # Save visual of augmented version (optional, first few only)
            if class_name == 'tumor' and aug_name in ['noise', 'blur']:
                aug_img_file = f"{output_base}/{class_name}/augmented/{sample_id}_{aug_name}.png"
                converter.save_as_image_file(aug_image, aug_img_file,
                                           title=f"{class_name}: {aug_name}")
        
        # Store stats
        if class_name not in dataset_stats:
            dataset_stats[class_name] = 0
        dataset_stats[class_name] += 1
    
    # Print dataset summary
    print("\n" + "="*60)
    print("📊 DATASET SUMMARY")
    print("="*60)
    
    total_original = 0
    total_augmented = 0
    
    for class_name in classes:
        if class_name in dataset_stats:
            original = dataset_stats[class_name]
            augmented = original * len(augmentations)
            total_original += original
            total_augmented += augmented
            
            print(f"\n{class_name.upper()}:")
            print(f"  Original samples: {original}")
            print(f"  Augmented samples: {augmented}")
            print(f"  Total: {original + augmented}")
    
    print(f"\n{'='*60}")
    print(f"TOTAL ORIGINAL SAMPLES: {total_original}")
    print(f"TOTAL AUGMENTED SAMPLES: {total_augmented}")
    print(f"GRAND TOTAL: {total_original + total_augmented}")
    print(f"{'='*60}")
    
    print(f"\n✅ Dataset created in: {output_base}/")
    print("\nFolder structure:")
    print(f"  {output_base}/")
    for class_name in classes:
        print(f"  ├── {class_name}/")
        print(f"  │   ├── images/        # Visual PNG files")
        print(f"  │   ├── numpy/         # Raw numpy arrays")
        print(f"  │   └── augmented/     # Augmented numpy arrays")
    
    # Create TensorFlow loading script
    create_tf_loader(output_base)

def create_tf_loader(dataset_path):
    """Create a script to load data in TensorFlow"""
    
    loader_code = '''
import numpy as np
import tensorflow as tf
from pathlib import Path

def load_pulmo_dataset(dataset_path, img_shape=(4, 201, 1)):
    """
    Load PULMO AI dataset for TensorFlow
    
    Args:
        dataset_path: Path to ml_dataset folder
        img_shape: Target image shape (paths, frequencies, channels)
    
    Returns:
        train_ds, val_ds: TensorFlow datasets
    """
    
    # Class mapping
    class_names = ['baseline', 'healthy', 'tumor']
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    images = []
    labels = []
    
    # Load original images
    for class_name in class_names:
        class_path = Path(dataset_path) / class_name / 'numpy'
        for npy_file in class_path.glob('*.npy'):
            if '_meta' not in str(npy_file):  # Skip metadata files
                img = np.load(npy_file)
                # Add channel dimension
                img = img.reshape(*img_shape)
                images.append(img)
                labels.append(class_to_idx[class_name])
    
    # Load augmented images
    for class_name in class_names:
        aug_path = Path(dataset_path) / class_name / 'augmented'
        for npy_file in aug_path.glob('*.npy'):
            img = np.load(npy_file)
            img = img.reshape(*img_shape)
            images.append(img)
            labels.append(class_to_idx[class_name])
    
    # Convert to numpy arrays
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    # Shuffle
    indices = np.random.permutation(len(images))
    images = images[indices]
    labels = labels[indices]
    
    # Split (80-20)
    split = int(0.8 * len(images))
    
    # Create TensorFlow datasets
    train_ds = tf.data.Dataset.from_tensor_slices((images[:split], labels[:split]))
    val_ds = tf.data.Dataset.from_tensor_slices((images[split:], labels[split:]))
    
    # Batch and prefetch
    train_ds = train_ds.batch(32).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(32).prefetch(tf.data.AUTOTUNE)
    
    print(f"✅ Loaded {len(images)} images")
    print(f"   Train: {split}, Val: {len(images)-split}")
    print(f"   Classes: {class_names}")
    
    return train_ds, val_ds, class_names

# Example usage:
# train_ds, val_ds, class_names = load_pulmo_dataset('ml_dataset_20260319_123456')
'''
    
    with open(f"{dataset_path}/load_in_tensorflow.py", 'w') as f:
        f.write(loader_code)
    
    print(f"\n📝 Created TensorFlow loader: {dataset_path}/load_in_tensorflow.py")

if __name__ == "__main__":
    main()
