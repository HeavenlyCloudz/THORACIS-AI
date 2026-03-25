# augment_dataset.py
"""
Generate synthetic dataset from 3 samples using realistic variations
Run this on Kaggle or locally after you have your data
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

def generate_synthetic_microwave_data(original_features, class_label, n_samples=50):
    """
    Generate synthetic samples from original features
    Adds realistic variations: noise, scaling, frequency shifts
    """
    synthetic = []
    
    for i in range(n_samples):
        # Copy original
        sample = original_features.copy()
        
        # Add Gaussian noise (simulates measurement variation)
        noise = np.random.normal(0, 0.5, sample.shape)
        sample = sample + noise
        
        # Add small frequency shift (simulates positioning variation)
        shift = np.random.randint(-3, 4)
        if shift != 0:
            sample = np.roll(sample, shift)
        
        # Add amplitude scaling (simulates phantom variation)
        scale = np.random.uniform(0.9, 1.1)
        sample = sample * scale
        
        # Ensure values stay in realistic range (-60 to 0 dB)
        sample = np.clip(sample, -60, 0)
        
        synthetic.append(sample)
    
    return synthetic

def generate_synthetic_cnn_images(original_image, class_label, n_samples=50):
    """
    Generate synthetic 2D images (2 paths × 201 freq points)
    """
    synthetic = []
    
    for i in range(n_samples):
        # Copy original
        img = original_image.copy()
        
        # Add Gaussian noise
        noise = np.random.normal(0, 0.02, img.shape)
        img = img + noise
        
        # Small frequency shift (for both paths together)
        shift = np.random.randint(-5, 6)
        if shift != 0:
            img = np.roll(img, shift, axis=1)
        
        # Small path-specific variation
        path_scale = np.random.uniform(0.95, 1.05)
        img[0] = img[0] * path_scale
        img[1] = img[1] * (1/path_scale)  # Opposite effect on other path
        
        # Normalize back to [0,1]
        img = np.clip(img, 0, 1)
        
        synthetic.append(img)
    
    return synthetic

def main():
    print("="*60)
    print("PULMO AI: Synthetic Data Generation")
    print("="*60)
    
    # ==================== LOAD YOUR ORIGINAL DATA ====================
    DATASET_PATH = "ml_dataset_final_20260324_154658"  # Your folder
    
    # Load raw features
    xgb_df = pd.read_csv(f"{DATASET_PATH}/pulmo_xgboost_data.csv")
    feature_cols = [c for c in xgb_df.columns if c.startswith('freq_')]
    
    baseline_raw = xgb_df[xgb_df['class'] == 0][feature_cols].values[0]
    healthy_raw = xgb_df[xgb_df['class'] == 1][feature_cols].values[0]
    tumor_raw = xgb_df[xgb_df['class'] == 2][feature_cols].values[0]
    
    # Load CNN images
    cnn_images = np.load(f"{DATASET_PATH}/02_cnn_images/cnn_images.npy")
    cnn_labels = np.load(f"{DATASET_PATH}/02_cnn_images/cnn_labels.npy")
    
    baseline_img = cnn_images[cnn_labels == 0][0]
    healthy_img = cnn_images[cnn_labels == 1][0]
    tumor_img = cnn_images[cnn_labels == 2][0]
    
    print(f"\n📁 Original data:")
    print(f"   Baseline: 1 raw, 1 image")
    print(f"   Healthy: 1 raw, 1 image")
    print(f"   Tumor: 1 raw, 1 image")
    
    # ==================== GENERATE SYNTHETIC DATA ====================
    print("\n🔧 Generating synthetic data...")
    
    N_SYNTHETIC = 50  # Generate 50 samples per class
    
    baseline_synth_raw = generate_synthetic_microwave_data(baseline_raw, 0, N_SYNTHETIC)
    healthy_synth_raw = generate_synthetic_microwave_data(healthy_raw, 1, N_SYNTHETIC)
    tumor_synth_raw = generate_synthetic_microwave_data(tumor_raw, 2, N_SYNTHETIC)
    
    baseline_synth_img = generate_synthetic_cnn_images(baseline_img, 0, N_SYNTHETIC)
    healthy_synth_img = generate_synthetic_cnn_images(healthy_img, 1, N_SYNTHETIC)
    tumor_synth_img = generate_synthetic_cnn_images(tumor_img, 2, N_SYNTHETIC)
    
    # ==================== COMBINE WITH ORIGINALS ====================
    
    # XGBoost dataset
    X_synth_raw = []
    y_synth = []
    
    for i, synth in enumerate(baseline_synth_raw):
        X_synth_raw.append(synth)
        y_synth.append(0)
    for i, synth in enumerate(healthy_synth_raw):
        X_synth_raw.append(synth)
        y_synth.append(1)
    for i, synth in enumerate(tumor_synth_raw):
        X_synth_raw.append(synth)
        y_synth.append(2)
    
    # Add originals
    X_synth_raw.append(baseline_raw)
    y_synth.append(0)
    X_synth_raw.append(healthy_raw)
    y_synth.append(1)
    X_synth_raw.append(tumor_raw)
    y_synth.append(2)
    
    X_synth_raw = np.array(X_synth_raw)
    y_synth = np.array(y_synth)
    
    # CNN dataset
    X_synth_img = []
    y_synth_img = []
    
    for img in baseline_synth_img:
        X_synth_img.append(img)
        y_synth_img.append(0)
    for img in healthy_synth_img:
        X_synth_img.append(img)
        y_synth_img.append(1)
    for img in tumor_synth_img:
        X_synth_img.append(img)
        y_synth_img.append(2)
    
    X_synth_img.append(baseline_img)
    y_synth_img.append(0)
    X_synth_img.append(healthy_img)
    y_synth_img.append(1)
    X_synth_img.append(tumor_img)
    y_synth_img.append(2)
    
    X_synth_img = np.array(X_synth_img)
    y_synth_img = np.array(y_synth_img)
    
    print(f"\n✅ Synthetic dataset created:")
    print(f"   XGBoost: {X_synth_raw.shape[0]} samples, {X_synth_raw.shape[1]} features")
    print(f"   CNN: {X_synth_img.shape[0]} images, shape {X_synth_img.shape[1:]}")
    print(f"\n   Class distribution:")
    print(f"      Baseline: {(y_synth == 0).sum()}")
    print(f"      Healthy: {(y_synth == 1).sum()}")
    print(f"      Tumor: {(y_synth == 2).sum()}")
    
    # ==================== SAVE SYNTHETIC DATASET ====================
    output_dir = "ml_dataset_synthetic"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save XGBoost data
    df_synth = pd.DataFrame(X_synth_raw)
    df_synth.columns = [f'freq_{i}' for i in range(X_synth_raw.shape[1])]
    df_synth['class'] = y_synth
    df_synth.to_csv(f"{output_dir}/pulmo_xgboost_synthetic.csv", index=False)
    
    # Save CNN data
    np.save(f"{output_dir}/cnn_images_synthetic.npy", X_synth_img)
    np.save(f"{output_dir}/cnn_labels_synthetic.npy", y_synth_img)
    
    # Save metadata
    metadata = {
        'num_samples': len(y_synth),
        'num_features': X_synth_raw.shape[1],
        'class_distribution': {
            'baseline': int((y_synth == 0).sum()),
            'healthy': int((y_synth == 1).sum()),
            'tumor': int((y_synth == 2).sum())
        },
        'description': 'Synthetic dataset generated from 3 original samples',
        'augmentation_methods': ['Gaussian noise', 'Frequency shift', 'Amplitude scaling']
    }
    
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n📁 Saved to: {output_dir}/")
    print("   - pulmo_xgboost_synthetic.csv")
    print("   - cnn_images_synthetic.npy")
    print("   - cnn_labels_synthetic.npy")
    print("   - metadata.json")
    
    # ==================== VISUALIZE SYNTHETIC DATA ====================
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Plot original microwave features
    classes = [baseline_raw, healthy_raw, tumor_raw]
    titles = ['Baseline (Original)', 'Healthy (Original)', 'Tumor (Original)']
    for i, (ax, data, title) in enumerate(zip(axes[0], classes, titles)):
        ax.plot(data, 'b-', alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel('Frequency Point')
        ax.set_ylabel('S21 (dB)')
        ax.grid(True, alpha=0.3)
    
    # Plot one synthetic example per class
    synth_classes = [baseline_synth_raw[0], healthy_synth_raw[0], tumor_synth_raw[0]]
    titles_synth = ['Baseline (Synthetic)', 'Healthy (Synthetic)', 'Tumor (Synthetic)']
    for i, (ax, data, title) in enumerate(zip(axes[1], synth_classes, titles_synth)):
        ax.plot(data, 'r-', alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel('Frequency Point')
        ax.set_ylabel('S21 (dB)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/synthetic_comparison.png", dpi=150)
    plt.show()
    
    print(f"\n✅ Saved visualization: synthetic_comparison.png")
    print("\n🎉 Now you can train with 153 samples instead of 3!")

if __name__ == "__main__":
    import os
    main()