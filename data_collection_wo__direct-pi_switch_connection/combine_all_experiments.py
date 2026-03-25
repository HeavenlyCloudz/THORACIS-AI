# combine_all_experiments_rotations.py
"""
Combine all experiments - keep rotations separate
Each sample = one rotation (all 4 paths concatenated)
Total: 3 experiments × 3 conditions × 3 rotations = 27 samples
"""
import numpy as np
import pandas as pd
from pathlib import Path
import os
import json
from collections import defaultdict

def load_experiment_by_rotation(exp_folder):
    """
    Load data from experiment, keeping rotations separate
    Returns: dict with condition -> rotation -> {path_num: s21_array}
    """
    conditions = ['01_baseline_air', '02_healthy_phantom', '03_tumor_phantom']
    exp_data = {}
    
    for condition in conditions:
        condition_folder = Path(exp_folder) / condition
        if not condition_folder.exists():
            continue
        
        # Group by rotation first
        rotation_data = defaultdict(lambda: defaultdict(list))
        
        for csv_file in condition_folder.glob('*.csv'):
            df = pd.read_csv(csv_file)
            s21 = df['S21_dB'].values
            
            # Extract metadata
            filename = csv_file.stem
            parts = filename.split('_')
            
            path_num = None
            rotation = None
            for part in parts:
                if part.startswith('path'):
                    path_num = int(part[4])
                if part.startswith('rot'):
                    rotation = int(part[3])
            
            if path_num and rotation is not None:
                rotation_data[rotation][path_num].append(s21)
        
        # Average multiple runs for each (rotation, path)
        averaged_rotations = {}
        for rotation, path_data in rotation_data.items():
            averaged_paths = {}
            for path_num, s21_list in path_data.items():
                if s21_list:
                    averaged_paths[path_num] = np.mean(s21_list, axis=0)
            if len(averaged_paths) == 4:  # All 4 paths present
                averaged_rotations[rotation] = averaged_paths
        
        if averaged_rotations:
            exp_data[condition] = averaged_rotations
    
    return exp_data

def main():
    print("="*70)
    print("PULMO AI: Combine Experiments (Keep Rotations Separate)")
    print("="*70)
    
    # Find all experiment folders
    exp_folders = sorted([d for d in os.listdir('.') if d.startswith('phantom_data_')])
    print(f"\n📁 Found {len(exp_folders)} experiment folders:")
    for f in exp_folders:
        print(f"   - {f}")
    
    all_samples = []
    all_labels = []
    all_metadata = []
    
    condition_to_class = {
        '01_baseline_air': 0,
        '02_healthy_phantom': 1,
        '03_tumor_phantom': 2
    }
    
    for exp_idx, exp_folder in enumerate(exp_folders):
        print(f"\n📂 Loading {exp_folder}...")
        
        exp_data = load_experiment_by_rotation(exp_folder)
        
        if not exp_data:
            print(f"   ⚠️ No valid data found")
            continue
        
        for condition, rotations_data in exp_data.items():
            class_label = condition_to_class[condition]
            
            for rotation, path_data in rotations_data.items():
                # Concatenate all 4 paths in order: path1, path2, path3, path4
                feature_vector = np.concatenate([
                    path_data[1],  # Path 1
                    path_data[2],  # Path 2
                    path_data[3],  # Path 3
                    path_data[4]   # Path 4
                ])
                
                all_samples.append(feature_vector)
                all_labels.append(class_label)
                all_metadata.append({
                    'experiment': exp_folder,
                    'condition': condition,
                    'class': class_label,
                    'exp_id': exp_idx,
                    'rotation': rotation
                })
                
                print(f"   {condition} - Rotation {rotation}°: added sample (class {class_label})")
    
    if not all_samples:
        print("❌ No samples loaded!")
        return
    
    X = np.array(all_samples)
    y = np.array(all_labels)
    
    print(f"\n✅ Total samples loaded: {len(X)}")
    print(f"   Features per sample: {X.shape[1]} (4 paths × 201 freq points)")
    print(f"   Class distribution:")
    print(f"      Baseline (0): {(y==0).sum()}")
    print(f"      Healthy (1): {(y==1).sum()}")
    print(f"      Tumor (2): {(y==2).sum()}")
    print(f"   Rotations per condition: {len(set(m['rotation'] for m in all_metadata))}")
    print(f"   Unique experiments: {len(set(m['exp_id'] for m in all_metadata))}")
    
    # Save combined dataset
    output_dir = "pulmo_combined_rotations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV for XGBoost
    df = pd.DataFrame(X)
    df.columns = [f'freq_{i}' for i in range(X.shape[1])]
    df['class'] = y
    df['exp_id'] = [m['exp_id'] for m in all_metadata]
    df['experiment'] = [m['experiment'] for m in all_metadata]
    df['condition'] = [m['condition'] for m in all_metadata]
    df['rotation'] = [m['rotation'] for m in all_metadata]
    df.to_csv(f"{output_dir}/pulmo_combined.csv", index=False)
    
    # Save numpy arrays
    np.save(f"{output_dir}/X_combined.npy", X)
    np.save(f"{output_dir}/y_combined.npy", y)
    
    # Save metadata
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump({
            'num_samples': len(X),
            'num_features': X.shape[1],
            'num_paths': 4,
            'freq_points': 201,
            'num_rotations': len(set(m['rotation'] for m in all_metadata)),
            'class_distribution': {
                'baseline': int((y==0).sum()),
                'healthy': int((y==1).sum()),
                'tumor': int((y==2).sum())
            },
            'experiments_used': len(set(m['exp_id'] for m in all_metadata)),
            'samples_per_experiment': len(all_metadata) // len(set(m['exp_id'] for m in all_metadata)),
            'description': 'Each sample = one rotation (all 4 paths concatenated)'
        }, f, indent=2)
    
    print(f"\n📁 Saved to: {output_dir}/")
    print("   - pulmo_combined.csv")
    print("   - X_combined.npy")
    print("   - y_combined.npy")
    print("   - metadata.json")
    
    print(f"\n📊 Per-experiment breakdown:")
    for exp_id in sorted(set(m['exp_id'] for m in all_metadata)):
        exp_samples = [m for m in all_metadata if m['exp_id'] == exp_id]
        print(f"   Experiment {exp_id}: {len(exp_samples)} samples")
    
    print(f"\n🎉 Now you have {len(X)} samples (keeping rotations separate)!")

if __name__ == "__main__":
    main()
