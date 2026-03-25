# combine_all_experiments.py
"""
Combine all phantom experiments into one large dataset
Uses ALL 4 paths after background subtraction
"""
import numpy as np
import pandas as pd
from pathlib import Path
import os
import json

def load_experiment_data(exp_folder, valid_paths=[1, 2, 3, 4]):
    """
    Load all condition data from an experiment folder
    Uses ALL paths (after background subtraction, they all work)
    """
    conditions = ['01_baseline_air', '02_healthy_phantom', '03_tumor_phantom']
    
    exp_data = {}
    
    for condition in conditions:
        condition_folder = Path(exp_folder) / condition
        if not condition_folder.exists():
            continue
        
        # Load all CSV files for this condition
        data_by_path = {p: [] for p in valid_paths}
        
        for csv_file in condition_folder.glob('*.csv'):
            df = pd.read_csv(csv_file)
            s21 = df['S21_dB'].values
            
            # Extract path number from filename
            filename = csv_file.stem
            parts = filename.split('_')
            path_num = None
            for part in parts:
                if part.startswith('path'):
                    path_num = int(part[4])
                    break
            
            if path_num in valid_paths:
                data_by_path[path_num].append(s21)
        
        # Average multiple runs per path
        path_averages = {}
        for path_num in valid_paths:
            if data_by_path[path_num]:
                path_averages[path_num] = np.mean(data_by_path[path_num], axis=0)
        
        exp_data[condition] = path_averages
    
    return exp_data

def load_all_experiments():
    """
    Load data from all phantom_data_* folders
    """
    all_samples = []
    all_labels = []
    
    # Find all experiment folders
    exp_folders = sorted([d for d in os.listdir('.') if d.startswith('phantom_data_')])
    
    print(f"📁 Found {len(exp_folders)} experiment folders:")
    for f in exp_folders:
        print(f"   - {f}")
    
    for exp_idx, exp_folder in enumerate(exp_folders):
        print(f"\n📂 Loading {exp_folder}...")
        
        exp_data = load_experiment_data(exp_folder)
        
        # Process each condition
        for condition_name, path_data in exp_data.items():
            if condition_name == '01_baseline_air':
                class_label = 0
                class_name = 'baseline'
            elif condition_name == '02_healthy_phantom':
                class_label = 1
                class_name = 'healthy'
            elif condition_name == '03_tumor_phantom':
                class_label = 2
                class_name = 'tumor'
            else:
                continue
            
            # Create one sample per path
            for path_num, s21 in path_data.items():
                # Convert to 402-dim vector (4 paths × 201 freq points)
                # But we need all 4 paths together for each sample
                # Actually, let's store each path separately for now
                all_samples.append({
                    'exp_id': exp_idx,
                    'condition': class_name,
                    'class': class_label,
                    'path': path_num,
                    's21': s21,
                    'experiment': exp_folder
                })
                
                all_labels.append(class_label)
        
        print(f"   Added {len(path_data) * 3} samples")
    
    return all_samples

def create_feature_matrix(all_samples):
    """
    Convert samples to feature matrix
    Each sample = concatenated S21 from all 4 paths = 804 features
    """
    # Group by condition + experiment (each "sample" should have all 4 paths)
    grouped = {}
    
    for sample in all_samples:
        key = (sample['exp_id'], sample['condition'])
        if key not in grouped:
            grouped[key] = {1: None, 2: None, 3: None, 4: None}
        grouped[key][sample['path']] = sample['s21']
    
    X = []
    y = []
    metadata = []
    
    for (exp_id, condition), path_data in grouped.items():
        # Check if we have all 4 paths
        if all(path_data[p] is not None for p in [1, 2, 3, 4]):
            # Concatenate all paths in order: path1, path2, path3, path4
            feature_vector = np.concatenate([
                path_data[1], path_data[2], path_data[3], path_data[4]
            ])
            X.append(feature_vector)
            
            # Get class label from first sample (all in this group have same class)
            label = next(s for s in all_samples if s['exp_id'] == exp_id and s['condition'] == condition)['class']
            y.append(label)
            
            metadata.append({
                'exp_id': exp_id,
                'condition': condition,
                'has_path1': True,
                'has_path2': True,
                'has_path3': True,
                'has_path4': True
            })
    
    return np.array(X), np.array(y), metadata

def main():
    print("="*70)
    print("PULMO AI: Combine All Experiments")
    print("="*70)
    
    # Load all experiments
    all_samples = load_all_experiments()
    print(f"\n✅ Total samples loaded: {len(all_samples)}")
    
    # Create feature matrix (804 features = 4 paths × 201 freq points)
    X, y, metadata = create_feature_matrix(all_samples)
    
    print(f"\n📊 Combined dataset:")
    print(f"   Total samples: {len(X)}")
    print(f"   Features per sample: {X.shape[1]}")
    print(f"   Class distribution:")
    print(f"      Baseline: {(y==0).sum()}")
    print(f"      Healthy: {(y==1).sum()}")
    print(f"      Tumor: {(y==2).sum()}")
    
    # Save combined dataset
    output_dir = "pulmo_combined_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV for XGBoost
    df = pd.DataFrame(X)
    df.columns = [f'freq_{i}' for i in range(X.shape[1])]
    df['class'] = y
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
            'class_distribution': {
                'baseline': int((y==0).sum()),
                'healthy': int((y==1).sum()),
                'tumor': int((y==2).sum())
            },
            'experiments_used': len(set(m['exp_id'] for m in metadata))
        }, f, indent=2)
    
    print(f"\n📁 Saved to: {output_dir}/")
    print("   - pulmo_combined.csv")
    print("   - X_combined.npy")
    print("   - y_combined.npy")
    print("   - metadata.json")
    
    # Quick stats
    print(f"\n📊 Statistics per class:")
    for class_name, class_label in [('baseline', 0), ('healthy', 1), ('tumor', 2)]:
        class_X = X[y == class_label]
        if len(class_X) > 0:
            print(f"   {class_name}: {len(class_X)} samples")
    
    print("\n🎉 Now you have a much larger dataset for training!")

if __name__ == "__main__":
    main()