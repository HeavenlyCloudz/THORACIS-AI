# export_for_octave.py
"""
Exports my data in formats compatible with GNU Octave/MATLAB
"""
import numpy as np
from pathlib import Path
import scipy.io as sio

def export_to_octave(dataset_path):
    """Export numpy arrays to .mat files for GNU Octave"""
    
    output_path = Path(dataset_path) / 'octave_format'
    output_path.mkdir(exist_ok=True)
    
    class_names = ['baseline', 'healthy', 'tumor']
    
    for class_name in class_names:
        # Load original numpy files
        numpy_path = Path(dataset_path) / class_name / 'numpy'
        images = []
        
        for npy_file in numpy_path.glob('*.npy'):
            if '_meta' not in str(npy_file):
                img = np.load(npy_file)
                images.append(img)
        
        if images:
            images = np.array(images)
            
            # Save as .mat file for Octave
            mat_file = output_path / f'{class_name}_data.mat'
            sio.savemat(mat_file, {'images': images, 'class': class_name})
            
            print(f"✅ Saved {mat_file}: {images.shape}")
    
    # Create combined dataset
    all_images = []
    all_labels = []
    
    for i, class_name in enumerate(class_names):
        numpy_path = Path(dataset_path) / class_name / 'numpy'
        for npy_file in numpy_path.glob('*.npy'):
            if '_meta' not in str(npy_file):
                img = np.load(npy_file)
                all_images.append(img)
                all_labels.append(i)
    
    all_images = np.array(all_images)
    all_labels = np.array(all_labels)
    
    mat_file = output_path / 'pulmo_dataset.mat'
    sio.savemat(mat_file, {
        'images': all_images,
        'labels': all_labels,
        'class_names': class_names
    })
    
    print(f"\n✅ Saved complete dataset: {mat_file}")
    print(f"   Images: {all_images.shape}")
    print(f"   Labels: {all_labels.shape}")
    
    # Create README for Octave
    readme = f"""
PULMO AI Dataset for GNU Octave/MATLAB
=======================================

Files:
- baseline_data.mat: Baseline (air) images
- healthy_data.mat: Healthy phantom images  
- tumor_data.mat: Tumor phantom images
- pulmo_dataset.mat: Complete dataset with labels

Data format:
Each image is a 4x201 matrix where:
- Rows 1-4: Antenna paths 1-4
- Columns 1-201: Frequency points from 2-3 GHz

Values are normalized S21 in dB scale (0 to 1).

To load in Octave:
    load pulmo_dataset.mat
    % images: 4x201xN array
    % labels: Nx1 array (0=baseline, 1=healthy, 2=tumor)
    
Example visualization:
    imagesc(squeeze(images(:,:,1)))
    colorbar
    xlabel('Frequency point')
    ylabel('Path number')
"""
    
    with open(output_path / 'README.txt', 'w') as f:
        f.write(readme)
    
    print(f"\n📝 Created README for Octave")

if __name__ == "__main__":
    dataset_folders = sorted(Path('.').glob('ml_dataset_*'))
    if dataset_folders:
        latest = dataset_folders[-1]
        print(f"📁 Exporting: {latest}")
        export_to_octave(latest)
    else:
        print("❌ No dataset found")
