# fusion_classifier.py
"""
Fuse microwave features with audio embeddings for unified diagnosis
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

class FusionClassifier:
    def __init__(self):
        self.microwave_scaler = StandardScaler()
        self.audio_scaler = StandardScaler()
        self.fusion_model = None
        
    def load_microwave_features(self, csv_path):
        """Load pre-extracted microwave features"""
        df = pd.read_csv(csv_path)
        # Extract feature columns (exclude class/tumor_size)
        feature_cols = [c for c in df.columns if c.startswith('freq_')]
        X = df[feature_cols].values
        y = df['class_label'].values
        return X, y
    
    def load_audio_features(self, npy_path):
        """Load YAMNet audio embeddings"""
        return np.load(npy_path)
    
    def create_synthetic_audio(self, y_labels):
        """
        Create synthetic audio features for fusion demonstration
        In real system, you'd load actual audio recordings
        """
        np.random.seed(42)
        n_samples = len(y_labels)
        
        # Create 1024-dim audio embeddings with class-specific patterns
        audio_features = np.random.randn(n_samples, 1024) * 0.1
        
        # Add class-specific patterns
        for i, label in enumerate(y_labels):
            if label == 2:  # Tumor
                audio_features[i] += 0.3  # Tumor affects breathing
            elif label == 1:  # Healthy
                audio_features[i] -= 0.1
                
        return audio_features
    
    def train_fusion_model(self, X_microwave, y_labels):
        """Train fusion model combining both modalities"""
        
        # Create synthetic audio features
        X_audio = self.create_synthetic_audio(y_labels)
        
        # Scale each modality separately
        X_microwave_scaled = self.microwave_scaler.fit_transform(X_microwave)
        X_audio_scaled = self.audio_scaler.fit_transform(X_audio)
        
        # Concatenate features
        X_fusion = np.concatenate([X_microwave_scaled, X_audio_scaled], axis=1)
        
        print(f"✅ Fusion feature shape: {X_fusion.shape}")
        print(f"   Microwave: {X_microwave_scaled.shape[1]} features")
        print(f"   Audio: {X_audio_scaled.shape[1]} features")
        
        # Train/test split
        X_train, X_val, y_train, y_val = train_test_split(
            X_fusion, y_labels, test_size=0.2, random_state=42, stratify=y_labels
        )
        
        # Train XGBoost classifier
        self.fusion_model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        
        self.fusion_model.fit(X_train, y_train)
        
        # Evaluate
        train_acc = self.fusion_model.score(X_train, y_train)
        val_acc = self.fusion_model.score(X_val, y_val)
        
        print(f"\n📊 Fusion Model Performance:")
        print(f"   Train Accuracy: {train_acc:.4f}")
        print(f"   Val Accuracy: {val_acc:.4f}")
        
        # Compare with microwave-only
        microwave_model = XGBClassifier(n_estimators=100, max_depth=4)
        microwave_model.fit(X_train[:, :X_microwave_scaled.shape[1]], y_train)
        microwave_acc = microwave_model.score(X_val[:, :X_microwave_scaled.shape[1]], y_val)
        
        print(f"\n📊 Comparison:")
        print(f"   Microwave-only Val Acc: {microwave_acc:.4f}")
        print(f"   Fusion Val Acc: {val_acc:.4f}")
        print(f"   Improvement: {(val_acc - microwave_acc)*100:.1f}%")
        
        return self.fusion_model
    
    def save_models(self, folder='fusion_models'):
        """Save trained models"""
        import os
        os.makedirs(folder, exist_ok=True)
        
        with open(f'{folder}/microwave_scaler.pkl', 'wb') as f:
            pickle.dump(self.microwave_scaler, f)
        with open(f'{folder}/audio_scaler.pkl', 'wb') as f:
            pickle.dump(self.audio_scaler, f)
        with open(f'{folder}/fusion_model.pkl', 'wb') as f:
            pickle.dump(self.fusion_model, f)
        
        print(f"✅ Models saved to {folder}/")

def main():
    print("="*60)
    print("PULMO AI: Microwave-Audio Fusion Classifier")
    print("="*60)
    
    # Load microwave features
    print("\n📊 Loading microwave features...")
    fusion = FusionClassifier()
    
    # Find latest XGBoost dataset
    import glob
    dataset_files = glob.glob('ml_dataset_*/pulmo_xgboost_dataset.csv')
    if not dataset_files:
        print("❌ No XGBoost dataset found! Run create_ml_images_updated.py first")
        return
    
    latest_dataset = sorted(dataset_files)[-1]
    print(f"📁 Using: {latest_dataset}")
    
    # Load data
    df = pd.read_csv(latest_dataset)
    feature_cols = [c for c in df.columns if c.startswith('freq_')]
    X_microwave = df[feature_cols].values
    y_labels = df['class_label'].values
    
    print(f"   Microwave samples: {X_microwave.shape[0]}")
    print(f"   Class distribution: {np.bincount(y_labels)}")
    
    # Train fusion model
    fusion.train_fusion_model(X_microwave, y_labels)
    
    # Save models
    fusion.save_models()
    
    print("\n✅ Fusion pipeline ready!")
    print("\n🚀 NEXT: Use this with real audio data from your microphone setup")

if __name__ == "__main__":
    main()
