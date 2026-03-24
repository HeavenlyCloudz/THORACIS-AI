# train_xgboost.py
"""
PULMO AI: Train XGBoost on Microwave Features
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import pickle
import json
from pathlib import Path

def main():
    print("="*70)
    print("PULMO AI: Train XGBoost on Microwave Features")
    print("="*70)
    
    # Find latest dataset
    dataset_folders = sorted(Path('.').glob('ml_dataset_final_*'))
    if not dataset_folders:
        print("❌ No ml_dataset_final_* folders found!")
        print("   Run create_ml_images_final.py first")
        return
    
    latest_folder = dataset_folders[-1]
    print(f"\n📁 Using dataset: {latest_folder}")
    
    # Load data
    data_path = latest_folder / 'pulmo_xgboost_data.csv'
    if not data_path.exists():
        print(f"❌ No XGBoost data found at {data_path}")
        return
    
    df = pd.read_csv(data_path)
    print(f"\n📊 Loaded {len(df)} samples")
    
    # Separate features and labels
    feature_cols = [c for c in df.columns if c.startswith('freq_')]
    X = df[feature_cols].values
    y = df['class'].values
    
    print(f"   Features: {X.shape[1]} (2 paths × 201 frequencies)")
    print(f"   Classes: {np.unique(y)}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/test split (handling small dataset)
    if len(np.unique(y)) >= 2 and min(np.bincount(y)) >= 2:
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"\n✅ Stratified split: {len(X_train)} train, {len(X_val)} val")
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        print(f"\n✅ Simple split: {len(X_train)} train, {len(X_val)} val")
    
    # Train XGBoost
    print("\n🏋️ Training XGBoost...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_val)
    y_pred_train = model.predict(X_train)
    
    print(f"\n📊 Training Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
    print(f"📊 Validation Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    
    # Feature importance
    importance = model.feature_importances_
    top_indices = np.argsort(importance)[-20:]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(20), importance[top_indices])
    plt.yticks(range(20), [f'Freq {i}' for i in top_indices])
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Most Important Frequency Features')
    plt.tight_layout()
    plt.savefig(latest_folder / 'xgboost_feature_importance.png', dpi=150)
    print(f"\n📊 Saved: xgboost_feature_importance.png")
    
    # Save model and scaler
    model_path = latest_folder / 'xgboost_model.pkl'
    scaler_path = latest_folder / 'xgboost_scaler.pkl'
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\n✅ Model saved to: {model_path}")
    print(f"✅ Scaler saved to: {scaler_path}")
    
    # Save metadata
    metadata = {
        'model_type': 'XGBoost',
        'num_features': X.shape[1],
        'num_samples': len(df),
        'class_distribution': {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        'train_accuracy': float(accuracy_score(y_train, y_pred_train)),
        'val_accuracy': float(accuracy_score(y_val, y_pred))
    }
    
    with open(latest_folder / 'xgboost_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n🚀 DONE! XGBoost model trained successfully!")

if __name__ == "__main__":
    main()