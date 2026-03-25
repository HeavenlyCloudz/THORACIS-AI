# train_xgboost.py
"""
PULMO AI: Train XGBoost on Synthetic Dataset
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import pickle
import json

def main():
    print("="*60)
    print("PULMO AI: XGBoost Training (Fixed for Small Dataset)")
    print("="*60)
    
    # ==================== LOAD SYNTHETIC DATA ====================
    df = pd.read_csv("ml_dataset_synthetic/pulmo_xgboost_synthetic.csv")
    
    feature_cols = [c for c in df.columns if c.startswith('freq_')]
    X = df[feature_cols].values
    y = df['class'].values
    
    print(f"\n📁 Loaded {len(X)} samples")
    print(f"   Class distribution: {np.bincount(y)}")
    
    # ==================== SPLIT WITH STRATIFICATION ====================
    # Now we have enough samples to split properly
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Train: {len(X_train)} samples")
    print(f"📊 Validation: {len(X_val)} samples")
    print(f"   Train class dist: {np.bincount(y_train)}")
    print(f"   Val class dist: {np.bincount(y_val)}")
    
    # ==================== SCALE ====================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # ==================== TRAIN ====================
    print("\n🎯 Training XGBoost...")
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    model.fit(X_train_scaled, y_train)
    
    # ==================== EVALUATE ====================
    y_pred = model.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, y_pred)
    
    print(f"\n✅ Validation Accuracy: {val_acc:.4f}")
    print("\n📊 Classification Report:")
    print(classification_report(y_val, y_pred, target_names=['baseline', 'healthy', 'tumor']))
    
    # ==================== CONFUSION MATRIX ====================
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['baseline', 'healthy', 'tumor'],
                yticklabels=['baseline', 'healthy', 'tumor'])
    plt.title('XGBoost Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('xgboost_confusion_matrix.png')
    plt.show()
    
    # ==================== SAVE ====================
    with open('xgboost_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('xgboost_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("\n✅ Saved: xgboost_model.pkl, xgboost_scaler.pkl")

if __name__ == "__main__":
    main()
