# xgboost_group_kfold.py
"""
XGBoost training with Group K-Fold to prevent data leakage
Each experiment's data stays together during validation
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import pickle
import json

def main():
    print("="*70)
    print("PULMO AI: XGBoost with Group K-Fold")
    print("="*70)
    
    # Load combined dataset
    df = pd.read_csv("pulmo_augmented_dataset/pulmo_augmented.csv")
    
    feature_cols = [c for c in df.columns if c.startswith('freq_')]
    X = df[feature_cols].values
    y = df['class'].values
    
    # Group by experiment (prevent data leakage)
    # You'll need to create a groups column in your dataset
    if 'experiment_id' not in df.columns:
        # Create synthetic groups if not present
        groups = np.arange(len(df)) // 20
    else:
        groups = df['experiment_id'].values
    
    print(f"\n📁 Dataset: {len(df)} samples")
    print(f"   Features: {X.shape[1]}")
    print(f"   Class distribution: {np.bincount(y)}")
    print(f"   Groups: {len(np.unique(groups))}")
    
    # Group K-Fold
    gkf = GroupKFold(n_splits=3)
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}")
        print(f"{'='*50}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_val_scaled)
        acc = accuracy_score(y_val, y_pred)
        fold_scores.append(acc)
        
        print(f"Validation Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred, target_names=['baseline', 'healthy', 'tumor']))
    
    print(f"\n{'='*70}")
    print(f"📊 Cross-Validation Results:")
    print(f"   Fold scores: {fold_scores}")
    print(f"   Mean: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()