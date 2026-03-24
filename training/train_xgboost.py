# train_xgboost.py
"""
Train XGBoost classifier on extracted microwave features
"""
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import seaborn as sns
from pathlib import Path

def main():
    print("="*60)
    print("PULMO AI: Train XGBoost on Microwave Features")
    print("="*60)
    
    # Find latest dataset
    dataset_folders = sorted(Path('.').glob('ml_dataset_final_*'))
    if not dataset_folders:
        print("❌ No ml_dataset_final_* folders found!")
        return
    
    latest_dataset = dataset_folders[-1]
    print(f"\n📁 Using dataset: {latest_dataset}")
    
    # Load engineered features (better than raw)
    engineered_path = latest_dataset / 'pulmo_xgboost_engineered.csv'
    
    if not engineered_path.exists():
        print("❌ Engineered features not found! Run create_ml_images_final.py first")
        return
    
    df = pd.read_csv(engineered_path)
    print(f"✅ Loaded {len(df)} samples with {len(df.columns)-3} features")
    
    # Separate features and labels
    feature_cols = [c for c in df.columns if c not in ['class', 'tumor_size_mm', 'sample_id']]
    X = df[feature_cols].values
    y = df['class'].values
    
    print(f"\n📊 Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Train: {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")
    
    # Train XGBoost
    print("\n🏗️ Training XGBoost...")
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Predictions
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)
    
    # Metrics
    val_acc = accuracy_score(y_val, y_pred)
    print(f"\n✅ Validation Accuracy: {val_acc:.4f}")
    
    # Classification report
    class_names = ['baseline', 'healthy', 'tumor']
    print("\n📊 Classification Report:")
    print(classification_report(y_val, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('XGBoost Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{latest_dataset}/xgboost_confusion_matrix.png', dpi=150)
    print(f"✅ Saved: {latest_dataset}/xgboost_confusion_matrix.png")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📈 Top 10 Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance.head(15)['feature'], feature_importance.head(15)['importance'])
    plt.xlabel('Importance')
    plt.title('Top 15 Features - XGBoost')
    plt.tight_layout()
    plt.savefig(f'{latest_dataset}/xgboost_feature_importance.png', dpi=150)
    print(f"✅ Saved: {latest_dataset}/xgboost_feature_importance.png")
    
    # Save model and scaler
    with open(f'{latest_dataset}/xgboost_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open(f'{latest_dataset}/xgboost_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature names
    with open(f'{latest_dataset}/xgboost_features.json', 'w') as f:
        import json
        json.dump(feature_cols, f, indent=2)
    
    print(f"\n✅ Model saved: {latest_dataset}/xgboost_model.pkl")
    print(f"✅ Scaler saved: {latest_dataset}/xgboost_scaler.pkl")
    
    print("\n🚀 XGBoost training complete!")

if __name__ == "__main__":
    main()
