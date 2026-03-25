# xgboost_stratified_kfold_optimized.py
"""
XGBoost training with Stratified K-Fold
- Proper separation of synthetic data (never used in validation)
- Group-based cross-validation to prevent data leakage
- Class weights for imbalanced performance
- Feature importance analysis
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def add_time_domain_features(X, freqs=None):
    """
    Add time-domain features via IFFT
    Helps XGBoost learn temporal patterns
    """
    n_samples, n_features = X.shape
    n_freq = n_features  # 804 features = 4 paths × 201 freq points
    n_paths = 4
    freq_per_path = n_freq // n_paths
    
    time_features = []
    
    for sample in X:
        sample_time_features = []
        for path in range(n_paths):
            # Extract this path's frequency response
            start_idx = path * freq_per_path
            end_idx = (path + 1) * freq_per_path
            freq_response = sample[start_idx:end_idx]
            
            # IFFT to get time domain
            time_response = np.fft.ifft(freq_response)
            time_magnitude = np.abs(time_response)
            
            # Extract time-domain features
            sample_time_features.extend([
                np.max(time_magnitude),          # Peak in time domain
                np.argmax(time_magnitude),       # Location of peak
                np.mean(time_magnitude),         # Average energy
                np.std(time_magnitude),          # Variation
                np.percentile(time_magnitude, 90),  # 90th percentile
                np.percentile(time_magnitude, 10),  # 10th percentile
                np.sum(time_magnitude),          # Total energy
                np.max(time_magnitude) - np.min(time_magnitude),  # Range
                np.sum(np.square(time_magnitude)),  # Energy
            ])
        
        time_features.append(sample_time_features)
    
    time_features = np.array(time_features)
    # Concatenate with original features
    X_augmented = np.concatenate([X, time_features], axis=1)
    
    print(f"   Added {time_features.shape[1]} time-domain features")
    return X_augmented

def plot_feature_importance(importance_list, feature_names, top_k=20):
    """
    Plot feature importance with proper labels
    """
    avg_importance = np.mean(importance_list, axis=0)
    std_importance = np.std(importance_list, axis=0)
    
    # Get top k features
    top_idx = np.argsort(avg_importance)[-top_k:]
    top_importance = avg_importance[top_idx]
    top_std = std_importance[top_idx]
    
    # Create labels
    labels = []
    for idx in top_idx:
        if idx < len(feature_names):
            labels.append(feature_names[idx])
        else:
            labels.append(f'Time Feature {idx - len(feature_names)}')
    
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(top_idx))
    
    plt.barh(y_pos, top_importance, xerr=top_std, capsize=3, alpha=0.8, 
             color=plt.cm.viridis(top_importance / max(top_importance)))
    plt.yticks(y_pos, labels)
    plt.xlabel('Average Feature Importance')
    plt.title(f'Top {top_k} Most Important Features\n(Averaged Across {len(importance_list)} Folds)')
    plt.tight_layout()
    plt.savefig('xgboost_feature_importance_final.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return top_idx, top_importance

def main():
    print("="*70)
    print("PULMO AI: XGBoost Training with Proper Validation")
    print("="*70)
    
    # Load enhanced dataset
    try:
        df = pd.read_csv("pulmo_augmented_rotations_enhanced/pulmo_augmented_enhanced.csv")
        print(f"\n📁 Loaded enhanced dataset: {len(df)} samples")
    except:
        try:
            df = pd.read_csv("pulmo_augmented_rotations/pulmo_augmented.csv")
            print(f"\n📁 Loaded augmented dataset: {len(df)} samples")
        except:
            try:
                df = pd.read_csv("pulmo_combined_rotations/pulmo_combined.csv")
                print(f"\n📁 Loaded original rotation dataset: {len(df)} samples")
            except:
                print("❌ No dataset found. Please run augmentation first.")
                return
    
    # Get features
    feature_cols = [c for c in df.columns if c.startswith('freq_')]
    X = df[feature_cols].values
    y = df['class'].values
    
    # Check for synthetic flag
    has_synthetic = 'synthetic' in df.columns
    if has_synthetic:
        synthetic_mask = df['synthetic'].fillna(False).values
        real_mask = ~synthetic_mask
        print(f"\n📊 Data composition:")
        print(f"   Real samples: {real_mask.sum()}")
        print(f"   Synthetic samples: {synthetic_mask.sum()}")
    else:
        real_mask = np.ones(len(df), dtype=bool)
        synthetic_mask = np.zeros(len(df), dtype=bool)
        print(f"\n📊 No synthetic flag found - using all data")
    
    # Get group labels
    if 'exp_id' in df.columns:
        groups = df['exp_id'].values
        n_unique_groups = len(np.unique(groups[real_mask])) if has_synthetic else df['exp_id'].nunique()
        print(f"\n📊 Using experiment IDs: {n_unique_groups} unique experiments")
        print(f"   Experiment IDs present: {np.unique(groups)}")
    else:
        groups = None
        n_unique_groups = 0
        print("\n⚠️  No experiment IDs found")
    
    print(f"\n📊 Dataset info:")
    print(f"   Total samples: {len(df)}")
    print(f"   Original features: {X.shape[1]}")
    print(f"   Class distribution: baseline={(y==0).sum()}, healthy={(y==1).sum()}, tumor={(y==2).sum()}")
    
    # Add time-domain features
    print("\n🔧 Adding enhanced time-domain features...")
    X_augmented = add_time_domain_features(X)
    print(f"   New feature count: {X_augmented.shape[1]}")
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    sample_weights = np.array([class_weights[int(label)] for label in y])
    print(f"\n📊 Class weights: baseline={class_weights[0]:.2f}, healthy={class_weights[1]:.2f}, tumor={class_weights[2]:.2f}")
    
    # Hyperparameters
    params = {
        'n_estimators': 200,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.5,
        'reg_lambda': 2.0,
        'min_child_weight': 2,
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'verbosity': 0
    }
    
    # Setup cross-validation
    if has_synthetic:
        # Separate real and synthetic data
        X_real = X_augmented[real_mask]
        y_real = y[real_mask]
        sample_weights_real = sample_weights[real_mask]
        groups_real = groups[real_mask] if groups is not None else None
        
        X_synthetic = X_augmented[synthetic_mask]
        y_synthetic = y[synthetic_mask]
        sample_weights_synthetic = sample_weights[synthetic_mask]
        
        print(f"\n✅ Synthetic data will only be used for training (never validation)")
        
        # Determine CV strategy for real data
        if groups_real is not None and len(np.unique(groups_real)) >= 5:
            cv = GroupKFold(n_splits=5)
            print(f"\n✅ Using Group K-Fold with 5 splits on real data")
        elif groups_real is not None:
            cv = LeaveOneGroupOut()
            n_folds = len(np.unique(groups_real))
            print(f"\n✅ Using Leave-One-Group-Out with {n_folds} folds on real data")
        else:
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            print(f"\n✅ Using Stratified K-Fold with 5 splits")
    else:
        # No synthetic data, use regular CV
        X_real = X_augmented
        y_real = y
        sample_weights_real = sample_weights
        groups_real = groups
        X_synthetic = None
        y_synthetic = None
        sample_weights_synthetic = None
        
        if groups_real is not None and len(np.unique(groups_real)) >= 5:
            cv = GroupKFold(n_splits=5)
            print(f"\n✅ Using Group K-Fold with 5 splits")
        elif groups_real is not None:
            cv = LeaveOneGroupOut()
            print(f"\n✅ Using Leave-One-Group-Out with {len(np.unique(groups_real))} folds")
        else:
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            print(f"\n✅ Using Stratified K-Fold with 5 splits")
    
    # Store results
    fold_scores = []
    fold_f1_scores = []
    fold_auc_scores = []
    all_y_val = []
    all_y_pred = []
    all_y_prob = []
    feature_importance_list = []
    models_list = []
    scalers_list = []
    
    # Perform cross-validation
    for fold, (train_real_idx, val_idx) in enumerate(cv.split(X_real, y_real, groups=groups_real)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}")
        print(f"{'='*50}")
        
        # Show validation experiments if using groups
        if groups_real is not None:
            val_experiments = np.unique(groups_real[val_idx])
            print(f"Validation experiments: {val_experiments}")
        
        # Build training set: real training data + all synthetic data
        if X_synthetic is not None:
            X_train = np.vstack([X_real[train_real_idx], X_synthetic])
            y_train = np.concatenate([y_real[train_real_idx], y_synthetic])
            sample_weights_train = np.concatenate([sample_weights_real[train_real_idx], sample_weights_synthetic])
        else:
            X_train = X_real[train_real_idx]
            y_train = y_real[train_real_idx]
            sample_weights_train = sample_weights_real[train_real_idx]
        
        # Validation set: only real data
        X_val = X_real[val_idx]
        y_val = y_real[val_idx]
        sample_weights_val = sample_weights_real[val_idx]
        
        print(f"Train: baseline={(y_train==0).sum()}, healthy={(y_train==1).sum()}, tumor={(y_train==2).sum()}")
        print(f"Val: baseline={(y_val==0).sum()}, healthy={(y_val==1).sum()}, tumor={(y_val==2).sum()}")
        
        if X_synthetic is not None:
            print(f"   (Train includes {len(X_synthetic)} synthetic samples)")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(X_train_scaled, y_train, sample_weight=sample_weights_train)
        
        # Evaluate
        y_pred = model.predict(X_val_scaled)
        y_prob = model.predict_proba(X_val_scaled)
        
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        # Calculate AUC (one-vs-rest)
        try:
            auc = roc_auc_score(y_val, y_prob, multi_class='ovr', average='weighted')
        except:
            auc = 0
        
        fold_scores.append(acc)
        fold_f1_scores.append(f1)
        fold_auc_scores.append(auc)
        
        print(f"\nValidation Accuracy: {acc:.4f}")
        print(f"Validation Weighted F1: {f1:.4f}")
        print(f"Validation AUC: {auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred, target_names=['baseline', 'healthy', 'tumor']))
        
        all_y_val.extend(y_val)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)
        feature_importance_list.append(model.feature_importances_)
        models_list.append(model)
        scalers_list.append(scaler)
    
    # Overall results
    print(f"\n{'='*70}")
    print(f"📊 Cross-Validation Results:")
    print(f"   Fold accuracies: {[f'{s:.4f}' for s in fold_scores]}")
    print(f"   Mean Accuracy: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"   Mean Weighted F1: {np.mean(fold_f1_scores):.4f} ± {np.std(fold_f1_scores):.4f}")
    print(f"   Mean AUC: {np.mean(fold_auc_scores):.4f} ± {np.std(fold_auc_scores):.4f}")
    print(f"   Accuracy Range: {np.min(fold_scores):.4f} - {np.max(fold_scores):.4f}")
    print(f"{'='*70}")
    
    # Confusion matrix with percentages
    cm = confusion_matrix(all_y_val, all_y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['baseline', 'healthy', 'tumor'],
                yticklabels=['baseline', 'healthy', 'tumor'])
    plt.title('XGBoost Confusion Matrix (Final Model)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add percentages
    for i in range(3):
        for j in range(3):
            plt.text(j+0.5, i+0.5, f'\n({cm_percent[i,j]:.1%})', 
                    ha='center', va='center', color='red', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('xgboost_confusion_matrix_final.png', dpi=150)
    plt.show()
    
    # Feature importance with proper labels
    feature_names = feature_cols  # Original frequency features
    top_idx, top_importance = plot_feature_importance(feature_importance_list, feature_names, top_k=20)
    
    # Print top features
    print(f"\n📊 Top 10 Most Important Features:")
    for i in range(min(10, len(top_idx))):
        idx = top_idx[-(i+1)]
        if idx < len(feature_names):
            print(f"   {i+1}. {feature_names[idx]}: {np.mean(feature_importance_list, axis=0)[idx]:.4f}")
        else:
            print(f"   {i+1}. Time Feature {idx - len(feature_names)}: {np.mean(feature_importance_list, axis=0)[idx]:.4f}")
    
    # Train final model on all data
    print("\n💾 Training final model on all data...")
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X_augmented)
    
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X_scaled, y, sample_weight=sample_weights)
    
    # Save model and scaler
    with open('xgboost_final_model.pkl', 'wb') as f:
        pickle.dump(final_model, f)
    with open('xgboost_final_scaler.pkl', 'wb') as f:
        pickle.dump(scaler_final, f)
    
    # Save cross-validation models (ensemble)
    with open('xgboost_cv_models.pkl', 'wb') as f:
        pickle.dump(models_list, f)
    with open('xgboost_cv_scalers.pkl', 'wb') as f:
        pickle.dump(scalers_list, f)
    
    # Save results
    results = {
        'cv_mean_accuracy': float(np.mean(fold_scores)),
        'cv_std_accuracy': float(np.std(fold_scores)),
        'cv_mean_f1': float(np.mean(fold_f1_scores)),
        'cv_std_f1': float(np.std(fold_f1_scores)),
        'cv_mean_auc': float(np.mean(fold_auc_scores)),
        'cv_std_auc': float(np.std(fold_auc_scores)),
        'fold_scores': fold_scores,
        'fold_f1_scores': fold_f1_scores,
        'fold_auc_scores': fold_auc_scores,
        'num_samples': len(df),
        'num_real_samples': int(real_mask.sum()) if has_synthetic else len(df),
        'num_synthetic_samples': int(synthetic_mask.sum()) if has_synthetic else 0,
        'original_features': X.shape[1],
        'final_features': X_augmented.shape[1],
        'num_folds': len(fold_scores),
        'cv_type': type(cv).__name__,
        'class_names': ['baseline', 'healthy', 'tumor'],
        'class_distribution': {
            'baseline': int((y==0).sum()),
            'healthy': int((y==1).sum()),
            'tumor': int((y==2).sum())
        },
        'class_weights': class_weights.tolist(),
        'parameters': params,
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_percentages': cm_percent.tolist(),
        'top_features': {
            'indices': top_idx.tolist(),
            'importance': [float(np.mean(feature_importance_list, axis=0)[i]) for i in top_idx]
        }
    }
    
    with open('xgboost_results_final.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Saved Files:")
    print("   Models:")
    print("   - xgboost_final_model.pkl (best single model)")
    print("   - xgboost_cv_models.pkl (ensemble of CV models)")
    print("   - xgboost_final_scaler.pkl")
    print("   - xgboost_cv_scalers.pkl")
    print("\n   Results:")
    print("   - xgboost_results_final.json")
    print("   - xgboost_confusion_matrix_final.png")
    print("   - xgboost_feature_importance_final.png")
    
    print(f"\n{'='*70}")
    print(f"🎉 FINAL MODEL READY!")
    print(f"{'='*70}")
    print(f"   Cross-Validation Accuracy: {np.mean(fold_scores):.2%} ± {np.std(fold_scores):.2%}")
    print(f"   Cross-Validation F1 Score: {np.mean(fold_f1_scores):.2%} ± {np.std(fold_f1_scores):.2%}")
    print(f"   Cross-Validation AUC: {np.mean(fold_auc_scores):.2%} ± {np.std(fold_auc_scores):.2%}")
    
    if has_synthetic:
        print(f"\n   ✅ Synthetic Data Strategy:")
        print(f"      - {synthetic_mask.sum()} synthetic samples used only for training")
        print(f"      - Validation only on {real_mask.sum()} real samples")
        print(f"      - Prevents overestimation of model performance")
    
    if groups is not None:
        print(f"\n   ✅ Group-Based Validation:")
        print(f"      - {n_unique_groups} unique experiments")
        print(f"      - No data leakage between experiments")
    
    print(f"\n   📊 Class Performance:")
    for i, class_name in enumerate(['baseline', 'healthy', 'tumor']):
        tp = cm[i, i]
        total = cm[i].sum()
        print(f"      {class_name}: {tp}/{total} ({tp/total:.1%})")

if __name__ == "__main__":
    main()
