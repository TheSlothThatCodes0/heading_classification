import pandas as pd
import numpy as np
import datasets
import lightgbm as lgbm
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler  # Oversampling  # type: ignore

# --- Configuration ---
BINARY_DATA_PATH = "/home/pi0/heading_classification/doclaynet_training/data_binary_classification"
MODEL_SAVE_PATH = "/home/pi0/heading_classification/doclaynet_training/models/lgbm_binary_classifier.pkl"
FEATURE_IMPORTANCE_PATH = "/home/pi0/heading_classification/doclaynet_training/models/binary_feature_importance.png"

def train_binary_model():
    print("🚀 Starting BINARY CLASSIFICATION model training...")
    
    # Load the binary classification dataset
    print("Loading binary classification dataset...")
    labeled_dataset = datasets.load_from_disk(BINARY_DATA_PATH)
    df = labeled_dataset.to_pandas()
    print(f"Successfully loaded {len(df)} text elements.")
    
    # Define features for binary classification
    features = [
        'font_size', 'font_weight', 'norm_x', 'norm_y', 'text_length', 
        'num_words', 'is_all_caps', 'is_centered',
        'numbering_depth', 'relative_font_size', 'percent_punct'
    ]
    target = 'is_heading'
    
    # Ensure all feature columns are numeric
    for col in features:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Prepare features and target
    X = df[features]
    y = df[target]
    
    # Print class distribution
    class_counts = y.value_counts()
    print(f"\nClass Distribution:")
    print(f"  Body Text (0): {class_counts[0]:,} ({class_counts[0]/len(y)*100:.1f}%)")
    print(f"  Headings (1): {class_counts[1]:,} ({class_counts[1]/len(y)*100:.1f}%)")
    
    # Split data for training and testing
    print("\nSplitting data into training (80%) and testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Further split training data for validation
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Training set: {len(X_train_final):,} samples")
    print(f"Validation set: {len(X_val):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    
    # Train LightGBM binary classifier with optimized parameters
    print("\nTraining LightGBM binary classifier...")
    
    # 1) Oversample the minority 'heading' class
    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train_final, y_train_final)
    print(f"After oversampling: {np.bincount(y_train_res)} (body,heading)")
    
    # 2) Hyperparameter grid search to maximize F1 on validation set
    best_params, best_f1 = {}, 0
    for nl in [15, 31, 63]:
        for lr in [0.01, 0.05, 0.1]:
            tmp = lgbm.LGBMClassifier(
                objective='binary',
                num_leaves=nl,
                learning_rate=lr,
                random_state=42,
                class_weight='balanced',
                n_estimators=500
            )
            tmp.fit(
                X_train_res, y_train_res,
                eval_set=[(X_val, y_val)],
                callbacks=[lgbm.early_stopping(stopping_rounds=20), lgbm.log_evaluation(0)]
            )
            yv_pred = tmp.predict(X_val)
            f1 = f1_score(y_val, yv_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_params = {'num_leaves': nl, 'learning_rate': lr}
    print(f"→ Best val-F1={best_f1:.4f} with {best_params}")
    
    # Rebuild final model with optimal hyperparameters
    model = lgbm.LGBMClassifier(
        objective='binary',
        num_leaves=best_params['num_leaves'],
        learning_rate=best_params['learning_rate'],
        random_state=42,
        class_weight='balanced',
        n_estimators=1000
    )
    model.fit(
        X_train_res, y_train_res,
        eval_set=[(X_val, y_val)],
        callbacks=[lgbm.early_stopping(stopping_rounds=50), lgbm.log_evaluation(0)]
    )
    print("✅ Model retrained with optimal hyperparameters")
    
    # 3) Find optimal probability threshold on validation set
    y_val_proba = model.predict_proba(X_val)[:, 1]
    best_thresh, best_f1 = 0.5, 0
    for t in np.linspace(0.1, 0.9, 81):
        y_pred_t = (y_val_proba > t).astype(int)
        f1 = f1_score(y_val, y_pred_t)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    print(f"→ Optimal threshold={best_thresh:.2f}, val-F1={best_f1:.4f}")
    
    # Use threshold moving for evaluation
    print("✅ Model training complete.")
    # Store threshold for later evaluation
    threshold = best_thresh
    
    # Evaluate on validation set
    print("\n--- Validation Set Performance @ threshold ---")
    y_val_proba = model.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_proba > threshold).astype(int)
    
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_proba)
    
    print(f"Validation Accuracy @ thresh={threshold:.2f}: {val_accuracy:.4f}")
    print(f"Validation AUC-ROC: {val_auc:.4f}")
    print("\nValidation Classification Report:")
    print(classification_report(y_val, y_val_pred, target_names=['Body Text', 'Heading'], digits=4))
    
    # Evaluate on test set
    print("\n--- Test Set Performance @ threshold ---")
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba > threshold).astype(int)
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"Test Accuracy @ thresh={threshold:.2f}:   {test_accuracy:.4f}")
    print(f"Test AUC-ROC: {test_auc:.4f}")
    print("\nTest Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Body Text', 'Heading'], digits=4))
    
    # Confusion Matrix
    print("\nConfusion Matrix (Test):")
    cm = confusion_matrix(y_test, y_test_pred)
    print("           PredBody  PredHead")
    print(f"ActualBody {cm[0,0]:6,}    {cm[0,1]:6,}")
    print(f"ActualHead {cm[1,0]:6,}    {cm[1,1]:6,}")
    
    # Feature importance analysis
    print("\n--- Feature Importance Analysis ---")
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:20s}: {row['importance']:.4f}")
    
    # Cross-validation for robustness check
    print("\n--- Cross-Validation Analysis ---")
    cv_scores = cross_val_score(
        lgbm.LGBMClassifier(
            objective='binary',
            num_leaves=31,
            learning_rate=0.05,
            random_state=42,
            class_weight='balanced',
            n_estimators=500,
            verbose=-1
        ),
        X_train, y_train, cv=5, scoring='roc_auc'
    )
    
    print(f"5-Fold CV AUC-ROC Scores: {cv_scores}")
    print(f"Mean CV AUC-ROC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Save the model
    print(f"\nSaving binary classifier model to {MODEL_SAVE_PATH}")
    joblib.dump(model, MODEL_SAVE_PATH)
    
    # Save feature importance plot
    try:
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
        plt.title('Top 10 Feature Importance - Binary Heading Classifier')
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        plt.savefig(FEATURE_IMPORTANCE_PATH, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {FEATURE_IMPORTANCE_PATH}")
    except Exception as e:
        print(f"Warning: Could not save feature importance plot: {e}")
    
    # Model summary
    print(f"\n✅✅✅ Binary Classification Model Training Complete!")
    print(f"📊 Final Performance Summary:")
    print(f"   • Test Accuracy: {test_accuracy:.4f}")
    print(f"   • Test AUC-ROC: {test_auc:.4f}")
    print(f"   • Cross-Validation AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"   • Model saved to: {MODEL_SAVE_PATH}")
    print(f"\n🎯 This binary model should be much more reliable at distinguishing")
    print(f"   headings from body text without the complexity of H1-H6 classification!")

if __name__ == "__main__":
    train_binary_model()
