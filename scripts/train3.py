import pandas as pd
import datasets
import lightgbm as lgbm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
import joblib
import optuna

# --- Configuration ---
FINAL_DATA_PATH = "/home/pi0/heading_classification/doclaynet_training/data_final_headings"
MODEL_SAVE_PATH = "/home/pi0/heading_classification/doclaynet_training/lgbm_model_tuned.pkl"
ENCODER_SAVE_PATH = "/home/pi0/heading_classification/doclaynet_training/label_encoder_tuned.pkl"
N_TRIALS = 50 

def tune_model():
    print("🚀 Starting hyperparameter tuning process...")

    # 1. Load and prepare the dataset
    print(f"Loading final dataset from {FINAL_DATA_PATH}...")
    labeled_dataset = datasets.load_from_disk(FINAL_DATA_PATH)
    df = labeled_dataset.to_pandas()

    print("Cleaning and preparing features...")
    features = ['font_size', 'font_weight', 'norm_x', 'norm_y', 'text_length', 'num_words', 'is_all_caps', 'is_centered']
    target = 'heading_level'
    df = df[df[target] != 'Other-Heading']

    X = df[features]
    y_str = df[target]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Define the Objective Function for Optuna
    def objective(trial):
        params = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'n_jobs': -1,
            'random_state': 42,
            'class_weight': 'balanced'
        }
        
        # --- MODIFICATION: Added verbose=-1 to silence warnings ---
        model = lgbm.LGBMClassifier(**params, verbose=-1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        score = f1_score(y_test, y_pred, average='weighted')
        return score

    # 3. Run the optimization
    print(f"\nRunning Optuna optimization for {N_TRIALS} trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n✅ Optimization complete!")
    print(f"Best trial's F1-score: {study.best_value:.4f}")
    print("Best hyperparameters found:")
    print(study.best_params)

    # 4. Train the final model with the best parameters
    print("\nTraining final model with the best hyperparameters...")
    best_params = study.best_params
    best_params['n_jobs'] = -1
    best_params['random_state'] = 42
    best_params['class_weight'] = 'balanced'

    final_model = lgbm.LGBMClassifier(**best_params, verbose=-1)
    final_model.fit(X_train, y_train)

    # 5. Show final report and save the model
    print("\nEvaluating final tuned model...")
    y_pred = final_model.predict(X_test)
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    
    labels = sorted(list(set(y_test_labels) | set(y_pred_labels)))
    report = classification_report(y_test_labels, y_pred_labels, labels=labels, digits=3, zero_division=0)
    print("\n--- Final Tuned Model Classification Report ---")
    print(report)
    print("-------------------------------------------\n")

    print(f"Saving tuned model to {MODEL_SAVE_PATH}")
    joblib.dump(final_model, MODEL_SAVE_PATH)
    print(f"Saving label encoder to {ENCODER_SAVE_PATH}")
    joblib.dump(label_encoder, ENCODER_SAVE_PATH)

    print("\n✅ All done! Tuned model and encoder are saved.")

if __name__ == "__main__":
    tune_model()