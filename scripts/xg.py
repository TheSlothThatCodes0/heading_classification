# Save this as train_model_xgb_fixed.py
import pandas as pd
import datasets
import xgboost as xgb
import numpy as np  # Import numpy to handle infinities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
import joblib

# --- Configuration ---
FINAL_DATA_PATH = "/home/pi0/heading_classification/doclaynet_training/data_final_headings"
MODEL_SAVE_PATH = "/home/pi0/heading_classification/doclaynet_training/xgb_model_ultimate.pkl"
ENCODER_SAVE_PATH = "/home/pi0/heading_classification/doclaynet_training/label_encoder_ultimate.pkl"

def train_model():
    print("🚀 Starting ULTIMATE model training process with XGBoost...")
    labeled_dataset = datasets.load_from_disk(FINAL_DATA_PATH)
    df = labeled_dataset.to_pandas()
    print(f"Successfully loaded {len(df)} headings.")

    # --- Use the ultimate feature set ---
    features = [
        'font_size', 'font_weight', 'norm_y', 'text_length',
        'num_words', 'is_all_caps', 'is_centered', 'has_verb', 'is_date',
        'contains_page_word', 'percent_punct', 'is_in_table', 'numbering_depth',
        'starts_with_bullet', 'ends_with_colon', 'relative_font_size'
    ]
    target = 'heading_level'

    df = df[df[target] != 'Other-Heading'].copy()
    print(f"Filtered to {len(df)} main headings (H1-H6).")

    # --- UPDATED & MORE ROBUST DATA CLEANING ---
    # Convert feature columns to numeric, coercing errors to NaN
    for col in features:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Explicitly replace infinite values with NaN, then fill all NaNs (from any source) with 0
    df[features] = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    print("Cleaned data: Replaced infinities and filled missing values.")

    X = df[features]
    y_str = df[target]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)

    print("Splitting data into training (80%) and validation (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining XGBoost classifier with class balancing...")

    # 1. Calculate sample weights
    sample_weights = compute_sample_weight(
        class_weight='balanced',
        y=y_train
    )

    # 2. Instantiate the XGBoost model
    model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')

    # 3. Fit the model
    model.fit(X_train, y_train, sample_weight=sample_weights)
    print("✅ Model training complete.")

    print("\nEvaluating final model performance...")
    y_pred = model.predict(X_test)
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    labels = sorted(list(set(y_test_labels) | set(y_pred_labels)))
    report = classification_report(y_test_labels, y_pred_labels, labels=labels, digits=3, zero_division=0)
    print("\n--- ULTIMATE Classification Report (XGBoost) ---")
    print(report)

    print(f"\nSaving final model to {MODEL_SAVE_PATH}")
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"Saving final label encoder to {ENCODER_SAVE_PATH}")
    joblib.dump(label_encoder, ENCODER_SAVE_PATH)

    print("\n✅ All done! Ultimate XGBoost model and encoder are saved.")

if __name__ == "__main__":
    train_model()