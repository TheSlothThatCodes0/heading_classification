# Save this as train_model.py

import pandas as pd
import datasets
import lightgbm as lgbm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

# --- Configuration ---
FINAL_DATA_PATH = "/home/pi0/heading_classification/doclaynet_training/data_final_headings"
MODEL_SAVE_PATH = "/home/pi0/heading_classification/doclaynet_training/lgbm_model_v2.pkl"
ENCODER_SAVE_PATH = "/home/pi0/heading_classification/doclaynet_training/label_encoder_v2.pkl"

def train_model():
    print("🚀 Starting model training process with ADVANCED FEATURES...")
    labeled_dataset = datasets.load_from_disk(FINAL_DATA_PATH)
    df = labeled_dataset.to_pandas()
    print(f"Successfully loaded {len(df)} headings with advanced features.")

    # --- UPDATED: Use all the new features ---
    features = [
        'font_size', 
        'font_weight', 
        'norm_x', 
        'norm_y', 
        'text_length', 
        'num_words',
        'is_all_caps',
        'is_centered'
    ]
    target = 'heading_level'

    # Filter out 'Other-Heading' as we only want to classify the main levels
    df = df[df[target] != 'Other-Heading']
    print(f"Filtered to {len(df)} main headings (H1-H6).")

    X = df[features]
    y_str = df[target]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)

    print("Splitting data into training (80%) and validation (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # --- UPDATED: Add class_weight='balanced' ---
    print("\nTraining LightGBM classifier with class balancing...")
    model = lgbm.LGBMClassifier(random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    print("✅ Model training complete.")

    print("\nEvaluating model performance on the validation set...")
    y_pred = model.predict(X_test)
    
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    
    # Get all unique labels for the report
    labels = sorted(list(set(y_test_labels) | set(y_pred_labels)))
    report = classification_report(y_test_labels, y_pred_labels, labels=labels, digits=3, zero_division=0)
    print("\n--- Classification Report ---")
    print(report)
    print("---------------------------\n")

    print(f"Saving model to {MODEL_SAVE_PATH}")
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"Saving label encoder to {ENCODER_SAVE_PATH}")
    joblib.dump(label_encoder, ENCODER_SAVE_PATH)
    
    print("\n✅ All done! New model and encoder are saved.")

if __name__ == "__main__":
    train_model()