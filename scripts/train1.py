import os
import pandas as pd
import datasets
import lightgbm as lgbm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

# --- Configuration ---
FINAL_DATA_PATH = "/home/pi0/heading_classification/doclaynet_training/data_final_headings"
MODEL_SAVE_PATH = "/home/pi0/heading_classification/doclaynet_training/lgbm_model.pkl"
ENCODER_SAVE_PATH = "/home/pi0/heading_classification/doclaynet_training/label_encoder.pkl"

def train_model():
    """
    Loads the final labeled dataset, cleans it by removing rare classes,
    trains a LightGBM classifier, evaluates it, and saves the final model.
    """
    print("🚀 Starting model training process...")

    # 1. Load the dataset
    print(f"Loading final dataset from {FINAL_DATA_PATH}...")
    try:
        labeled_dataset = datasets.load_from_disk(FINAL_DATA_PATH)
        df = labeled_dataset.to_pandas()
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    print(f"Successfully loaded {len(df)} headings.")

    # --- NEW: Data Cleaning Step ---
    print("\nCleaning data by removing rare heading levels...")
    class_counts = df['heading_level'].value_counts()
    # Identify classes with fewer than 2 members (the minimum for stratification)
    rare_classes = class_counts[class_counts < 2].index.tolist()

    if rare_classes:
        print(f"Found rare classes with only 1 member: {rare_classes}. These will be removed.")
        original_rows = len(df)
        df = df[~df['heading_level'].isin(rare_classes)]
        print(f"Removed {original_rows - len(df)} rows. New dataset size: {len(df)} headings.")
    else:
        print("No rare classes with only 1 member found.")
    # --- END: Data Cleaning Step ---

    # 2. Feature Engineering & Preprocessing
    print("\nPreparing features and labels...")
    
    features = ['font_size', 'font_weight', 'vertical_position']
    target = 'heading_level'

    X = df[features]
    y_str = df[target]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)

    # 3. Split data into training and validation sets
    print("Splitting data into training (80%) and validation (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {len(X_train)} samples")
    print(f"Validation set size: {len(X_test)} samples")

    # 4. Train the LightGBM model
    print("\nTraining LightGBM classifier...")
    model = lgbm.LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("✅ Model training complete.")

    # 5. Evaluate the model
    print("\nEvaluating model performance on the validation set...")
    y_pred = model.predict(X_test)
    
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    
    report = classification_report(y_test_labels, y_pred_labels, digits=3, zero_division=0)
    print("\n--- Classification Report ---")
    print(report)
    print("---------------------------\n")

    # 6. Save the trained model and the label encoder
    print(f"Saving model to {MODEL_SAVE_PATH}")
    joblib.dump(model, MODEL_SAVE_PATH)

    print(f"Saving label encoder to {ENCODER_SAVE_PATH}")
    joblib.dump(label_encoder, ENCODER_SAVE_PATH)
    
    print("\n✅ All done! Model and encoder are saved and ready for use.")

if __name__ == "__main__":
    train_model()