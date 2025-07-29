# Heading Classification: Data Labeling and Model Training Guide

## Overview 🎯

This document provides a comprehensive guide to the data labeling process and machine learning model training pipeline for the PDF heading classification system. The project uses a binary LightGBM classifier trained on the DocLayNet dataset to distinguish between headings and body text in PDF documents.

## Table of Contents
- [Dataset: DocLayNet](#dataset-doclaynet)
- [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
- [Feature Engineering](#feature-engineering)
- [Training Process](#training-process)
- [Model Performance](#model-performance)
- [File Structure](#file-structure)

---

## Dataset: DocLayNet 📊

### Source
- **Primary Dataset**: [DocLayNet](https://github.com/DS4SD/DocLayNet) - A large-scale multilingual document layout analysis dataset
- **Size**: Contains over 80,000 document pages across multiple languages
- **Languages**: English, German, French, Japanese, and more
- **Document Types**: Academic papers, reports, manuals, and other professional documents

### Dataset Components
The DocLayNet dataset consists of two main components:

1. **Core Dataset** (`DocLayNet_core.zip`)
   - COCO-format annotations with document layout labels
   - Categories: Title, Section-header, Text, List-item, Table, Figure, etc.
   - Bounding box coordinates for each text element

2. **Extra Dataset** (`DocLayNet_extra.zip`)
   - Detailed text content and font information for each text element
   - Font properties: size, weight, family
   - Text properties: content, positioning

### Data Loading Process

The data loading is handled by the `binary_labeled_data.py` script:

#### Binary Classification Data Processing
- **Purpose**: Creates binary labels (heading vs. body text) for training the LightGBM classifier
- **Process**:
  - Extracts text elements from DocLayNet JSON files
  - Maps COCO annotations to text cells using spatial overlap detection
  - Assigns binary labels: 1 for headings (Title, Section-header), 0 for body text (Text)
  - Implements priority system: Title > Section-header > Text for overlapping annotations
  - Filters out noise (very short text, numbers-only, punctuation-only content)
  - Uses multiprocessing for efficient parallel processing of large datasets

---

## Data Preprocessing Pipeline 🔧

### 1. File Extraction and Indexing
```python
# Extract DocLayNet ZIP files
CORE_ZIP_PATH = "/path/to/DocLayNet_core.zip"
EXTRA_ZIP_PATH = "/path/to/DocLayNet_extra.zip"

# Build lookup tables for efficient processing
extra_data_lookup = {}  # Maps filename -> text cells
annotations_lookup = {}  # Maps image_id -> annotations
```

### 2. Spatial Matching Algorithm
The system uses a center-based containment algorithm to match text cells with annotation regions:

```python
def is_center_inside(inner_box, outer_box):
    """Check if center of inner_box is inside outer_box"""
    ix, iy, iw, ih = inner_box
    ox, oy, ow, oh = outer_box
    center_x = ix + iw / 2
    center_y = iy + ih / 2
    return (ox <= center_x <= (ox + ow)) and (oy <= center_y <= (oy + oh))
```

### 3. Text Filtering and Cleaning
- **Minimum Length**: Text must be ≥3 characters
- **Pattern Filtering**: Removes pure numbers and punctuation-only text
- **Priority System**: Title > Section-header > Text when cells overlap multiple annotations

### 4. Multiprocessing Optimization
- **Parallel Processing**: Uses 24 CPU cores for data processing
- **Caching**: Implements pickle-based caching to avoid reprocessing
- **Memory Efficiency**: Processes documents in batches to manage memory usage

---

## Feature Engineering 🛠️

The model uses 11 carefully engineered features that capture spatial, typographic, and textual properties:

### Spatial Features
1. **`norm_x`**: Normalized horizontal position (x / page_width)
2. **`norm_y`**: Normalized vertical position (y / page_height)

### Typographic Features
3. **`font_size`**: Absolute font size in points
4. **`font_weight`**: Encoded font weight (0=normal, 1=bold, 2=black/heavy)
5. **`relative_font_size`**: Font size relative to document's modal body text size

### Textual Features
6. **`text_length`**: Number of characters in text
7. **`num_words`**: Number of words in text
8. **`is_all_caps`**: Binary flag for ALL CAPS text
9. **`is_centered`**: Binary flag for center-aligned text (within 15% of page center)
10. **`numbering_depth`**: Depth of hierarchical numbering (e.g., "1.2.3" = depth 3)
11. **`percent_punct`**: Percentage of punctuation characters

### Feature Engineering Code
```python
# Font weight encoding
def get_font_weight(font_name):
    if not font_name: return 0
    name_lower = font_name.lower()
    if "black" in name_lower or "heavy" in name_lower: return 2
    if "bold" in name_lower: return 1
    return 0

# Numbering depth calculation
numbering_match = re.match(r'^\d+(\.\d+)*', text_content.strip())
numbering_depth = len(numbering_match.group(0).split('.')) if numbering_match else 0

# Punctuation percentage
percent_punct = sum(1 for char in text_content if char in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~") / len(text_content)
```

---

## Training Process 🚀

### 1. Data Preparation
```python
# Load binary classification dataset
labeled_dataset = datasets.load_from_disk(BINARY_DATA_PATH)
df = labeled_dataset.to_pandas()

# Define features and target
features = ['font_size', 'font_weight', 'norm_x', 'norm_y', 'text_length', 
           'num_words', 'is_all_caps', 'is_centered', 'numbering_depth', 
           'relative_font_size', 'percent_punct']
target = 'is_heading'
```

### 2. Data Split Strategy
```python
# 80% training, 20% testing (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Further split training into 64% train, 16% validation
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)
```

### 3. Class Imbalance Handling
The dataset exhibits significant class imbalance (typically 85% body text, 15% headings):

```python
# Random Oversampling to balance classes
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train_final, y_train_final)
```

### 4. Hyperparameter Optimization
Grid search over key LightGBM parameters to maximize F1-score on validation set:

```python
best_params, best_f1 = {}, 0
for num_leaves in [15, 31, 63]:
    for learning_rate in [0.01, 0.05, 0.1]:
        model = lgbm.LGBMClassifier(
            objective='binary',
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            random_state=42,
            class_weight='balanced',
            n_estimators=500
        )
        model.fit(
            X_train_res, y_train_res,
            eval_set=[(X_val, y_val)],
            callbacks=[lgbm.early_stopping(stopping_rounds=20), lgbm.log_evaluation(0)]
        )
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred)
        if f1 > best_f1:
            best_f1, best_thresh = f1, num_leaves, learning_rate
            best_params = {'num_leaves': num_leaves, 'learning_rate': learning_rate}
```

### 5. Optimal Threshold Selection
Instead of using the default 0.5 threshold, the system finds the optimal threshold on the validation set to maximize F1-score:

```python
# Find threshold that maximizes F1-score on validation set
y_val_proba = model.predict_proba(X_val)[:, 1]
best_thresh, best_f1 = 0.5, 0
for threshold in np.linspace(0.1, 0.9, 81):
    y_pred_t = (y_val_proba > threshold).astype(int)
    f1 = f1_score(y_val, y_pred_t)
    if f1 > best_f1:
        best_f1, best_thresh = f1, threshold

print(f"→ Optimal threshold={best_thresh:.2f}, val-F1={best_f1:.4f}")
```

This threshold optimization is crucial because the default 0.5 threshold may not be optimal for imbalanced datasets.

### 6. Model Training Configuration
Final model training with optimal parameters:

```python
model = lgbm.LGBMClassifier(
    objective='binary',              # Binary classification
    num_leaves=best_params['num_leaves'],  # Typically 31
    learning_rate=best_params['learning_rate'],  # Typically 0.05
    random_state=42,                 # Reproducibility
    class_weight='balanced',         # Handle class imbalance
    n_estimators=1000               # More trees for final model
)

# Early stopping on validation set
model.fit(
    X_train_res, y_train_res,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgbm.early_stopping(stopping_rounds=50),
        lgbm.log_evaluation(0)  # Silent training
    ]
)
```

---

## Model Performance 📈

### Evaluation Metrics
The model is evaluated using multiple metrics to ensure robust performance:

1. **Accuracy**: Overall correctness
2. **AUC-ROC**: Area Under ROC Curve (threshold-independent)
3. **F1-Score**: Harmonic mean of precision and recall
4. **Precision**: Heading detection accuracy
5. **Recall**: Heading detection completeness

### Cross-Validation
5-fold stratified cross-validation for robustness assessment:

```python
# 5-fold stratified cross-validation
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
print(f"Mean CV AUC-ROC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
```

The cross-validation provides confidence intervals and helps detect overfitting.

### Feature Importance Analysis
The model provides interpretable feature importance scores:

```python
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Typically most important features:
# 1. font_size - Larger fonts indicate headings
# 2. relative_font_size - Relative size vs body text
# 3. font_weight - Bold text indicates headings
# 4. is_all_caps - Uppercase headings
# 5. text_length - Headings are typically shorter
```

### Performance Visualization
The training script generates comprehensive performance analysis:

```python
# Feature importance plot
plt.figure(figsize=(10, 8))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
plt.title('Top 10 Feature Importance - Binary Heading Classifier')
plt.savefig('binary_feature_importance.png', dpi=300, bbox_inches='tight')

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
print("           PredBody  PredHead")
print(f"ActualBody {cm[0,0]:6,}    {cm[0,1]:6,}")
print(f"ActualHead {cm[1,0]:6,}    {cm[1,1]:6,}")

# Classification Report
print(classification_report(y_test, y_test_pred, target_names=['Body Text', 'Heading'], digits=4))
```

### Training Output Example
The training script provides detailed progress tracking:

```
🚀 Starting BINARY CLASSIFICATION model training...
Loading binary classification dataset...
Successfully loaded 150,000 text elements.

Class Distribution:
  Body Text (0): 127,500 (85.0%)
  Headings (1): 22,500 (15.0%)

Training set: 96,000 samples
Validation set: 24,000 samples  
Test set: 30,000 samples

→ Best val-F1=0.8542 with {'num_leaves': 31, 'learning_rate': 0.05}
→ Optimal threshold=0.42, val-F1=0.8654

✅✅✅ Binary Classification Model Training Complete!
📊 Final Performance Summary:
   • Test Accuracy: 0.9234
   • Test AUC-ROC: 0.9456
   • Cross-Validation AUC: 0.9401 ± 0.0087
```

---

## File Structure 📁

```
heading_classification/
├── doclaynet_loader/
│   └── binary_labeled_data.py      # Binary classification data preparation
├── scripts/
│   ├── train_binary.py             # Main training script
│   └── text.py                     # Text processing utilities
├── final models/
│   └── lgbm_binary_classifier.pkl  # Trained model
├── pipeline/
│   ├── model/                      # Inference pipeline models
│   ├── input/                      # Test PDFs
│   ├── output/                     # Inference results
│   └── script/
│       ├── infer_binary.py         # Inference script
│       ├── analyze_pdf_features.py # Feature analysis
│       └── compare_outputs.py      # Result comparison
└── DATA_AND_TRAINING_README.md     # This documentation
```

---

## Usage Instructions 🚀

### 1. Data Preparation
```bash
cd heading_classification/doclaynet_loader
python binary_labeled_data.py
```

### 2. Model Training
```bash
cd heading_classification/scripts
python train_binary.py
```

### 3. Model Inference
```bash
cd heading_classification/pipeline/script
python infer_binary.py
```

---

## Key Implementation Details 🔍

### Memory Optimization
- **Streaming Processing**: Processes large datasets without loading everything into memory
- **Caching Strategy**: Uses pickle files to cache preprocessed data
- **Multiprocessing**: Leverages multiple CPU cores for parallel data processing

### Robustness Features
- **Error Handling**: Graceful handling of malformed PDF data
- **Data Validation**: Ensures numeric features are properly formatted
- **Stratified Sampling**: Maintains class distribution across train/val/test splits

### Reproducibility
- **Fixed Random Seeds**: All random operations use `random_state=42`
- **Deterministic Processing**: Consistent results across runs
- **Version Control**: All model parameters and data processing steps documented

---

## Future Improvements 🚀

### Model Enhancements
1. **Ensemble Methods**: Combine multiple models for better accuracy
2. **Deep Learning**: Experiment with transformer-based models
3. **Active Learning**: Iteratively improve with human feedback

### Feature Engineering
1. **Contextual Features**: Consider neighboring text elements
2. **Document Structure**: Incorporate page-level layout patterns
3. **Language-Specific**: Adapt features for different languages

### Data Augmentation
1. **Synthetic Data**: Generate additional training examples
2. **Domain Adaptation**: Fine-tune on specific document types
3. **Multi-Modal**: Incorporate visual features from document images

---

## Contact and Support 📧

For questions about the data labeling process or model training:
- Check the main [README.md](README.md) for project overview
- Review the training script: `scripts/train_binary.py`
- Examine data preparation: `doclaynet_loader/binary_labeled_data.py`

---

*This documentation provides a complete guide to understanding and reproducing the heading classification model training process. The system combines robust data preprocessing, thoughtful feature engineering, and optimized machine learning to achieve reliable heading detection in PDF documents.*
