import fitz  # PyMuPDF
import joblib
import json
import pandas as pd
import re
from pathlib import Path
from collections import Counter

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
INPUT_DIR = PIPELINE_DIR / "input"
OUTPUT_DIR = PIPELINE_DIR / "output/infer2"
MODEL_DIR = PIPELINE_DIR / "model"
MODEL_PATH = MODEL_DIR / "lgbm_model_medium.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoder_medium.pkl"

# --- Helper Functions ---
def get_font_weight(font_name):
    if not font_name: return 0
    name_lower = font_name.lower()
    if "black" in name_lower or "heavy" in name_lower: return 2
    if "bold" in name_lower: return 1
    return 0

def parse_pdf_and_merge_lines(pdf_path):
    doc = fitz.open(pdf_path)
    merged_blocks = []
    for page_num, page in enumerate(doc):
        page_width, page_height = page.rect.width, page.rect.height
        blocks = page.get_text("dict", flags=11)["blocks"]
        raw_lines = []
        for block in blocks:
            if block['type'] == 0:
                for line in block['lines']:
                    line_text = " ".join([span['text'].strip() for span in line['spans'] if span['text'].strip()])
                    if not line_text: continue
                    avg_size = sum(s['size'] for s in line['spans']) / len(line['spans']) if line['spans'] else 0
                    font_name = line['spans'][0]['font'] if line['spans'] else ""
                    raw_lines.append({ "text": line_text, "bbox": line['bbox'], "size": avg_size, "font": font_name, "page_num": page_num + 1, "page_width": page_width, "page_height": page_height })
        if not raw_lines: continue
        current_line = raw_lines[0]
        for i in range(1, len(raw_lines)):
            next_line = raw_lines[i]
            if (current_line['font'] == next_line['font'] and abs(current_line['size'] - next_line['size']) < 1 and
                abs(next_line['bbox'][1] - current_line['bbox'][3]) < (current_line['size'] * 0.8) and
                len(current_line['text'].split()) + len(next_line['text'].split()) < 25):
                current_line['text'] += " " + next_line['text']
                new_bbox = (min(current_line['bbox'][0], next_line['bbox'][0]), min(current_line['bbox'][1], next_line['bbox'][1]), max(current_line['bbox'][2], next_line['bbox'][2]), max(current_line['bbox'][3], next_line['bbox'][3]))
                current_line['bbox'] = new_bbox
            else:
                merged_blocks.append(current_line)
                current_line = next_line
        merged_blocks.append(current_line)
    doc.close()
    return merged_blocks

def engineer_features(text_blocks):
    if not text_blocks: return pd.DataFrame()
    df = pd.DataFrame(text_blocks)
    df.rename(columns={'size': 'font_size', 'font': 'font_name'}, inplace=True)
    
    if not df.empty and len(df[df['text'].str.split().str.len() > 4]) > 0:
        body_size = df[df['text'].str.split().str.len() > 4]['font_size'].mode()[0]
    else:
        body_size = 10.0
    
    df['font_weight'] = df['font_name'].apply(get_font_weight)
    df['x'] = df['bbox'].apply(lambda b: b[0]); df['y'] = df['bbox'].apply(lambda b: b[1])
    df['w'] = df['bbox'].apply(lambda b: b[2] - b[0])
    df['norm_x'] = df['x'] / df['page_width']; df['norm_y'] = df['y'] / df['page_height']
    df['text_length'] = df['text'].str.len(); df['num_words'] = df['text'].str.split().str.len()
    df['is_all_caps'] = df['text'].apply(lambda x: 1 if x.isupper() and len(x.strip()) > 1 else 0)
    center_x = df['x'] + df['w'] / 2
    df['is_centered'] = (abs(center_x - df['page_width'] / 2) < (df['page_width'] * 0.15)).astype(int)
    
    df['numbering_depth'] = df['text'].str.strip().apply(lambda x: len(re.match(r'^\d+(\.\d+)*', x).group(0).split('.')) if re.match(r'^\d+(\.\d+)*', x) else 0)
    df['relative_font_size'] = df['font_size'] / body_size
    df['percent_punct'] = df['text'].apply(lambda x: sum(1 for char in x if char in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~") / len(x) if x else 0)

    return df

def main():
    print("🚀 Starting FINAL HYBRID PDF outline extraction...")
    try:
        model = joblib.load(MODEL_PATH)
        label_encoder = joblib.load(ENCODER_PATH)
        print("✅ Model and encoder loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Error: Model or encoder not found at '{MODEL_PATH}' or '{ENCODER_PATH}'.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    
    for pdf_path in pdf_files:
        print(f"\nProcessing '{pdf_path.name}'...")
        try:
            merged_blocks = parse_pdf_and_merge_lines(pdf_path)
            features_df = engineer_features(merged_blocks)
            if features_df.empty: continue
                
            feature_columns = [
                'font_size', 'font_weight', 'norm_x', 'norm_y', 'text_length', 
                'num_words', 'is_all_caps', 'is_centered',
                'numbering_depth', 'relative_font_size', 'percent_punct'
            ]
            
            X_predict = features_df[feature_columns]
            
            predictions_numeric = model.predict(X_predict)
            features_df['predicted_level'] = label_encoder.inverse_transform(predictions_numeric)
            
            headings_df = features_df[features_df['predicted_level'].isin(['H1', 'H2', 'H3', 'H4', 'H5', 'H6'])].copy()
            
            # --- NEW, AGGRESSIVE POST-PROCESSING FILTER ---
            # Rule 1: Filter out text that looks like sentences or noise
            headings_df = headings_df[headings_df['num_words'] < 15]
            headings_df = headings_df[~headings_df['text'].str.strip().str.endswith(('.', ':'))]
            
            # Rule 2: Filter out specific noise patterns like dates, page numbers, and bullet points
            date_pattern = r'\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{1,2}-\d{1,2}|(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{1,2},\s\d{4}'
            page_num_pattern = r'^\s*(page|pages)?\s*\d+\s*$'
            bullet_pattern = r'^\s*[•●-]\s+'
            form_noise_pattern = r'(-|_|\.){4,}' # Catches '----'

            headings_df = headings_df[~headings_df['text'].str.contains(date_pattern, regex=True, case=False)]
            headings_df = headings_df[~headings_df['text'].str.lower().str.match(page_num_pattern)]
            headings_df = headings_df[~headings_df['text'].str.match(bullet_pattern)]
            headings_df = headings_df[~headings_df['text'].str.contains(form_noise_pattern, regex=True)]

            # Rule 3: Remove headers/footers (text that repeats on multiple pages)
            # This is more effective if we process all pages first, but for now, we'll use a simpler text counter
            text_counts = headings_df['text'].value_counts()
            repeated_text = text_counts[text_counts > 2].index.tolist()
            if repeated_text:
                headings_df = headings_df[~headings_df['text'].isin(repeated_text)]

            title = "Untitled Document"
            h1s = headings_df[headings_df['predicted_level'] == 'H1'].sort_values(by=['page_num', 'norm_y'])
            if not h1s.empty: title = h1s.iloc[0]['text']

            outline = []
            final_headings = headings_df[headings_df['predicted_level'].isin(['H1', 'H2', 'H3'])]
            for _, row in final_headings.sort_values(by=['page_num', 'norm_y']).iterrows():
                outline.append({"level": row['predicted_level'], "text": row['text'].strip(), "page": int(row['page_num'])})
                
            output_data = {"title": title.strip(), "outline": outline}
            output_path = OUTPUT_DIR / (pdf_path.stem + ".json")
            with open(output_path, 'w') as f: json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"✅ Successfully created outline at '{output_path}'")
        except Exception as e:
            print(f"An error occurred while processing {pdf_path.name}: {e}")

    print("\nAll PDFs processed.")

if __name__ == "__main__":
    main()