import fitz  # PyMuPDF
import joblib
import json
import pandas as pd
import re
from pathlib import Path
from collections import Counter
import spacy

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
INPUT_DIR = PIPELINE_DIR / "input"
OUTPUT_DIR = PIPELINE_DIR / "output"
MODEL_DIR = PIPELINE_DIR / "model"
MODEL_PATH = MODEL_DIR / "lgbm_model_ultimate.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoder_ultimate.pkl"

# --- Helper Functions ---
def get_font_weight(font_name):
    if not font_name: return 0
    name_lower = font_name.lower()
    if "black" in name_lower or "heavy" in name_lower: return 2
    if "bold" in name_lower: return 1
    return 0

def is_center_inside(inner_box, outer_box):
    if not inner_box or not outer_box: return False
    ix0, iy0, ix1, iy1 = inner_box; ox0, oy0, ox1, oy1 = outer_box
    center_x = (ix0 + ix1) / 2; center_y = (iy0 + iy1) / 2
    return (ox0 <= center_x <= ox1) and (oy0 <= center_y <= oy1)

def parse_pdf_and_merge_lines(pdf_path):
    doc = fitz.open(pdf_path)
    all_blocks = []
    for page_num, page in enumerate(doc):
        page_width, page_height = page.rect.width, page.rect.height
        tables = page.find_tables()
        table_bboxes = [list(t.bbox) for t in tables]
        blocks = page.get_text("dict", flags=11)["blocks"]
        raw_lines = []
        for block in blocks:
            if block['type'] == 0:
                for line in block['lines']:
                    line_text = " ".join([span['text'].strip() for span in line['spans'] if span['text'].strip()])
                    if not line_text: continue
                    avg_size = sum(s['size'] for s in line['spans']) / len(line['spans'])
                    font_name = line['spans'][0]['font']
                    raw_lines.append({
                        "text": line_text, "bbox": line['bbox'], "size": avg_size,
                        "font": font_name, "page_num": page_num + 1,
                        "page_width": page_width, "page_height": page_height,
                        "table_bboxes": table_bboxes
                    })
        if not raw_lines: continue
        current_line = raw_lines[0]
        for i in range(1, len(raw_lines)):
            next_line = raw_lines[i]
            if (current_line['font'] == next_line['font'] and 
                abs(current_line['size'] - next_line['size']) < 1 and
                abs(next_line['bbox'][1] - current_line['bbox'][3]) < (current_line['size'] * 0.5) and
                len(current_line['text'].split()) < 20):
                current_line['text'] += " " + next_line['text']
                new_bbox = (min(current_line['bbox'][0], next_line['bbox'][0]), min(current_line['bbox'][1], next_line['bbox'][1]),
                            max(current_line['bbox'][2], next_line['bbox'][2]), max(current_line['bbox'][3], next_line['bbox'][3]))
                current_line['bbox'] = new_bbox
            else:
                all_blocks.append(current_line)
                current_line = next_line
        all_blocks.append(current_line)
    doc.close()
    return all_blocks

def engineer_features(text_blocks, nlp):
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
    df['norm_y'] = df['y'] / df['page_height']
    df['text_length'] = df['text'].str.len(); df['num_words'] = df['text'].str.split().str.len()
    df['is_all_caps'] = df['text'].apply(lambda x: 1 if x.isupper() and len(x.strip()) > 1 else 0)
    center_x = df['x'] + df['w'] / 2
    df['is_centered'] = (abs(center_x - df['page_width'] / 2) < (df['page_width'] * 0.15)).astype(int)
    
    df['relative_font_size'] = df['font_size'] / body_size
    df['ends_with_colon'] = df['text'].str.strip().str.endswith(':').astype(int)
    df['starts_with_bullet'] = df['text'].str.strip().str.match(r'^[•●-]\s').astype(int)
    
    def get_nlp_features(text):
        doc = nlp(text)
        return {
            'has_verb': 1 if any(token.pos_ == 'VERB' for token in doc) else 0,
            'is_date': 1 if any(ent.label_ == 'DATE' for ent in doc.ents) else 0,
            'contains_page_word': 1 if 'page' in text.lower() else 0,
            'percent_punct': sum(1 for char in text if char in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~") / len(text) if text else 0
        }

    nlp_features_df = df['text'].apply(get_nlp_features).apply(pd.Series)
    df = pd.concat([df, nlp_features_df], axis=1)

    df['is_in_table'] = df.apply(lambda row: 1 if any(is_center_inside(row['bbox'], t_bbox) for t_bbox in row['table_bboxes']) else 0, axis=1)
    df['numbering_depth'] = df['text'].str.strip().apply(lambda x: len(re.match(r'^\d+(\.\d+)*', x).group(0).split('.')) if re.match(r'^\d+(\.\d+)*', x) else 0)
    df['norm_x'] = df['x'] / df['page_width']
    
    return df

def main():
    print("🚀 Starting FINAL HYBRID PDF outline extraction...")
    try:
        model = joblib.load(MODEL_PATH)
        label_encoder = joblib.load(ENCODER_PATH)
        nlp = spacy.load("en_core_web_sm")
        print("✅ Models and NLP pipeline loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    
    for pdf_path in pdf_files:
        print(f"\nProcessing '{pdf_path.name}'...")
        try:
            merged_blocks = parse_pdf_and_merge_lines(pdf_path)
            features_df = engineer_features(merged_blocks, nlp)
            if features_df.empty:
                print("Could not extract any processable text blocks."); continue
                
            # --- THE FIX IS HERE ---
            # This list now has 16 features, perfectly matching the training script
            feature_columns = [
                'font_size', 'font_weight', 'norm_y', 'text_length', 
                'num_words', 'is_all_caps', 'is_centered', 'has_verb', 'is_date',
                'contains_page_word', 'percent_punct', 'is_in_table', 'numbering_depth',
                'starts_with_bullet', 'ends_with_colon', 'relative_font_size'
            ]
            
            X_predict = features_df[feature_columns]
            predictions_numeric = model.predict(X_predict)
            features_df['predicted_level'] = label_encoder.inverse_transform(predictions_numeric)
            
            headings_df = features_df[features_df['predicted_level'].isin(['H1', 'H2', 'H3', 'H4', 'H5', 'H6'])].copy()
            
            # Post-processing
            headings_df = headings_df[headings_df['num_words'] <= 15]
            headings_df = headings_df[~headings_df['text'].str.strip().str.endswith('.')]
            text_counts = headings_df['text'].value_counts()
            repeated_text = text_counts[text_counts > 2].index.tolist()
            if repeated_text:
                headings_df = headings_df[~headings_df['text'].isin(repeated_text)]

            final_headings = []
            last_level = 0
            for _, row in headings_df.sort_values(by=['page_num', 'norm_y']).iterrows():
                current_level = int(row['predicted_level'][1:])
                if current_level > last_level + 1:
                    current_level = last_level + 1
                row['predicted_level'] = f"H{current_level}"
                last_level = current_level
                final_headings.append(row)
            
            if not final_headings:
                output_data = {"title": "Untitled Document", "outline": []}
            else:
                headings_df = pd.DataFrame(final_headings)
                title = "Untitled Document"
                h1s = headings_df[(headings_df['predicted_level'] == 'H1') & (headings_df['page_num'] <= 2)]
                if not h1s.empty:
                    title = h1s.sort_values(by='font_size', ascending=False).iloc[0]['text']

                outline = []
                outline_df = headings_df[headings_df['predicted_level'].isin(['H1', 'H2', 'H3'])]
                for _, row in outline_df.iterrows():
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