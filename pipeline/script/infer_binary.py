import fitz  # PyMuPDF
import joblib
import json
import pandas as pd
import re
from pathlib import Path

# --- Configuration ---
SCRIPT_DIR   = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
INPUT_DIR    = PIPELINE_DIR / "input"
OUTPUT_DIR   = PIPELINE_DIR / "output/infer_binary"
MODEL_DIR    = PIPELINE_DIR / "model"
MODEL_PATH   = MODEL_DIR / "lgbm_binary_classifier.pkl"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Regex patterns for noise filtering
date_pattern     = re.compile(r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}", re.IGNORECASE)
footer_pattern   = re.compile(r"(?i)page\s*\d+|^\d+\s+\d+\s+\d+\s+\d+$|version|copyright")
repeated_numbers = re.compile(r"^\d+(?:\s+\d+)+$")

# Bullet detection
def is_bullet_point(text: str) -> bool:
    return bool(re.match(r'^[•\-\*\+◦◆◇○●■□▪▫]\s+', text))

# Font weight helper
def get_font_weight(font_name: str) -> int:
    nl = font_name.lower() if font_name else ''
    if 'black' in nl or 'heavy' in nl: return 2
    if 'bold' in nl: return 1
    return 0

# PDF parsing & merging lines (copy of infer4 version)
def parse_pdf_and_merge_lines_v2(pdf_path: Path):
    doc = fitz.open(str(pdf_path))
    all_blocks = []
    for page_idx, page in enumerate(doc):
        w, h = page.rect.width, page.rect.height
        lines = []
        for block in page.get_text('dict', flags=11)['blocks']:
            if block['type'] != 0: continue
            for line in block['lines']:
                txt = ' '.join(span['text'].strip() for span in line['spans'] if span['text'].strip())
                if not txt: continue
                avg_size = sum(s['size'] for s in line['spans']) / len(line['spans'])
                font_name = max(set(s['font'] for s in line['spans']), key=lambda f: sum(1 for s in line['spans'] if s['font']==f))
                lines.append({
                    'text': txt, 'bbox': line['bbox'], 'size': avg_size,
                    'font': font_name, 'page_num': page_idx+1,
                    'page_width': w, 'page_height': h
                })
        if not lines: continue
        lines.sort(key=lambda L: L['bbox'][1])
        grouped, cur = [], [lines[0]]
        for ln in lines[1:]:
            prev = cur[-1]
            gap = ln['bbox'][1] - prev['bbox'][3]
            same = abs(ln['size']-prev['size'])<1 and get_font_weight(ln['font'])==get_font_weight(prev['font'])
            if gap < prev['size']*0.5 and same:
                cur.append(ln)
            else:
                grouped.append(cur)
                cur = [ln]
        grouped.append(cur)
        for grp in grouped:
            xs = [L['bbox'][0] for L in grp]; ys = [L['bbox'][1] for L in grp]
            x2 = [L['bbox'][2] for L in grp]; y2 = [L['bbox'][3] for L in grp]
            all_blocks.append({
                'text': ' '.join(L['text'] for L in grp),
                'bbox': (min(xs),min(ys),max(x2),max(y2)),
                'font_size': grp[0]['size'], 'font_name': grp[0]['font'],
                'page_num': grp[0]['page_num'], 'page_width': w, 'page_height': h
            })
    doc.close()
    return all_blocks

# Feature engineering (copy of infer4 version)
def engineer_features(blocks):
    df = pd.DataFrame(blocks)
    if df.empty: return df
    df['text']      = df['text'].astype(str)
    df['font_name'] = df['font_name'].astype(str)
    # body font size (mode of longer lines)
    long = df[df['text'].str.split().str.len()>6]
    body_size = long['font_size'].mode().iat[0] if not long.empty else 10.0
    df['font_weight']       = df['font_name'].apply(get_font_weight)
    df['x']                 = df['bbox'].apply(lambda b: b[0])
    df['y']                 = df['bbox'].apply(lambda b: b[1])
    df['w']                 = df['bbox'].apply(lambda b: b[2]-b[0])
    df['norm_x']            = df['x']/df['page_width']
    df['norm_y']            = df['y']/df['page_height']
    df['text_length']       = df['text'].str.len()
    df['num_words']         = df['text'].str.split().str.len()
    df['is_all_caps']       = df['text'].apply(lambda t:1 if t.isupper() and len(t.strip())>1 else 0)
    cx = df['x'] + df['w']/2
    df['is_centered']       = (abs(cx - df['page_width']/2) < df['page_width']*0.15).astype(int)
    df['numbering_depth']   = df['text'].str.strip().apply(lambda t:len(re.match(r'^(\d+(?:\.\d+)*)',t).group(1).split('.')) if re.match(r'^(\d+(?:\.\d+)*)',t) else 0)
    df['relative_font_size']= df['font_size']/body_size
    df['percent_punct']     = df['text'].apply(lambda t: sum(1 for c in t if c in r'''!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~''')/len(t) if t else 0)
    return df

# Prediction and rule-based filtering
FEATURE_COLS = [
    'font_size','font_weight','norm_x','norm_y',
    'text_length','num_words','is_all_caps','is_centered',
    'numbering_depth','relative_font_size','percent_punct'
]

def infer_pdf_binary(pdf_path: Path):
    blocks = parse_pdf_and_merge_lines_v2(pdf_path)
    feats = engineer_features(blocks)
    if feats.empty: return {'outline': []}
    X = feats[FEATURE_COLS]
    if X.empty: return {'outline': []}
    preds = model.predict(X)
    feats['is_heading'] = preds
    hdrs = feats[feats['is_heading']==1].copy()
    # rule-based false-positive removal
    hdrs = hdrs[hdrs['percent_punct']<0.4]
    hdrs = hdrs[~hdrs['text'].str.match(date_pattern, na=False)]
    hdrs = hdrs[~hdrs['text'].str.match(footer_pattern, na=False)]
    hdrs = hdrs[~hdrs['text'].str.match(repeated_numbers, na=False)]
    hdrs = hdrs[~hdrs['text'].apply(is_bullet_point)]
    hdrs = hdrs[~hdrs['text'].str.fullmatch(r'[A-Za-z]{1,2}', na=False)]
    # Remove overly long headings (likely false positives)
    hdrs = hdrs[hdrs['num_words'] <= 15]
    # Sort by location
    final = hdrs.sort_values(['page_num','norm_y'])
    if final.empty:
        return {'title': '', 'outline': []}

    # Assign heading levels based on relative font size scaling
    max_fs = final['relative_font_size'].max()
    min_fs = final['relative_font_size'].min()
    # Scale to levels 1-6: largest font -> H1, smallest -> H6
    def compute_level(r):
        if max_fs == min_fs:
            return 1
        lvl = round((max_fs - r['relative_font_size'])/(max_fs - min_fs)*5) + 1
        return min(max(int(lvl), 1), 6)
    final['level'] = final.apply(lambda r: f"H{compute_level(r)}", axis=1)

    # Title detection: top-level header on page 1
    title = ''
    p1 = final[final['page_num'] == 1]
    if not p1.empty:
        h1 = p1[p1['level'] == 'H1']
        candidates = h1 if not h1.empty else p1
        candidates = candidates.copy()
        candidates['score'] = candidates['relative_font_size'] - candidates['norm_y']
        best = candidates.loc[candidates['score'].idxmax()]
        title = best['text'].strip()

    # Build outline
    outline = [
        {'level': r['level'], 'text': r['text'].strip(), 'page': int(r['page_num'])}
        for _, r in final.iterrows()
    ]
    return {'title': title, 'outline': outline}

# Main loop

def main():
    global model
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    for pdf in INPUT_DIR.glob('*.pdf'):
        print(f"▶️ Processing {pdf.name}")
        out = OUTPUT_DIR / f"{pdf.stem}.json"
        try:
            res = infer_pdf_binary(pdf)
            with open(out,'w',encoding='utf-8') as f:
                json.dump(res,f,indent=2,ensure_ascii=False)
            print(f"   ✅ Wrote {out.name} ({len(res['outline'])} headings)")
        except Exception as e:
            print(f"   ❌ Error {pdf.name}: {e}")

if __name__=='__main__':
    main()
