import fitz  # PyMuPDF
import pandas as pd
import re
from pathlib import Path

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
INPUT_DIR = PIPELINE_DIR / "input"
OUT_DIR = SCRIPT_DIR

# Regex and helpers (copy from infer_binary)
def is_bullet_point(text: str) -> bool:
    return bool(re.match(r'^[•\-\*\+◦◆◇○●■□▪▫]\s+', text))

def get_font_weight(font_name: str) -> int:
    nl = font_name.lower() if font_name else ''
    if 'black' in nl or 'heavy' in nl: return 2
    if 'bold' in nl: return 1
    return 0

# PDF parsing & merging lines
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

# Feature engineering
def engineer_features(blocks):
    df = pd.DataFrame(blocks)
    if df.empty: return df
    df['text']      = df['text'].astype(str)
    df['font_name'] = df['font_name'].astype(str)
    long = df[df['text'].str.split().str.len()>6]
    body_size = long['font_size'].mode().iat[0] if not long.empty else 10.0
    df['font_weight'] = df['font_name'].apply(get_font_weight)
    df['x'] = df['bbox'].apply(lambda b: b[0])
    df['y'] = df['bbox'].apply(lambda b: b[1])
    df['norm_x'] = df['x']/df['page_width']
    df['norm_y'] = df['y']/df['page_height']
    df['text_length'] = df['text'].str.len()
    df['num_words'] = df['text'].str.split().str.len()
    df['is_all_caps'] = df['text'].apply(lambda t:1 if t.isupper() and len(t.strip())>1 else 0)
    cx = df['x'] + (df['bbox'].apply(lambda b: b[2]-b[0]) / 2)
    df['is_centered'] = (abs(cx - df['page_width']/2) < df['page_width']*0.15).astype(int)
    df['numbering_depth'] = df['text'].str.strip().apply(lambda t:len(re.match(r'^(\d+(?:\.\d+)*)',t).group(1).split('.')) if re.match(r'^(\d+(?:\.\d+)*)',t) else 0)
    df['relative_font_size'] = df['font_size']/body_size
    df['percent_punct'] = df['text'].apply(lambda t: sum(1 for c in t if c in r'''!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~''')/len(t) if t else 0)
    return df

# Main analysis
def main():
    # target PDF
    pdf_name = 'file02.pdf'
    pdf_path = INPUT_DIR / pdf_name
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        return

    print(f"Parsing PDF: {pdf_path.name}")
    blocks = parse_pdf_and_merge_lines_v2(pdf_path)
    df = engineer_features(blocks)

    print(f"Total blocks parsed: {len(df)}")
    unique_texts = df['text'].nunique()
    print(f"Unique text strings: {unique_texts}")
    ov = df[df['text'].str.strip() == 'Overview']
    print(f"'Overview' blocks: {len(ov)}\n")
    if not ov.empty:
        print(ov[['page_num','font_size','relative_font_size','numbering_depth','percent_punct','is_all_caps','is_centered']])

    # Save full feature table
    out_csv = OUT_DIR / 'file02_features.csv'
    df.to_csv(out_csv, index=False)
    print(f"Full feature table saved to {out_csv}")

if __name__=='__main__':
    main()
