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
OUTPUT_DIR   = PIPELINE_DIR / "output/infer4"
MODEL_DIR    = PIPELINE_DIR / "model"
MODEL_PATH   = MODEL_DIR / "lgbm_model_medium.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoder_medium.pkl"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Simple regex patterns for obvious non-headings
date_pattern = re.compile(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}', re.IGNORECASE)
footer_pattern = re.compile(r'(?i)page\s*\d+|^\d+\s+\d+\s+\d+\s+\d+$|version|copyright', flags=re.IGNORECASE)
repeated_numbers = re.compile(r'^\d+(\s+\d+)+$')  # Pattern like "12 12" or "12 12 12 12"

def is_bullet_point(text):
    """Simple bullet point detection"""
    return bool(re.match(r'^[•\-\*\+◦◆◇◈○●■□▪▫]\s', text))

def get_font_weight(font_name: str) -> int:
    if not font_name:
        return 0
    nl = font_name.lower()
    if "black" in nl or "heavy" in nl:
        return 2
    if "bold" in nl:
        return 1
    return 0

def parse_pdf_and_merge_lines_v2(pdf_path: Path):
    doc = fitz.open(str(pdf_path))
    all_blocks = []
    for page_idx, page in enumerate(doc):
        w, h = page.rect.width, page.rect.height
        lines = []
        for block in page.get_text("dict", flags=11)["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                txt = " ".join(span["text"].strip()
                               for span in line["spans"]
                               if span["text"].strip())
                if not txt:
                    continue
                avg_size = sum(s["size"] for s in line["spans"]) / len(line["spans"])
                font_name = max(
                    set(s["font"] for s in line["spans"]),
                    key=lambda f: sum(1 for s in line["spans"] if s["font"] == f)
                )
                lines.append({
                    "text": txt,
                    "bbox": line["bbox"],
                    "size": avg_size,
                    "font": font_name,
                    "page_num": page_idx + 1,
                    "page_width": w,
                    "page_height": h,
                })
        if not lines:
            continue
        lines.sort(key=lambda L: L["bbox"][1])
        grouped, cur = [], [lines[0]]
        for ln in lines[1:]:
            prev = cur[-1]
            gap = ln["bbox"][1] - prev["bbox"][3]
            same_style = (
                abs(ln["size"] - prev["size"]) < 1
                and get_font_weight(ln["font"]) == get_font_weight(prev["font"])
            )
            if gap < prev["size"] * 0.5 and same_style:
                cur.append(ln)
            else:
                grouped.append(cur)
                cur = [ln]
        grouped.append(cur)
        for grp in grouped:
            xs = [L["bbox"][0] for L in grp]
            ys = [L["bbox"][1] for L in grp]
            x2 = [L["bbox"][2] for L in grp]
            y2 = [L["bbox"][3] for L in grp]
            all_blocks.append({
                "text": " ".join(L["text"] for L in grp),
                "bbox": (min(xs), min(ys), max(x2), max(y2)),
                "size": grp[0]["size"],
                "font": grp[0]["font"],
                "page_num": grp[0]["page_num"],
                "page_width": w,
                "page_height": h,
            })
    doc.close()
    return all_blocks

def engineer_features(blocks):
    if not blocks:
        return pd.DataFrame()
    df = pd.DataFrame(blocks)
    df.rename(columns={"size": "font_size", "font": "font_name"}, inplace=True)
    df["text"] = df["text"].astype(str)
    df["font_name"] = df["font_name"].astype(str)

    # Calculate body font size (most common font size for longer lines)
    long_lines = df[df["text"].str.split().str.len() > 6]
    body_size = long_lines["font_size"].mode().iat[0] if not long_lines.empty else 10.0

    # Basic features
    df["font_weight"] = df["font_name"].apply(get_font_weight)
    df["x"] = df["bbox"].apply(lambda b: b[0])
    df["y"] = df["bbox"].apply(lambda b: b[1])
    df["w"] = df["bbox"].apply(lambda b: b[2] - b[0])
    df["norm_x"] = df["x"] / df["page_width"]
    df["norm_y"] = df["y"] / df["page_height"]
    df["text_length"] = df["text"].str.len()
    df["num_words"] = df["text"].str.split().str.len()
    df["is_all_caps"] = df["text"].apply(lambda t: 1 if t.isupper() and len(t.strip()) > 1 else 0)
    
    # Centering check
    cx = df["x"] + df["w"] / 2
    df["is_centered"] = (abs(cx - df["page_width"] / 2) < df["page_width"] * 0.15).astype(int)
    
    # Numbering depth
    df["numbering_depth"] = df["text"].str.strip().apply(
        lambda t: len(re.match(r"^(\d+(?:\.\d+)*)", t).group(1).split(".")) if re.match(r"^(\d+(?:\.\d+)*)", t) else 0
    )
    
    # Relative font size
    df["relative_font_size"] = df["font_size"] / body_size
    
    # Punctuation percentage
    df["percent_punct"] = df["text"].apply(
        lambda t: sum(1 for c in t if c in r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""") / len(t) if t else 0
    )

    return df

def infer_pdf(pdf_path: Path):
    blocks = parse_pdf_and_merge_lines_v2(pdf_path)
    feats = engineer_features(blocks)

    if feats.empty:
        return {"title": "", "outline": []}

    # BALANCED RULES - More selective but still generalizable
    
    # Rule 1: Drop trivial text (too short)
    feats = feats[feats["text"].str.strip().str.len() > 4]
    
    # Rule 2: Keep potential headings - be more selective
    keep_mask = (
        (feats["relative_font_size"] >= 1.02) |  # Slightly larger fonts
        ((feats["numbering_depth"] > 0) & (feats["num_words"] <= 8)) |  # Short numbered items
        ((feats["is_all_caps"] == 1) & (feats["num_words"] <= 5)) |  # Short all caps text
        ((feats["font_weight"] > 0) & (feats["num_words"] <= 10)) |  # Short bold text
        ((feats["text"].str.endswith(':')) & (feats["num_words"] <= 6)) |  # Short colon endings
        ((feats["is_centered"] == 1) & (feats["num_words"] <= 8))  # Short centered text
    )
    feats = feats[keep_mask]
    
    if feats.empty:
        return {"title": "", "outline": []}

    # Predict with model
    FEATURE_COLS = [
        "font_size", "font_weight",
        "norm_x", "norm_y",
        "text_length", "num_words",
        "is_all_caps", "is_centered",
        "numbering_depth", "relative_font_size",
        "percent_punct",
    ]
    X = feats[FEATURE_COLS]
    if X.empty:
        return {"title": "", "outline": []}

    preds = model.predict(X)
    feats["predicted_level"] = label_encoder.inverse_transform(preds)

    # Rule 3: Override numbering levels
    num_mask = feats["numbering_depth"] > 0
    feats.loc[num_mask, "predicted_level"] = (
        feats.loc[num_mask, "numbering_depth"]
             .clip(1, 4)
             .astype(int)
             .apply(lambda d: f"H{d}")
    )

    # Rule 4: Keep only predicted headings
    hdrs = feats[feats["predicted_level"].str.startswith("H")].copy()
    
    if hdrs.empty:
        return {"title": "", "outline": []}

    # Rule 5: Remove obvious noise more aggressively
    hdrs = hdrs[~hdrs["text"].str.match(date_pattern, na=False)]
    hdrs = hdrs[~hdrs["text"].str.match(footer_pattern, na=False)]
    hdrs = hdrs[~hdrs["text"].str.match(repeated_numbers, na=False)]  # Remove "12 12 12 12" patterns
    hdrs = hdrs[~hdrs["text"].str.fullmatch(r"\d+", na=False)]  # Pure numbers
    
    # Additional date patterns
    hdrs = hdrs[~hdrs["text"].str.contains(r'(?i)(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d+,?\s+\d{4}', na=False)]
    
    # Remove single letters or very short meaningless text
    hdrs = hdrs[~hdrs["text"].str.fullmatch(r'[A-Za-z]{1,2}', na=False)]
    
    # Rule 6: Remove repeated headers/footers across pages
    text_counts = hdrs.groupby('text')['page_num'].nunique()
    repeated_across_pages = text_counts[text_counts > 2].index  # Appears on 3+ pages
    hdrs = hdrs[~hdrs['text'].isin(repeated_across_pages)]
    
    # Rule 7: Remove bullet points
    hdrs = hdrs[~hdrs["text"].apply(is_bullet_point)]
    
    # Rule 8: Remove very high punctuation content (likely noise)
    hdrs = hdrs[hdrs["percent_punct"] < 0.4]
    
    # Rule 9: Remove very long text (likely body paragraphs)
    hdrs = hdrs[hdrs["num_words"] <= 15]
    
    # Rule 10: Remove copyright, version patterns, and incomplete sentences
    copyright_pattern = re.compile(r'(?i)(?:copyright|version \d+|\d{4} page \d+)')
    hdrs = hdrs[~hdrs["text"].str.contains(copyright_pattern, na=False)]
    
    # Rule 11: Remove incomplete sentences and fragments  
    hdrs = hdrs[~hdrs["text"].str.endswith('.') | (hdrs["num_words"] <= 5)]  # Allow short text ending with periods
    
    # Rule 12: Filter out structural patterns (no content-specific words)
    structural_noise_patterns = [
        r'^\$\d+[MK]?\s+\$\d+[MK]?$',  # Money patterns like "$50M $75M"
        r'\d{4}\s+\d{4}',  # Year patterns like "2007 2017"
        r'^[A-Z\s]+\s+[A-Z\s]+\s+\d{4}\s+\d{4}$',  # Pattern like "FUNDING SOURCE 2007 2017"
        r'^[A-Z\s]+\s+[A-Z\s]+$',  # Short all-caps phrases that are likely table headers
        r'^result:\s+',  # Lines starting with "Result:" (likely table/list items)
    ]
    
    for pattern in structural_noise_patterns:
        hdrs = hdrs[~hdrs["text"].str.contains(pattern, na=False, regex=True)]

    # Final sort
    final = hdrs.sort_values(["page_num", "norm_y"])
    
    # Simple title selection - prefer larger fonts and higher position
    title = ""
    p1f = final[final["page_num"] == 1]
    if not p1f.empty:
        # Find the best title candidate based on font size and position
        p1f_scored = p1f.copy()
        p1f_scored['title_score'] = p1f_scored['relative_font_size'] - p1f_scored['norm_y']
        title_candidate = p1f_scored.loc[p1f_scored['title_score'].idxmax()]
        title = title_candidate["text"].strip()
    elif not final.empty:
        title = final.iloc[0]["text"].strip()

    outline = [
        {"level": r["predicted_level"], "text": r["text"].strip(), "page": int(r["page_num"])}
        for _, r in final.iterrows()
    ]
    
    return {"title": title, "outline": outline}

def main():
    global model, label_encoder
    try:
        model = joblib.load(MODEL_PATH)
        label_encoder = joblib.load(ENCODER_PATH)
    except Exception as e:
        print(f"❌ Could not load model or encoder from {MODEL_PATH} / {ENCODER_PATH}")
        print(f"Error: {e}")
        return

    for pdf in INPUT_DIR.glob("*.pdf"):
        print(f"▶️ Processing {pdf.name}")
        out = OUTPUT_DIR / f"{pdf.stem}.json"
        try:
            result = infer_pdf(pdf)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"   ✅ Wrote {out.name} - {len(result['outline'])} headings found")
        except Exception as e:
            print(f"   ❌ Failed on {pdf.name}: {e}")

if __name__ == "__main__":
    main()
