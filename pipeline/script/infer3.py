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
OUTPUT_DIR   = PIPELINE_DIR / "output/infer5"
MODEL_DIR    = PIPELINE_DIR / "model"
MODEL_PATH   = MODEL_DIR / "lgbm_model_medium.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoder_medium.pkl"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Precompile regexes
date_pattern        = re.compile(
    r'(?i)(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{1,2},?\s\d{4}'
    r'|\d{1,2}/\d{1,2}/\d{2,4}'
)
footer_pattern      = re.compile(
    r'(?i)^\s*(page\s*\d+(?:\s*of\s*\d+)?|version\s*[\d.]+\s*page\s*\d+)\s*$'
)
glue_pattern = re.compile(
    r'\b(?:the|an|is|in|of|for|to|will|be|are|was|that|by|with|and|as)\b',
    flags=re.IGNORECASE
)

def is_bullet_point(text):
    """Detect bullet point lists that shouldn't be headings"""
    # Check for bullet-like patterns (•, -, *, etc.)
    bullet_pattern = re.compile(r'^[•\-\*\+◦◆◇◈○●■□▪▫]+\s')
    if bullet_pattern.search(text):
        return True
    # Check for multiple bullet points in same text
    if text.count('•') > 1 or text.count('●') > 1:
        return True
    return False

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
    df["text"]      = df["text"].astype(str)
    df["font_name"] = df["font_name"].astype(str)

    long_lines = df[df["text"].str.split().str.len() > 6]
    body_size  = long_lines["font_size"].mode().iat[0] if not long_lines.empty else 10.0

    df["font_weight"]        = df["font_name"].apply(get_font_weight)
    df["x"]                  = df["bbox"].apply(lambda b: b[0])
    df["y"]                  = df["bbox"].apply(lambda b: b[1])
    df["w"]                  = df["bbox"].apply(lambda b: b[2] - b[0])
    df["norm_x"]             = df["x"] / df["page_width"]
    df["norm_y"]             = df["y"] / df["page_height"]
    df["text_length"]        = df["text"].str.len()
    df["num_words"]          = df["text"].str.split().str.len()
    df["is_all_caps"]        = df["text"].apply(lambda t: 1 if t.isupper() and len(t.strip()) > 1 else 0)
    cx                       = df["x"] + df["w"] / 2
    df["is_centered"]        = (abs(cx - df["page_width"] / 2) < df["page_width"] * 0.15).astype(int)
    df["numbering_depth"]    = df["text"].str.strip().apply(
        lambda t: len(re.match(r"^(\d+(?:\.\d+)*)", t).group(1).split(".")) if re.match(r"^(\d+(?:\.\d+)*)", t) else 0
    )
    df["relative_font_size"] = df["font_size"] / body_size
    df["percent_punct"]      = df["text"].apply(
        lambda t: sum(1 for c in t if c in r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""") / len(t) if t else 0
    )

    df["percent_digit"]      = df["text"].apply(
        lambda t: sum(c.isdigit() for c in t) / len(t) if t else 0
    )

    # Add cover page detection
    df["is_cover_page"] = (df["page_num"] == 1).astype(int)
    
    # Add position-based features for cover detection
    df["is_upper_half"] = (df["norm_y"] < 0.5).astype(int)
    df["is_very_top"] = (df["norm_y"] < 0.3).astype(int)

    # Add improved bullet point detection
    df["is_bullet_point"] = df["text"].apply(is_bullet_point).astype(bool)
    
    # Add a stricter "likely_body" detector
    df["likely_body"] = ((df["num_words"] > 15) | 
                        (df["text"].str.endswith('.')) | 
                        (df["text"].str.contains(r'\b(?:and|the|that|with)\b', regex=True))).astype(bool)

    return df

def infer_pdf(pdf_path: Path):
    blocks = parse_pdf_and_merge_lines_v2(pdf_path)
    feats  = engineer_features(blocks)

    # Rule 0: drop trivial
    feats = feats[feats["text"].str.strip().str.len() > 3]

    # Rule 0.5: ensure page‑1's largest fonts always protected
    p1 = feats[feats["page_num"] == 1]
    if not p1.empty:
        # Get top fonts by size (more than just 3)
        top_fonts = p1.nlargest(min(5, len(p1)), "font_size")
        # Also protect any very large fonts (>1.2x body)
        large_fonts = p1[p1["relative_font_size"] > 1.2]
        if not large_fonts.empty:
            protected = pd.concat([top_fonts, large_fonts]).drop_duplicates()
        else:
            protected = top_fonts
        # Only concat if we have new items to add
        new_items = protected[~protected.index.isin(feats.index)]
        if not new_items.empty:
            feats = pd.concat([feats, new_items])

    # Rule 1: better filtering criteria - more restrictive
    keep_mask = (
        (feats["relative_font_size"] >= 1.1)  # More restrictive font size threshold
        | ((feats["numbering_depth"] > 0) & ~feats["is_bullet_point"] & (feats["relative_font_size"] >= 1.0))  # Numbered sections with decent font
        | ((feats["page_num"] == 1) & (feats["relative_font_size"] >= 1.05))  # Page 1 content with larger font
        | ((feats["text"].str.strip().str.endswith(':')) & (feats["num_words"] < 6) & (feats["relative_font_size"] >= 1.0))  # Section headers with colon
        | (feats["is_all_caps"] & (feats["num_words"] <= 4) & (feats["relative_font_size"] >= 1.0))  # Short all-caps headings
    )
    feats = feats[keep_mask]
    if feats.empty:
        return {"title": "", "outline": []}

    # Predict with exactly 11 features
    FEATURE_COLS = [
        "font_size","font_weight",
        "norm_x","norm_y",
        "text_length","num_words",
        "is_all_caps","is_centered",
        "numbering_depth","relative_font_size",
        "percent_punct",
    ]
    X = feats[FEATURE_COLS]
    if X.empty:
        return {"title": "", "outline": []}

    preds = model.predict(X)
    feats["predicted_level"] = label_encoder.inverse_transform(preds)

    # Rule 2: override numbering, clamp to H4
    num_mask = feats["numbering_depth"] > 0
    feats.loc[num_mask, "predicted_level"] = (
        feats.loc[num_mask, "numbering_depth"]
             .clip(1,4)
             .astype(int)
             .apply(lambda d: f"H{d}")
    )

    hdrs = feats[feats["predicted_level"].str.startswith("H")].copy()

    # Rule 3: remove repeated across >2 pages
    tp = hdrs[['text','page_num','norm_y']].drop_duplicates()
    stats = tp.groupby('text').agg(page_count=('page_num','nunique'),
                                   y_min=('norm_y','min'),
                                   y_max=('norm_y','max'))
    hdrs = hdrs.merge(stats, left_on='text', right_index=True)

    # Rule 4: drop header/footer noise by position+frequency
    noise = (
        (hdrs['page_count'] > 2)
        & ((hdrs['y_min'] < 0.1) | (hdrs['y_max'] > 0.9))
    )
    hdrs = hdrs[~noise].drop(['page_count','y_min','y_max'], axis=1)

    # Rule 5: sentence/body filters with improved protection for real headings
    hdrs = hdrs[~hdrs["text"].str.match(date_pattern)]
    hdrs = hdrs[~hdrs["text"].str.match(footer_pattern)]
    hdrs = hdrs[~hdrs["text"].str.fullmatch(r"\d+")]
    
    # Rule 5.1: More aggressively filter bullet points except key section names
    bullet_filter = hdrs["is_bullet_point"] & ~(hdrs["text"].str.contains(r':(?:\s*)?$', regex=True))
    hdrs = hdrs[~bullet_filter]
    
    # Rule 5.2: Protect headings with colon endings (likely section headers)
    colon_headers = hdrs["text"].str.match(r'^[^\.]+:\s*$', na=False)
    
    # Rule 5.3: Filter body-like text more aggressively
    try:
        body_text_filter = (
            hdrs["likely_body"] &
            (hdrs["num_words"] > 10) &
            ~colon_headers &
            ~((hdrs["page_num"] == 1) & (hdrs["relative_font_size"] > 1.3))
        )
        hdrs = hdrs[~body_text_filter]
    except Exception as e:
        print(f"Warning: Rule 5.3 failed: {e}")
        # Fall back to simpler filtering
        hdrs = hdrs[~hdrs["likely_body"] | (hdrs["num_words"] <= 10)]
    
    # Rule 5.4: Fix heading levels to prefer H1-H3 over H4-H6 for main headings
    # This aligns more with expected output's hierarchy
    try:
        numbered_headings = hdrs["text"].str.match(r'^\d+\.', na=False) 
        mask1 = hdrs["predicted_level"].isin(["H4", "H5", "H6"]) & colon_headers
        mask2 = hdrs["predicted_level"].isin(["H5", "H6"]) & numbered_headings
        
        hdrs.loc[mask1, "predicted_level"] = "H3"
        hdrs.loc[mask2, "predicted_level"] = "H3"
    except Exception as e:
        print(f"Warning: Rule 5.4 failed: {e}")
        pass
    
    # Rule 6: Additional structural filtering
    # Filter out very similar/duplicate headings on the same page
    hdrs['text_lower'] = hdrs['text'].str.lower().str.strip()
    page_text_counts = hdrs.groupby(['page_num', 'text_lower']).size()
    duplicate_mask = hdrs.set_index(['page_num', 'text_lower']).index.isin(
        page_text_counts[page_text_counts > 1].index
    )
    
    # Keep only the first occurrence of duplicates within each page
    hdrs = hdrs[~duplicate_mask | ~hdrs.duplicated(['page_num', 'text_lower'])]
    hdrs = hdrs.drop('text_lower', axis=1)
    
    # Rule 7: Filter very short non-meaningful text
    hdrs = hdrs[~((hdrs['num_words'] == 1) & (hdrs['text_length'] < 4) & ~hdrs['is_all_caps'])]

    # Keep existing filters
    hdrs = hdrs[hdrs["percent_punct"] < 0.4]
    hdrs = hdrs[~hdrs["text"].str.match(r"^\d+\s+\d+$", na=False)]
    hdrs = hdrs[hdrs["percent_digit"] < 0.5]

    # Final sort and JSON build
    final = hdrs.sort_values(["page_num","norm_y"])
    
    # Better title selection with cleaning
    p1f = final[final["page_num"] == 1]
    if not p1f.empty:
        # First try: find title candidates with large font
        candidates = p1f[p1f["relative_font_size"] > 1.2]
        
        if candidates.empty:
            # Fall back to top 3 items on page 1
            candidates = p1f.head(3)
        
        # Choose title with best combination of font size and position
        if len(candidates) > 1:
            candidates = candidates.copy()  # Avoid SettingWithCopyWarning
            candidates['title_score'] = (candidates['relative_font_size'] * 10) - candidates['norm_y']
            best_candidate = candidates.loc[candidates['title_score'].idxmax()]
        else:
            best_candidate = candidates.iloc[0]
        
        # Clean OCR artifacts from title more aggressively
        title_text = best_candidate["text"]
        # Remove repeated character patterns like "RFP: RFP:" 
        title = re.sub(r'\b(\w+):\s+\1:', r'\1:', title_text)
        # Remove spaced out letters like "R quest f r Pr oposal"
        title = re.sub(r'([A-Z])\s+([a-z])\s+([A-Z])', r'\1\2\3', title)
        # Remove repeated words
        words = title.split()
        seen = set()
        clean_words = []
        for word in words:
            if word.lower() not in seen or len(word) > 3:
                clean_words.append(word)
                seen.add(word.lower())
        title = ' '.join(clean_words)
        title = re.sub(r'\s{2,}', ' ', title).strip()
    else:
        title = final.iloc[0]["text"] if not final.empty else ""

    outline = [
        {"level": r["predicted_level"], "text": r["text"].strip(), "page": int(r["page_num"])}
        for _, r in final.iterrows()
    ]
    return {"title": title, "outline": outline}

def main():
    global model, label_encoder
    try:
        model         = joblib.load(MODEL_PATH)
        label_encoder = joblib.load(ENCODER_PATH)
    except:
        print(f"❌ Could not load model or encoder from {MODEL_PATH} / {ENCODER_PATH}")
        return

    for pdf in INPUT_DIR.glob("*.pdf"):
        print(f"▶️ Processing {pdf.name}")
        out = OUTPUT_DIR / f"{pdf.stem}.json"
        try:
            result = infer_pdf(pdf)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"   ✅ Wrote {out.name}")
        except Exception as e:
            print(f"   ❌ Failed on {pdf.name}: {e}")

if __name__ == "__main__":
    main()
