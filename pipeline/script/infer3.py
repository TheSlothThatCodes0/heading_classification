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
    return df

def infer_pdf(pdf_path: Path):
    blocks = parse_pdf_and_merge_lines_v2(pdf_path)
    feats  = engineer_features(blocks)

    # Rule 0: drop trivial
    feats = feats[feats["text"].str.strip().str.len() > 3]

    # Rule 0.5: ensure page‑1’s 3 largest always in
    p1 = feats[feats["page_num"] == 1]
    if not p1.empty:
        top3 = p1.nlargest(3, "font_size")
        feats = pd.concat([feats, top3]).drop_duplicates()

    # Rule 1: keep ≥1.0× or any numbered
    feats = feats[
        (feats["relative_font_size"] >= 1.0)
        | (feats["numbering_depth"] > 0)
    ]
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

    # Rule 5: sentence/body filters, glue only on long lines
    hdrs = hdrs[~hdrs["text"].str.match(date_pattern)]
    hdrs = hdrs[~hdrs["text"].str.match(footer_pattern)]
    hdrs = hdrs[~hdrs["text"].str.fullmatch(r"\d+")]
    long_lines = hdrs["num_words"] > 7
    hdrs = hdrs[~(long_lines & hdrs["text"].str.contains(glue_pattern))]
    hdrs = hdrs[~hdrs["text"].str.match(r"^[a-z]")]
    hdrs = hdrs[~hdrs["text"].str.endswith(".")]
    hdrs = hdrs[hdrs["percent_punct"] < 0.4]
    hdrs = hdrs[~hdrs["text"].str.match(r"^\d+\s+\d+$")]
    hdrs = hdrs[hdrs["percent_digit"] < 0.5]

    # Final sort and JSON build
    final = hdrs.sort_values(["page_num","norm_y"])
    p1f = final[final["page_num"] == 1]
    if not p1f.empty:
        mx = p1f["font_size"].max()
        title = " ".join(p1f[p1f["font_size"] == mx]["text"])
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
