````markdown
# PDF Outline Extractor

## Inspiration 🌟
PDFs power everything from research papers to user manuals, yet their visual layout obscures their logical structure. In Round 1A (“Understand Your Document”) of the Adobe India Hackathon 2025—Challenge Theme: Connecting the Dots Through Docs—our mission is to teach machines to “read” a PDF and extract its Title and hierarchical headings (H1, H2, H3) in JSON. This structured outline becomes the foundation for semantic search, knowledge graphs, and intelligent document workflows.

## What It Does 💡

**Input:**  
- A PDF file (≤ 50 pages)

**Output:**  
```json
{
  "title": "Understanding AI",
  "outline": [
    { "level": "H1", "text": "Introduction",    "page": 1 },
    { "level": "H2", "text": "What is AI?",     "page": 2 },
    { "level": "H3", "text": "History of AI",   "page": 3 }
  ]
}
````

## Our Solution 💪

We combine an open-source LightGBM binary classifier ([LightGBM](https://github.com/microsoft/LightGBM)), fine-tuned on the DocLayNet dataset, with spatial, typographic, and textual rules. Fine-tuning on DocLayNet enables robust, multilingual heading detection—for English, German, French, Japanese, and more—while preserving a lightweight (< 200 MB) model footprint.

## Methodology 📝

### 1. Parsing & Line-Merging

* Use **PyMuPDF** to extract text spans from each page.
* Compute each line’s **average font size** and **dominant font name**.
* Sort lines by vertical position, then group adjacent lines into blocks when:

  * Vertical gap < 50 % of font size
  * Font size within ±1 pt and same font weight
* Merge each group into a single block (concatenate text, unify bounding box, retain font/page info).

### 2. Feature Engineering

* Build a **pandas** DataFrame of blocks with features:

  * **Spatial:** `x`, `y`, `width`, normalized `norm_x`, `norm_y`
  * **Typography:** `font_size`, `relative_font_size` (vs. modal body size), `font_weight` (0/1/2)
  * **Textual:** `text_length`, `num_words`, `is_all_caps`, `is_centered`, `percent_punct`, `numbering_depth`

### 3. Inference & Filtering

* Run **LightGBM** to predict heading vs. body blocks.
* Drop false positives via rules:

  * Boilerplate (running headers/footers)
  * Dates, page-numbers, bullets, table-like rows
  * Excessive punctuation (> 40 %)
  * Overly long lines (> 15 words)

### 4. Hierarchical Level Assignment

* Traverse filtered headings in reading order using a **stack** of `(level, font_size)`.
* Assign levels:

  * **Deeper** if font size significantly smaller
  * **Same** if within ±5 %
  * **Ascend** (or reset to H1) if larger
* Cap levels at H6.

### 5. Title Detection

* From page 1 headings, compute a score:

  ```text
  score = relative_font_size − norm_y
  ```
* Select the heading with the highest score as the **Title**.

## What Sets This Apart 🌠

* **Hybrid ML + Rules:** LightGBM’s speed and precision, enhanced by DocLayNet fine-tuning for multilingual support
* **Robust Merging:** Accurately reconstructs fragmented text blocks across diverse layouts
* **Adaptive Hierarchy:** Tolerance-driven algorithm flexibly adapts to varied document designs


The container will process all `/app/input/*.pdf` and write corresponding `.json` outlines into `/app/output`.
