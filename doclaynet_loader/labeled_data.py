import json
import os
import zipfile
import shutil
import pandas as pd
from tqdm import tqdm
import datasets
import multiprocessing
import re
import pickle
from functools import partial

# --- Configuration ---
CORE_ZIP_PATH = "/home/pi0/Downloads/DocLayNet_core.zip"
EXTRA_ZIP_PATH = "/home/pi0/Downloads/DocLayNet_extra.zip"
FINAL_OUTPUT_PATH = "/home/pi0/heading_classification/doclaynet_training/data_final_headings"
EXTRACTION_ROOT_PATH = "/home/pi0/heading_classification/doclaynet_training/extracted_files"
MERGE_CACHE_PATH = "/home/pi0/heading_classification/doclaynet_training/cache/merged_headings_medium.pkl" # New cache file
LOOKUP_CACHE_PATH = "/home/pi0/heading_classification/doclaynet_training/cache/lookup_cache_medium.pkl" # New cache file
NUM_CPUS = 24

# --- Worker Functions for Multiprocessing ---
worker_extra_lookup, worker_annotations_lookup, worker_category_mapping = None, None, None

def init_worker(extra_lookup, annotations_lookup, category_mapping):
    global worker_extra_lookup, worker_annotations_lookup, worker_category_mapping
    worker_extra_lookup, worker_annotations_lookup, worker_category_mapping = extra_lookup, annotations_lookup, category_mapping

def is_center_inside(inner_box, outer_box):
    if not inner_box or not outer_box: return False
    ix, iy, iw, ih = inner_box; ox, oy, ow, oh = outer_box
    center_x = ix + iw / 2; center_y = iy + ih / 2
    return (ox <= center_x <= (ox + ow)) and (oy <= center_y <= (oy + oh))

def get_font_weight(font_name):
    if not font_name: return 0
    name_lower = font_name.lower()
    if "black" in name_lower or "heavy" in name_lower: return 2
    if "bold" in name_lower: return 1
    return 0

def process_page_worker(image_info):
    base_filename = os.path.splitext(os.path.basename(image_info['file_name']))[0]
    text_cells = worker_extra_lookup.get(base_filename, [])
    image_annotations = worker_annotations_lookup.get(image_info['id'], [])
    page_headings = []
    if not text_cells or not image_annotations: return page_headings

    page_width, page_height = image_info.get('coco_width', 1025), image_info.get('coco_height', 1025)
    
    for obj in image_annotations:
        category_name = worker_category_mapping.get(obj['category_id'])
        if category_name in ["Title", "Section-header"]:
            contained_cells = [cell for cell in text_cells if is_center_inside(cell['bbox'], obj['bbox'])]
            if not contained_cells: continue
            
            contained_cells.sort(key=lambda c: c['bbox'][0])
            text_content = " ".join(cell['text'].strip() for cell in contained_cells).strip()
            if not text_content: continue

            x, y, w, h = obj['bbox']
            
            # --- ADDING REQUESTED FEATURES ---
            numbering_match = re.match(r'^\d+(\.\d+)*', text_content.strip())
            numbering_depth = len(numbering_match.group(0).split('.')) if numbering_match else 0
            percent_punct = sum(1 for char in text_content if char in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~") / len(text_content) if text_content else 0

            page_headings.append({
                "doc_name": image_info['doc_name'], "page_no": image_info['page_no'],
                "category": category_name, "bbox": obj['bbox'], "text": text_content,
                "font_size": sum(c['font_size'] for c in contained_cells) / len(contained_cells),
                "font_weight": get_font_weight(contained_cells[0]['font_name']),
                "norm_x": x / page_width, "norm_y": y / page_height,
                "text_length": len(text_content), "num_words": len(text_content.split()),
                "is_all_caps": 1 if text_content.isupper() and len(text_content) > 1 else 0,
                "is_centered": 1 if abs((x + w / 2) - page_width / 2) < (page_width * 0.15) else 0,
                "numbering_depth": numbering_depth,
                "percent_punct": percent_punct,
            })
    return page_headings

def generate_final_dataset():
    # This function is a simplified representation. The full parallel and caching logic is included.
    print("🚀 Starting MEDIUM-WEIGHT dataset generation...")
    # ... (Full script logic from before) ...
    # Key change is the post-processing step below
    core_extract_path = os.path.join(EXTRACTION_ROOT_PATH, "core"); extra_extract_path = os.path.join(EXTRACTION_ROOT_PATH, "extra")
    if not os.path.exists(core_extract_path):
        print("1. Extracting zip files..."); os.makedirs(EXTRACTION_ROOT_PATH, exist_ok=True)
        with zipfile.ZipFile(CORE_ZIP_PATH, 'r') as zf: zf.extractall(core_extract_path)
        with zipfile.ZipFile(EXTRA_ZIP_PATH, 'r') as zf: zf.extractall(extra_extract_path)
    else: print("1. ✅ Found previously extracted files.")
    coco_path = os.path.join(core_extract_path, "COCO"); extra_json_path_base = os.path.join(extra_extract_path, "JSON")
    if os.path.exists(LOOKUP_CACHE_PATH):
        print(f"2. ✅ Found cache. Loading lookup table from {LOOKUP_CACHE_PATH}...");
        with open(LOOKUP_CACHE_PATH, 'rb') as f: extra_data_lookup = pickle.load(f)
    else:
        print(f"2. Building lookup tables...")
        extra_data_lookup = {}
        for filename in tqdm(os.listdir(extra_json_path_base), desc="Indexing JSONs"):
            if not filename.endswith('.json'): continue
            with open(os.path.join(extra_json_path_base, filename), 'r') as f: data = json.load(f)
            all_cells = [{"bbox": [float(c) for c in cell.get('bbox', [])], "text": cell.get('text', ''), "font_size": float(cell.get('font', {}).get('size', 0)), "font_name": cell.get('font', {}).get('name', '')} for cell in data.get('cells', [])]
            extra_data_lookup[os.path.splitext(filename)[0]] = all_cells
        with open(LOOKUP_CACHE_PATH, 'wb') as f: pickle.dump(extra_data_lookup, f)
    
    if os.path.exists(MERGE_CACHE_PATH):
        print(f"3. ✅ Found cache. Loading merged headings from {MERGE_CACHE_PATH}...");
        with open(MERGE_CACHE_PATH, 'rb') as f: all_headings = pickle.load(f)
    else:
        all_images_info, annotations_lookup, category_mapping = [], {}, {}
        for split_name in ["train", "val", "test"]:
            with open(os.path.join(coco_path, f"{split_name}.json"), 'r') as f: core_data = json.load(f)
            all_images_info.extend(core_data['images'])
            for ann in core_data['annotations']:
                if ann['image_id'] not in annotations_lookup: annotations_lookup[ann['image_id']] = []
                annotations_lookup[ann['image_id']].append(ann)
            if not category_mapping: category_mapping = {cat['id']: cat['name'] for cat in core_data['categories']}
        print(f"3. Merging data in parallel using {NUM_CPUS} CPUs..."); all_headings = []
        with multiprocessing.Pool(processes=NUM_CPUS, initializer=init_worker, initargs=(extra_data_lookup, annotations_lookup, category_mapping)) as pool:
            results = pool.imap_unordered(process_page_worker, all_images_info)
            for page_headings in tqdm(results, total=len(all_images_info), desc="Merging pages"):
                all_headings.extend(page_headings)
        with open(MERGE_CACHE_PATH, 'wb') as f: pickle.dump(all_headings, f)

    print(f"\nFound a total of {len(all_headings)} headings.")
    if not all_headings: print("❌ Critical error: Found no headings."); return

    print("4. Generating H-level labels and final features...")
    headings_df = pd.DataFrame(all_headings)

    # --- ADDING relative_font_size FEATURE ---
    doc_body_sizes = headings_df[headings_df['num_words'] > 5].groupby('doc_name')['font_size'].agg(lambda x: x.mode()[0] if not x.mode().empty else 10.0)
    headings_df['body_font_size'] = headings_df['doc_name'].map(doc_body_sizes).fillna(10.0)
    headings_df['relative_font_size'] = headings_df['font_size'] / headings_df['body_font_size']
    
    headings_df.sort_values(by=["doc_name", "relative_font_size", "font_weight", "norm_y"], ascending=[True, False, False, True], inplace=True)
    def assign_levels(group):
        h_level = 1; last_style = None; levels = []
        for _, row in group.iterrows():
            current_style = (round(row['relative_font_size'], 1), row['font_weight'])
            if last_style and current_style != last_style: h_level += 1
            levels.append(f"H{h_level}" if h_level <= 6 else "Other-Heading"); last_style = current_style
        group['heading_level'] = levels
        group.loc[group['category'] == 'Title', 'heading_level'] = 'H1'
        return group
    labeled_df = headings_df.groupby("doc_name", group_keys=False).apply(assign_levels)

    print("5. Creating and saving the final dataset...")
    final_dataset = datasets.Dataset.from_pandas(labeled_df)
    if os.path.exists(FINAL_OUTPUT_PATH): shutil.rmtree(FINAL_OUTPUT_PATH)
    final_dataset.save_to_disk(FINAL_OUTPUT_PATH)
    print(f"\n✅✅✅ Success! The MEDIUM-WEIGHT dataset has been created at: {FINAL_OUTPUT_PATH}")

if __name__ == "__main__":
    generate_final_dataset()