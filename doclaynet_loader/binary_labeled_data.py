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
FINAL_OUTPUT_PATH = "/home/pi0/heading_classification/doclaynet_training/data_binary_classification"
EXTRACTION_ROOT_PATH = "/home/pi0/heading_classification/doclaynet_training/extracted_files"
MERGE_CACHE_PATH = "/home/pi0/heading_classification/doclaynet_training/cache/merged_binary_data.pkl"
LOOKUP_CACHE_PATH = "/home/pi0/heading_classification/doclaynet_training/cache/lookup_cache_binary.pkl"
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
    page_data = []
    if not text_cells or not image_annotations: return page_data

    page_width, page_height = image_info.get('coco_width', 1025), image_info.get('coco_height', 1025)
    
    # Create a mapping of which cells belong to which category
    cell_categories = {}
    
    for obj in image_annotations:
        category_name = worker_category_mapping.get(obj['category_id'])
        
        # We're interested in headings AND body text (paragraphs)
        if category_name in ["Title", "Section-header", "Text"]:
            contained_cells = [cell for cell in text_cells if is_center_inside(cell['bbox'], obj['bbox'])]
            
            for cell in contained_cells:
                cell_id = id(cell)  # Use memory address as unique identifier
                # Priority: Title > Section-header > Text
                if cell_id not in cell_categories or (
                    category_name == "Title" or 
                    (category_name == "Section-header" and cell_categories[cell_id] != "Title")
                ):
                    cell_categories[cell_id] = category_name
    
    # Now process all categorized cells
    for cell in text_cells:
        cell_id = id(cell)
        if cell_id not in cell_categories:
            continue
            
        category_name = cell_categories[cell_id]
        text_content = cell['text'].strip()
        
        # Filter out very short or empty text
        if len(text_content) < 3:
            continue
            
        # Filter out obvious noise patterns
        if re.match(r'^\d+$', text_content):  # Just numbers
            continue
        if re.match(r'^[^\w\s]*$', text_content):  # Just punctuation
            continue
            
        x, y, w, h = cell['bbox']
        
        # Calculate features
        numbering_match = re.match(r'^\d+(\.\d+)*', text_content.strip())
        numbering_depth = len(numbering_match.group(0).split('.')) if numbering_match else 0
        percent_punct = sum(1 for char in text_content if char in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~") / len(text_content) if text_content else 0
        
        # Binary classification: 1 for headings (Title, Section-header), 0 for body text (Text)
        is_heading = 1 if category_name in ["Title", "Section-header"] else 0
        
        page_data.append({
            "doc_name": image_info['doc_name'], 
            "page_no": image_info['page_no'],
            "original_category": category_name,
            "is_heading": is_heading,  # Binary target variable
            "bbox": [x, y, w, h], 
            "text": text_content,
            "font_size": float(cell['font_size']),
            "font_weight": get_font_weight(cell['font_name']),
            "norm_x": x / page_width, 
            "norm_y": y / page_height,
            "text_length": len(text_content), 
            "num_words": len(text_content.split()),
            "is_all_caps": 1 if text_content.isupper() and len(text_content) > 1 else 0,
            "is_centered": 1 if abs((x + w / 2) - page_width / 2) < (page_width * 0.15) else 0,
            "numbering_depth": numbering_depth,
            "percent_punct": percent_punct,
        })
    
    return page_data

def generate_binary_dataset():
    print("🚀 Starting BINARY CLASSIFICATION dataset generation...")
    
    # Extract files if needed
    core_extract_path = os.path.join(EXTRACTION_ROOT_PATH, "core")
    extra_extract_path = os.path.join(EXTRACTION_ROOT_PATH, "extra")
    
    if not os.path.exists(core_extract_path):
        print("1. Extracting zip files...")
        os.makedirs(EXTRACTION_ROOT_PATH, exist_ok=True)
        with zipfile.ZipFile(CORE_ZIP_PATH, 'r') as zf:
            zf.extractall(core_extract_path)
        with zipfile.ZipFile(EXTRA_ZIP_PATH, 'r') as zf:
            zf.extractall(extra_extract_path)
    else:
        print("1. ✅ Found previously extracted files.")
    
    coco_path = os.path.join(core_extract_path, "COCO")
    extra_json_path_base = os.path.join(extra_extract_path, "JSON")
    
    # Load or build lookup cache
    if os.path.exists(LOOKUP_CACHE_PATH):
        print(f"2. ✅ Found cache. Loading lookup table from {LOOKUP_CACHE_PATH}...")
        with open(LOOKUP_CACHE_PATH, 'rb') as f:
            extra_data_lookup = pickle.load(f)
    else:
        print(f"2. Building lookup tables...")
        extra_data_lookup = {}
        for filename in tqdm(os.listdir(extra_json_path_base), desc="Indexing JSONs"):
            if not filename.endswith('.json'):
                continue
            with open(os.path.join(extra_json_path_base, filename), 'r') as f:
                data = json.load(f)
            
            all_cells = []
            for cell in data.get('cells', []):
                bbox = [float(c) for c in cell.get('bbox', [])]
                if len(bbox) == 4:  # Valid bbox
                    all_cells.append({
                        "bbox": bbox,
                        "text": cell.get('text', ''),
                        "font_size": float(cell.get('font', {}).get('size', 0)),
                        "font_name": cell.get('font', {}).get('name', '')
                    })
            
            extra_data_lookup[os.path.splitext(filename)[0]] = all_cells
        
        with open(LOOKUP_CACHE_PATH, 'wb') as f:
            pickle.dump(extra_data_lookup, f)
    
    # Load or process annotations
    if os.path.exists(MERGE_CACHE_PATH):
        print(f"3. ✅ Found cache. Loading merged data from {MERGE_CACHE_PATH}...")
        with open(MERGE_CACHE_PATH, 'rb') as f:
            all_data = pickle.load(f)
    else:
        print("3. Loading COCO annotations and processing...")
        all_images_info = []
        annotations_lookup = {}
        category_mapping = {}
        
        for split_name in ["train", "val", "test"]:
            coco_file = os.path.join(coco_path, f"{split_name}.json")
            with open(coco_file, 'r') as f:
                core_data = json.load(f)
            
            all_images_info.extend(core_data['images'])
            
            for ann in core_data['annotations']:
                if ann['image_id'] not in annotations_lookup:
                    annotations_lookup[ann['image_id']] = []
                annotations_lookup[ann['image_id']].append(ann)
            
            if not category_mapping:
                category_mapping = {cat['id']: cat['name'] for cat in core_data['categories']}
        
        print(f"4. Processing data in parallel using {NUM_CPUS} CPUs...")
        all_data = []
        
        with multiprocessing.Pool(processes=NUM_CPUS, initializer=init_worker, 
                                initargs=(extra_data_lookup, annotations_lookup, category_mapping)) as pool:
            results = pool.imap_unordered(process_page_worker, all_images_info)
            for page_data in tqdm(results, total=len(all_images_info), desc="Processing pages"):
                all_data.extend(page_data)
        
        with open(MERGE_CACHE_PATH, 'wb') as f:
            pickle.dump(all_data, f)

    print(f"\nFound a total of {len(all_data)} text elements.")
    if not all_data:
        print("❌ Critical error: Found no data.")
        return

    # Convert to dataframe and add relative font size
    print("5. Generating final features...")
    df = pd.DataFrame(all_data)
    
    # Calculate relative font size based on document body text
    doc_body_sizes = df[
        (df['is_heading'] == 0) & (df['num_words'] > 5)
    ].groupby('doc_name')['font_size'].agg(
        lambda x: x.mode()[0] if not x.mode().empty else 10.0
    )
    
    df['body_font_size'] = df['doc_name'].map(doc_body_sizes).fillna(10.0)
    df['relative_font_size'] = df['font_size'] / df['body_font_size']
    
    # Print dataset statistics
    heading_count = (df['is_heading'] == 1).sum()
    body_count = (df['is_heading'] == 0).sum()
    print(f"\nDataset Statistics:")
    print(f"  Headings: {heading_count:,} ({heading_count/len(df)*100:.1f}%)")
    print(f"  Body Text: {body_count:,} ({body_count/len(df)*100:.1f}%)")
    print(f"  Total: {len(df):,}")
    
    # Create and save dataset
    print("6. Creating and saving the final binary classification dataset...")
    final_dataset = datasets.Dataset.from_pandas(df)
    if os.path.exists(FINAL_OUTPUT_PATH):
        shutil.rmtree(FINAL_OUTPUT_PATH)
    final_dataset.save_to_disk(FINAL_OUTPUT_PATH)
    
    print(f"\n✅✅✅ Success! Binary classification dataset created at: {FINAL_OUTPUT_PATH}")
    print(f"Ready for binary classification: Heading (1) vs Body Text (0)")

if __name__ == "__main__":
    generate_binary_dataset()
