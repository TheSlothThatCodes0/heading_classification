import json
import os
import re
import zipfile
import shutil
import datasets
from tqdm import tqdm

def process_data():
    """
    Manually processes DocLayNet data using a permanent extraction folder
    to speed up subsequent runs.
    """
    core_zip_path = "/home/pi0/Downloads/DocLayNet_core.zip"
    extra_zip_path = "/home/pi0/Downloads/DocLayNet_extra.zip"
    final_output_dir = "/home/pi0/heading_classification/doclaynet_training/data_extra"
    
    # --- MODIFICATION: Use a permanent folder instead of a temporary one ---
    extraction_root_path = "/home/pi0/heading_classification/doclaynet_training/extracted_files"
    core_extract_path = os.path.join(extraction_root_path, "core")
    extra_extract_path = os.path.join(extraction_root_path, "extra")

    print("🚀 Starting manual dataset processing...")
    print(f"Using permanent extraction path: {extraction_root_path}")

    # Check if data needs to be extracted
    if not os.path.exists(core_extract_path) or not os.path.exists(extra_extract_path):
        print("Extracted data not found. Extracting zip files now...")
        os.makedirs(extraction_root_path, exist_ok=True)

        print(f"Extracting {os.path.basename(core_zip_path)}...")
        with zipfile.ZipFile(core_zip_path, 'r') as zip_ref:
            zip_ref.extractall(core_extract_path)

        print(f"Extracting {os.path.basename(extra_zip_path)}...")
        with zipfile.ZipFile(extra_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extra_extract_path)
        print("✅ Extraction complete.")
    else:
        print("✅ Found previously extracted files. Skipping extraction.")
    # --- END MODIFICATION ---

    coco_path = os.path.join(core_extract_path, "COCO")
    json_path = os.path.join(extra_extract_path, "JSON")

    if not os.path.isdir(coco_path): raise FileNotFoundError(f"COCO directory not found in {core_extract_path}")
    if not os.path.isdir(json_path): raise FileNotFoundError(f"JSON directory not found in {extra_extract_path}")
    
    all_data = {"train": [], "validation": [], "test": []}
    split_map = {"train": "train", "validation": "val", "test": "test"}

    for split_key, split_file_prefix in split_map.items():
        print(f"\nProcessing '{split_key}' split...")
        
        split_annotation_file = os.path.join(coco_path, f"{split_file_prefix}.json")
        with open(split_annotation_file, 'r') as f:
            core_data = json.load(f)
        
        images_info = core_data['images']
        print(f"Found {len(images_info)} image records for '{split_key}' split.")

        for image_info in tqdm(images_info, desc=f"Parsing {split_key}"):
            png_filename_with_path = image_info['file_name']
            doc_name = image_info['doc_name']
            page_no = image_info['page_no']
            
            base_filename = os.path.basename(png_filename_with_path)
            json_filename = os.path.splitext(base_filename)[0] + ".json"
            extra_json_path = os.path.join(json_path, json_filename)

            if not os.path.exists(extra_json_path):
                continue

            with open(extra_json_path, 'r') as f:
                page_data = json.load(f)

            all_spans = []
            for block in page_data.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        all_spans.append({
                            "text": span.get("text"),
                            "bbox": span.get("bbox"),
                            "font_size": span.get("size"),
                            "font_name": span.get("font"),
                            "flags": span.get("flags"),
                        })
            
            all_data[split_key].append({
                "doc_name": doc_name,
                "page_no": page_no,
                "text_spans": all_spans,
            })

    print("\nCreating final DatasetDict...")
    
    final_dataset = datasets.DatasetDict({
        split: datasets.Dataset.from_list(data)
        for split, data in all_data.items()
    })
    
    print("\nFinal dataset splits:")
    print(final_dataset)

    if os.path.exists(final_output_dir):
        shutil.rmtree(final_output_dir)

    print(f"\nSaving dataset to disk at: {final_output_dir}")
    final_dataset.save_to_disk(final_output_dir)

    print("\n✅ All done! Dataset successfully processed and saved.")
    print("You can now re-run the script to skip the slow extraction step.")

if __name__ == "__main__":
    process_data()