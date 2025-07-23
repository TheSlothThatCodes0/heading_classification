import json
import os
import datasets
import collections

# ---------------------------------------------------------------------------
# PART 1: DATASET BUILDER DEFINITION
# (This is the class you provided, defining HOW to process the data)
# ---------------------------------------------------------------------------

class COCOBuilderConfig(datasets.BuilderConfig):
    """BuilderConfig for DocLayNet."""
    def __init__(self, name, splits, **kwargs):
        super().__init__(name, **kwargs)
        self.splits = splits

_CITATION = """\
@article{doclaynet2022,
  title = {DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis},
  doi = {10.1145/3534678.353904},
  url = {https://arxiv.org/abs/2206.01062},
  author = {Pfitzmann, Birgit and Auer, Christoph and Dolfi, Michele and Nassar, Ahmed S and Staar, Peter W J},
  year = {2022}
}
"""

_DESCRIPTION = "DocLayNet is a human-annotated document layout segmentation dataset from a broad variety of document sources."
_HOMEPAGE = "https://developer.ibm.com/exchanges/data/all/doclaynet/"
_LICENSE = "CDLA-Permissive-1.0"

# Using the local ZIP file as the source
_URLs = {
    "core": "/home/pi0/Downloads/DocLayNet_core.zip",
}

class DocLayNetBuilder(datasets.GeneratorBasedBuilder):
    """DatasetBuilder for DocLayNet to process local files."""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIG_CLASS = COCOBuilderConfig
    BUILDER_CONFIGS = [
        COCOBuilderConfig(name="doclaynet", splits=["train", "val", "test"]),
    ]
    DEFAULT_CONFIG_NAME = "doclaynet"

    def _info(self):
        features = datasets.Features(
            {
                "image_id": datasets.Value("int64"),
                "image": datasets.Image(),
                "width": datasets.Value("int32"),
                "height": datasets.Value("int32"),
                "doc_category": datasets.Value("string"),
                "collection": datasets.Value("string"),
                "doc_name": datasets.Value("string"),
                "page_no": datasets.Value("int64"),
                "objects": [
                    {
                        "category_id": datasets.ClassLabel(
                            names=[
                                "Caption", "Footnote", "Formula", "List-item",
                                "Page-footer", "Page-header", "Picture",
                                "Section-header", "Table", "Text", "Title",
                            ]
                        ),
                        "image_id": datasets.Value("string"),
                        "id": datasets.Value("int64"),
                        "area": datasets.Value("int64"),
                        "bbox": datasets.Sequence(datasets.Value("float32"), length=4),
                        "segmentation": [[datasets.Value("float32")]],
                        "iscrowd": datasets.Value("bool"),
                        "precedence": datasets.Value("int32"),
                    }
                ],
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # dl_manager.download_and_extract will use the local path from _URLs
        archive_path = dl_manager.download_and_extract(_URLs)
        core_path = archive_path["core"]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "json_path": os.path.join(core_path, "COCO", "train.json"),
                    "image_dir": os.path.join(core_path, "PNG"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "json_path": os.path.join(core_path, "COCO", "val.json"),
                    "image_dir": os.path.join(core_path, "PNG"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "json_path": os.path.join(core_path, "COCO", "test.json"),
                    "image_dir": os.path.join(core_path, "PNG"),
                },
            ),
        ]

    def _generate_examples(self, json_path, image_dir):
        """Yields examples as (key, example) tuples."""
        with open(json_path, encoding="utf8") as f:
            data = json.load(f)

        # Create a mapping from image_id to its annotations
        image_id_to_annotations = collections.defaultdict(list)
        for ann in data["annotations"]:
            image_id_to_annotations[ann["image_id"]].append(ann)

        # Yield each image with its corresponding annotations
        for idx, image_info in enumerate(data["images"]):
            annotations = image_id_to_annotations[image_info["id"]]
            
            # Map category_id from 1-based to 0-based for ClassLabel
            for ann in annotations:
                ann["category_id"] = ann["category_id"] - 1

            yield idx, {
                "image_id": image_info["id"],
                "image": os.path.join(image_dir, image_info["file_name"]),
                "width": image_info["width"],
                "height": image_info["height"],
                "doc_category": image_info["doc_category"],
                "collection": image_info["collection"],
                "doc_name": image_info["doc_name"],
                "page_no": image_info["page_no"],
                "objects": annotations,
            }

# ---------------------------------------------------------------------------
# PART 2: EXECUTION SCRIPT
# (This part runs the builder and saves the result to your folder)
# ---------------------------------------------------------------------------

def main():
    """
    Main function to run the dataset processing and save the output.
    """
    # Define the final output directory for the processed dataset
    output_dir = "/home/pi0/heading_classification/doclaynet_training/data"
    
    print("🚀 Starting dataset processing...")
    print(f"Final processed dataset will be saved to: {output_dir}")

    # 1. Instantiate the dataset builder
    builder = DocLayNetBuilder()
    print("Dataset builder created.")

    # 2. Download and prepare the data. This runs the main logic in the builder.
    # It uses the local zip, extracts it to a temporary cache, and generates the splits.
    builder.download_and_prepare()
    print("Data preparation complete.")

    # 3. Load the prepared data as a DatasetDict object
    doclaynet_dataset = builder.as_dataset()
    print("\nDataset splits loaded:")
    print(doclaynet_dataset)

    # 4. Save the final, processed dataset to your desired directory
    print(f"\nSaving dataset to disk at: {output_dir}")
    doclaynet_dataset.save_to_disk(output_dir)
    
    print("\n✅ Dataset successfully processed and saved!")
    print(f"You can now load it directly using: datasets.load_from_disk('{output_dir}')")

if __name__ == "__main__":
    main()