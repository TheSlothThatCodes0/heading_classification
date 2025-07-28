import datasets
import pandas as pd
from langdetect import detect, DetectorFactory

# fix randomness in langdetect
DetectorFactory.seed = 0

# paths
BINARY_DATA_PATH = "/home/pi0/heading_classification/doclaynet_training/data_binary_classification"
OUTPUT_PARQUET   = "/home/pi0/heading_classification/doclaynet_training/data_binary_classification_with_lang.parquet"

def main():
    # load
    ds = datasets.load_from_disk(BINARY_DATA_PATH)
    df = ds.to_pandas()

    # detect language on the 'text' field (fallback to 'unknown' on failure)
    def safe_detect(s):
        try:
            return detect(s)
        except:
            return "unknown"

    print("Detecting languages…")
    df["language"] = df["text"].astype(str).apply(safe_detect)

    # save to parquet (or overwrite your original dataset)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"✅ Wrote augmented data with `language` to {OUTPUT_PARQUET}")

if __name__ == "__main__":
    main()