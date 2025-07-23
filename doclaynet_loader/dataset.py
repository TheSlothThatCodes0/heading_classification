from doclaynet_loader import COCODataset # Import your class from the script
import datasets

# 1. Instantiate the builder for the specific config you want
# The config is defined in BUILDER_CONFIGS in your script
builder = COCODataset(name="2022.08")

# 2. Manually run the download and preparation process
# This will download, extract, and process the data, saving it to the cache.
builder.download_and_prepare()

# 3. Get the dataset splits as a DatasetDict
# This loads the newly prepared data from the cache.
doclaynet = builder.as_dataset()

# The result is the same DatasetDict
print(doclaynet)