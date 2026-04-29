import os
import urllib.request
import zipfile

DATA_URL = 'https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip'
ZIP_PATH = 'data.zip'
EXTRACT_DIR = './' # Extracts to ./data/ because the zip file contains a 'data' folder

print("Downloading dataset (~300MB), this may take a few minutes...")
urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
print("Download complete. Extracting files...")

with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)

print("Extraction complete. Cleaning up zip file...")
os.remove(ZIP_PATH)
print("Dataset is ready in the 'data' folder!")
