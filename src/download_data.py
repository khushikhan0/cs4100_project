import os
import requests
import zipfile

# Make sure the folder exists
raw_dir = "data/raw"
os.makedirs(raw_dir, exist_ok=True)

# Files to download
files = {
    "fma_small.zip": "https://os.unil.cloud.switch.ch/fma/fma_small.zip",
    "fma_metadata.zip": "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
}

for filename, url in files.items():
    filepath = os.path.join(raw_dir, filename)
    
    # Skip download if file already exists
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Finished downloading {filename}")

    # Extract the zip
    print(f"Extracting {filename}...")
    with zipfile.ZipFile(filepath, "r") as zip_ref:
        zip_ref.extractall(raw_dir)
    print(f"Finished extracting {filename}\n")

print("All files downloaded and extracted to data/raw")