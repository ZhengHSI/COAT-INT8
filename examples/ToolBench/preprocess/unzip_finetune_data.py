import zipfile
import os

# Define paths for each zip file and their extraction directory
zip_files = [
    'data.zip'
]
extract_to = './'

# Ensure the destination directory exists
os.makedirs(extract_to, exist_ok=True)

# Loop through each zip file and extract it to the specified directory
for zip_path in zip_files:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Unzipped {zip_path} to {extract_to}")
