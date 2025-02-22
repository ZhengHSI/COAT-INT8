import os
import subprocess
    
# 定义所有文件的 URL 列表
urls = [
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-000-00000.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-000-00001.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-001-00000.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-001-00001.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-002-00000.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-002-00001.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-003-00000.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-003-00001.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-004-00000.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-004-00001.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-005-00000.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-005-00001.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-006-00000.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-006-00001.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-006-00002.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-007-00000.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-007-00001.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-008-00000.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-008-00001.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-008-00002.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-009-00000.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-009-00001.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-010-00000.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-010-00001.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-010-00002.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-011-00000.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-011-00001.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-012-00000.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-012-00001.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-013-00000.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-013-00001.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-013-00002.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-014-00000.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-014-00001.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-014-00002.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-015-00000.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-015-00001.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-016-00000.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-016-00001.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-017-00000.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-017-00001.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-018-00000.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-018-00001.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-019-00000.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-019-00001.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-020-00000.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-020-00001.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-021-00000.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-021-00001.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-022-00000.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-022-00001.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-023-00000.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-023-00001.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-024-00000.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-024-00001.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-025-00000.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-025-00001.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-025-00002.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-026-00000.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-026-00001.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-027-00000.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-027-00001.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-027-00002.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-028-00000.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-028-00001.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-028-00002.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-029-00000.npy",
    "https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-029-00001.npy",
]

target_dir = "datasets/dolma/v1_5"

os.makedirs(target_dir, exist_ok=True)

for url in urls:
    filename = url.split("/")[-1]
    output_path = os.path.join(target_dir, filename)
    command = ["wget", url, "-O", output_path]
    try:
        print(f"Downloading {filename}...")
        subprocess.run(command, check=True)
        print(f"Downloaded {filename} to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {url}. Error: {e}")