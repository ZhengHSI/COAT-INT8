import os
import subprocess

datasets = {
    "perplexity/v3_small_gptneox20b/c4_en/val": [
        "https://olmo-data.org/eval-data/perplexity/v3_small_gptneox20b/c4_en/val/part-0-00000.npy"
    ],
    "perplexity/v3_small_gptneox20b/dolma_books/val": [
        "https://olmo-data.org/eval-data/perplexity/v3_small_gptneox20b/dolma_books/val/part-0-00000.npy"
    ],
    "perplexity/v3_small_gptneox20b/dolma_common-crawl/val": [
        "https://olmo-data.org/eval-data/perplexity/v3_small_gptneox20b/dolma_common-crawl/val/part-0-00000.npy"
    ],
    "perplexity/v3_small_gptneox20b/dolma_pes2o/val": [
        "https://olmo-data.org/eval-data/perplexity/v3_small_gptneox20b/dolma_pes2o/val/part-0-00000.npy"
    ],
    "perplexity/v3_small_gptneox20b/dolma_reddit/val": [
        "https://olmo-data.org/eval-data/perplexity/v3_small_gptneox20b/dolma_reddit/val/part-0-00000.npy"
    ],
    "perplexity/v3_small_gptneox20b/dolma_stack/val": [
        "https://olmo-data.org/eval-data/perplexity/v3_small_gptneox20b/dolma_stack/val/part-0-00000.npy"
    ],
    "perplexity/v3_small_gptneox20b/dolma_wiki/val": [
        "https://olmo-data.org/eval-data/perplexity/v3_small_gptneox20b/dolma_wiki/val/part-0-00000.npy"
    ],
    "perplexity/v3_small_gptneox20b/ice/val": [
        "https://olmo-data.org/eval-data/perplexity/v3_small_gptneox20b/ice/val/part-0-00000.npy"
    ],
    "perplexity/v3_small_gptneox20b/m2d2_s2orc/val": [
        "https://olmo-data.org/eval-data/perplexity/v3_small_gptneox20b/m2d2_s2orc/val/part-0-00000.npy"
    ],
    "perplexity/v3_small_gptneox20b/pile/val": [
        "https://olmo-data.org/eval-data/perplexity/v3_small_gptneox20b/pile/val/part-0-00000.npy"
    ],
    "perplexity/v3_small_gptneox20b/wikitext_103/val": [
        "https://olmo-data.org/eval-data/perplexity/v3_small_gptneox20b/wikitext_103/val/part-0-00000.npy"
    ],
    "perplexity/v2_small_gptneox20b/4chan": [
        "https://olmo-data.org/eval-data/perplexity/v2_small_gptneox20b/4chan/val.npy"
    ],
    "perplexity/v2_small_gptneox20b/c4_100_domains": [
        "https://olmo-data.org/eval-data/perplexity/v2_small_gptneox20b/c4_100_domains/val.npy"
    ],
    "perplexity/v2_small_gptneox20b/c4_en": [
        "https://olmo-data.org/eval-data/perplexity/v2_small_gptneox20b/c4_en/val.npy"
    ],
    "perplexity/v2_small_gptneox20b/gab": [
        "https://olmo-data.org/eval-data/perplexity/v2_small_gptneox20b/gab/val.npy"
    ],
    "perplexity/v2_small_gptneox20b/ice": [
        "https://olmo-data.org/eval-data/perplexity/v2_small_gptneox20b/ice/val.npy"
    ],
    "perplexity/v2_small_gptneox20b/m2d2_s2orc": [
        "https://olmo-data.org/eval-data/perplexity/v2_small_gptneox20b/m2d2_s2orc/val.npy"
    ],
    "perplexity/v2_small_gptneox20b/m2d2_wiki": [
        "https://olmo-data.org/eval-data/perplexity/v2_small_gptneox20b/m2d2_wiki/val.npy"
    ],
    "perplexity/v2_small_gptneox20b/manosphere": [
        "https://olmo-data.org/eval-data/perplexity/v2_small_gptneox20b/manosphere/val.npy"
    ],
    "perplexity/v2_small_gptneox20b/mc4_en": [
        "https://olmo-data.org/eval-data/perplexity/v2_small_gptneox20b/mc4_en/val.npy"
    ],
    "perplexity/v2_small_gptneox20b/pile": [
        "https://olmo-data.org/eval-data/perplexity/v2_small_gptneox20b/pile/val.npy"
    ],
    "perplexity/v2_small_gptneox20b/ptb": [
        "https://olmo-data.org/eval-data/perplexity/v2_small_gptneox20b/ptb/val.npy"
    ],
    "perplexity/v2_small_gptneox20b/twitterAEE": [
        "https://olmo-data.org/eval-data/perplexity/v2_small_gptneox20b/twitterAEE/val.npy"
    ],
    "perplexity/v2_small_gptneox20b/wikitext_103": [
        "https://olmo-data.org/eval-data/perplexity/v2_small_gptneox20b/wikitext_103/val.npy"
    ]
}

base_dir = "datasets/eval-data"

for sub_dir, urls in datasets.items():
    target_dir = os.path.join(base_dir, sub_dir)
    os.makedirs(target_dir, exist_ok=True)
    
    for url in urls:
        filename = url.split("/")[-1]
        output_path = os.path.join(target_dir, filename)
        command = ["wget", url, "-O", output_path]
        try:
            print(f"Downloading {filename} to {target_dir}...")
            subprocess.run(command, check=True)
            print(f"Downloaded {filename}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to download {url}. Error: {e}")