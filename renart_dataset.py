# usr/bin/env python3
"""
renart_dataset.py Dual dataset pipeline for Renart
Sources:
  1. WikiArt via Hugging Face (huggan/wikiart) ~5,000 Renaissance images
  2. National Gallery of Art (NGA) open data CSV + image API ~2,000 Renaissance images

Usage:
  pip install datasets huggingface_hub pillow requests pandas tqdm
  python3 renart_dataset.py

Output structure:
  data/renaissance/
    train/
      Leonardo_da_Vinci/
      Raphael/
      ...
    val/
      Leonardo_da_Vinci/
      Raphael/
      ...
    dataset_stats.json   <- summary of what was downloaded
"""

import os
import json
import time
import hashlib
import requests
import pandas as pd
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "output_dir": "./data/renaissance",
    "train_dir":  "./data/renaissance/train",
    "val_dir":    "./data/renaissance/val",
    "image_size": (224, 224),
    "val_split":  0.2,          # 80% train / 20% val
    "max_per_artist": 300,      # Cap per artist to keep dataset balanced
    "min_per_artist": 20,       # Skip artists with fewer than this many images

    # NGA open data CSV URL (updated daily by NGA)
    "nga_objects_csv": "https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/main/data/objects.csv",
    "nga_image_base":  "https://api.nga.gov/art/tms/objects/{object_id}/images",

    # Target Renaissance artists normalized for matching across both datasets
    "target_artists": [
        "Leonardo da Vinci",
        "Michelangelo",
        "Raphael",
        "Titian",
        "Sandro Botticelli",
        "Caravaggio",
        "Tintoretto",
        "Paolo Veronese",
        "Giorgione",
        "Fra Angelico",
        "Masaccio",
        "Andrea Mantegna",
        "Giovanni Bellini",
        "Filippino Lippi",
        "Luca Signorelli",
        "Pietro Perugino",
        "Domenico Ghirlandaio",
        "Jacopo Pontormo",
        "Piero della Francesca",
        "Lorenzo Lotto",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def normalize_artist_name(name: str) -> str:
    """Standardize artist name formatting for cross-dataset matching."""
    return name.strip().title()

def safe_folder_name(name: str) -> str:
    """Convert artist name to filesystem-safe folder name."""
    return name.replace(" ", "_").replace("/", "-").replace("'", "")

def get_split(count: int, val_split: float) -> str:
    """Deterministically assign train/val based on count."""
    return "val" if (count % 10) >= int((1 - val_split) * 10) else "train"

def save_image(img: Image.Image, path: str, size: tuple) -> bool:
    """Resize and save image as JPEG. Returns True on success."""
    try:
        img = img.convert("RGB")
        img = img.resize(size, Image.LANCZOS)
        img.save(path, "JPEG", quality=90)
        return True
    except Exception as e:
        print(f"    Could not save image: {e}")
        return False

def deduplicate(path: str, seen_hashes: set) -> bool:
    """Return True if image is a duplicate (hash already seen)."""
    with open(path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    if file_hash in seen_hashes:
        os.remove(path)
        return True
    seen_hashes.add(file_hash)
    return False

# ─────────────────────────────────────────────────────────────────────────────
# Source 1: WikiArt via Hugging Face
# Dataset: huggan/wikiart (~80k images, 129 artists, multiple styles)
# We filter to Renaissance artists only.
# ─────────────────────────────────────────────────────────────────────────────
def download_wikiart(config: dict, counts: dict, seen_hashes: dict) -> dict:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: Install datasets library: pip install datasets")
        return counts

    print("\n━━━ Source 1: WikiArt (Hugging Face) ━━━")
    print("Loading huggan/wikiart dataset in streaming mode...")
    print("(First run downloads ~4GB index — subsequent runs are instant)\n")

    try:
        dataset = load_dataset("huggan/wikiart", split="train", streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load WikiArt: {e}")
        print("Try: pip install datasets --upgrade")
        return counts

    # Build lookup for fast name matching
    target_set = {normalize_artist_name(a) for a in config["target_artists"]}

    skipped_artist = 0
    skipped_max = 0

    for item in tqdm(dataset, desc="WikiArt", unit="img"):
        # WikiArt uses numeric artist IDs — the dataset has an 'artist' field
        # that maps to an artist name string
        raw_artist = str(item.get("artist", ""))
        artist = normalize_artist_name(raw_artist)

        if artist not in target_set:
            skipped_artist += 1
            continue

        if counts.get(artist, 0) >= config["max_per_artist"]:
            skipped_max += 1
            # Check if all artists are at max
            if all(counts.get(a, 0) >= config["max_per_artist"]
                   for a in target_set):
                print("\nAll artists at max count — stopping WikiArt download.")
                break
            continue

        split = get_split(counts.get(artist, 0), config["val_split"])
        folder = os.path.join(config[f"{split}_dir"], safe_folder_name(artist))
        os.makedirs(folder, exist_ok=True)

        img_idx = counts.get(artist, 0)
        img_path = os.path.join(folder, f"wikiart_{img_idx:04d}.jpg")

        pil_image = item.get("image")
        if pil_image is None:
            continue

        if save_image(pil_image, img_path, config["image_size"]):
            if not deduplicate(img_path, seen_hashes.setdefault(artist, set())):
                counts[artist] = counts.get(artist, 0) + 1

    print(f"\nWikiArt complete.")
    print(f"  Skipped (wrong artist): {skipped_artist}")
    print(f"  Skipped (at max):       {skipped_max}")
    return counts

# ─────────────────────────────────────────────────────────────────────────────
# Source 2: National Gallery of Art (NGA)
# Uses their open CSV from GitHub + image API endpoint
# CSV has ~130k artworks; we filter by artist and period, then fetch images.
# ─────────────────────────────────────────────────────────────────────────────
def download_nga(config: dict, counts: dict, seen_hashes: dict) -> dict:
    print("\n━━━ Source 2: National Gallery of Art (NGA) ━━━")
    print("Downloading NGA open data CSV...")

    # Download the objects CSV
    try:
        response = requests.get(config["nga_objects_csv"], timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch NGA CSV: {e}")
        return counts

    # Parse CSV
    from io import StringIO
    df = pd.read_csv(StringIO(response.text), low_memory=False)
    print(f"NGA CSV loaded: {len(df)} total records")

    # Filter columns we need
    needed = ["objectid", "attribution", "title", "medium", "classification", "visualbrowsertimeperiod"]
    available = [c for c in needed if c in df.columns]
    df = df[available].dropna(subset=["objectid", "attribution"])

    # Filter to paintings only (not sculptures, prints, etc.)
    if "classification" in df.columns:
        df = df[df["classification"].str.contains("Painting", case=False, na=False)]

    # Filter to Renaissance time period
    if "visualbrowsertimeperiod" in df.columns:
        df = df[df["visualbrowsertimeperiod"].str.contains(
            "1400|1500|1600|Renaissance|Early Modern", case=False, na=False
        )]

    print(f"After filtering to Renaissance paintings: {len(df)} records")

    # Build artist→objectids mapping
    target_set = {normalize_artist_name(a) for a in config["target_artists"]}
    artist_objects: dict = {}

    for _, row in df.iterrows():
        attribution = str(row.get("attribution", ""))
        # NGA attribution often looks like "Leonardo da Vinci\nItalian, 1452–1519"
        # Take just the first line
        artist_line = attribution.split("\n")[0].strip()
        artist = normalize_artist_name(artist_line)

        # Fuzzy match — check if any target artist name is contained
        matched = None
        for target in target_set:
            if target.lower() in artist.lower() or artist.lower() in target.lower():
                matched = target
                break

        if matched:
            if matched not in artist_objects:
                artist_objects[matched] = []
            artist_objects[matched].append(int(row["objectid"]))

    print(f"Matched {len(artist_objects)} artists in NGA dataset")
    for artist, ids in artist_objects.items():
        print(f"  {artist}: {len(ids)} paintings")

    # Download images for each matched artist
    for artist, object_ids in artist_objects.items():
        print(f"\nDownloading NGA images for {artist}...")
        downloaded = 0

        for obj_id in tqdm(object_ids, desc=artist, unit="img"):
            if counts.get(artist, 0) >= config["max_per_artist"]:
                break

            # NGA image API endpoint
            image_url = f"https://api.nga.gov/iiif/{obj_id}/full/!224,224/0/default.jpg"

            try:
                resp = requests.get(image_url, timeout=15)
                if resp.status_code != 200:
                    continue

                img = Image.open(BytesIO(resp.content))

                split = get_split(counts.get(artist, 0), config["val_split"])
                folder = os.path.join(config[f"{split}_dir"], safe_folder_name(artist))
                os.makedirs(folder, exist_ok=True)

                img_idx = counts.get(artist, 0)
                img_path = os.path.join(folder, f"nga_{obj_id}.jpg")

                if save_image(img, img_path, config["image_size"]):
                    if not deduplicate(img_path, seen_hashes.setdefault(artist, set())):
                        counts[artist] = counts.get(artist, 0) + 1
                        downloaded += 1

                # Be polite to NGA servers — don't hammer them
                time.sleep(0.3)

            except Exception:
                continue

        print(f"  Downloaded {downloaded} new images for {artist}")

    return counts

# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  Renart Dataset Pipeline")
    print("  Sources: WikiArt (Hugging Face) + NGA Open Data")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    # Skip if already downloaded
    stats_path = os.path.join(CONFIG["output_dir"], "dataset_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        total = sum(stats.get("counts", {}).values())
        print(f"Dataset already exists ({total} images). Delete {stats_path} to re-download.")
        print(json.dumps(stats["counts"], indent=2))
        return

    # Create output dirs
    os.makedirs(CONFIG["train_dir"], exist_ok=True)
    os.makedirs(CONFIG["val_dir"], exist_ok=True)

    counts = {}         # artist> total image count
    seen_hashes = {}    # artist> set of md5 hashes (deduplication)

    # Download from both sources
    counts = download_wikiart(CONFIG, counts, seen_hashes)
    counts = download_nga(CONFIG, counts, seen_hashes)

    # Filter out artists with too few images
    removed = [a for a, c in counts.items() if c < CONFIG["min_per_artist"]]
    for artist in removed:
        print(f"Removing {artist} — only {counts[artist]} images (minimum is {CONFIG['min_per_artist']})")
        # Remove their folders
        for split in ["train", "val"]:
            folder = os.path.join(CONFIG[f"{split}_dir"], safe_folder_name(artist))
            if os.path.exists(folder):
                import shutil
                shutil.rmtree(folder)
        del counts[artist]

    # Save stats
    total = sum(counts.values())
    stats = {
        "total_images": total,
        "num_artists": len(counts),
        "counts": dict(sorted(counts.items(), key=lambda x: -x[1])),
        "sources": ["WikiArt (Hugging Face)", "National Gallery of Art"],
        "image_size": list(CONFIG["image_size"]),
        "val_split": CONFIG["val_split"],
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Final report
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  Download complete!")
    print(f"  Total images:  {total}")
    print(f"  Artists:       {len(counts)}")
    print(f"  Saved to:      {CONFIG['output_dir']}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("\nPer-artist counts:")
    for artist, count in stats["counts"].items():
        bar = "█" * (count // 10)
        print(f"  {artist:<30} {count:>4}  {bar}")

if __name__ == "__main__":
    main()
