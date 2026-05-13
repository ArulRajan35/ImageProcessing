import argparse
import hashlib
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import requests
from PIL import Image
from tqdm import tqdm


DEFAULT_QUERIES = ["road", "highway", "street", "driving", "urban road", "car dashboard"]


def parse_args():
    parser = argparse.ArgumentParser(description="Download road/driving images from Pixabay.")
    parser.add_argument("--api-key", default=os.getenv("PIXABAY_API_KEY"), help="Pixabay API key")
    parser.add_argument("--output-dir", default="dataset/raw", help="Directory to save images")
    parser.add_argument("--target-count", type=int, default=1000, help="Number of images to download")
    parser.add_argument("--per-page", type=int, default=200, help="Pixabay max is 200")
    parser.add_argument("--max-pages", type=int, default=25, help="Maximum pages per query")
    parser.add_argument("--min-width", type=int, default=640, help="Minimum image width filter")
    parser.add_argument("--min-height", type=int, default=360, help="Minimum image height filter")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout seconds")
    parser.add_argument("--retries", type=int, default=3, help="Retry attempts per image")
    return parser.parse_args()


def pixabay_search(
    api_key: str,
    query: str,
    page: int,
    per_page: int,
    min_width: int,
    min_height: int,
    timeout: int,
) -> List[Dict]:
    params = {
        "key": api_key,
        "q": query,
        "image_type": "photo",
        "orientation": "horizontal",
        "safesearch": "true",
        "per_page": per_page,
        "page": page,
        "min_width": min_width,
        "min_height": min_height,
    }
    resp = requests.get("https://pixabay.com/api/", params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("hits", [])


def download_file(url: str, timeout: int, retries: int) -> Optional[bytes]:
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.content
        except Exception:
            if attempt == retries:
                return None
            time.sleep(1.2 * attempt)
    return None


def content_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def is_valid_image(content: bytes) -> bool:
    try:
        from io import BytesIO

        with Image.open(BytesIO(content)) as img:
            img.verify()
        return True
    except Exception:
        return False


def save_jpeg(content: bytes, out_path: Path) -> bool:
    try:
        from io import BytesIO

        with Image.open(BytesIO(content)).convert("RGB") as img:
            img.save(out_path, format="JPEG", quality=95)
        return True
    except Exception:
        return False


def main():
    args = parse_args()
    if not args.api_key:
        raise ValueError("PIXABAY_API_KEY is missing. Pass --api-key or set environment variable.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_files = sorted(output_dir.glob("img_*.jpg"))
    next_idx = len(existing_files) + 1
    downloaded_hashes: Set[str] = set()

    print(f"[INFO] Output directory: {output_dir.resolve()}")
    print(f"[INFO] Existing images: {len(existing_files)}")
    print(f"[INFO] Target total images: {args.target_count}")

    progress = tqdm(total=max(0, args.target_count - len(existing_files)), desc="Downloading")

    for query in DEFAULT_QUERIES:
        if next_idx > args.target_count:
            break

        for page in range(1, args.max_pages + 1):
            if next_idx > args.target_count:
                break

            try:
                hits = pixabay_search(
                    api_key=args.api_key,
                    query=query,
                    page=page,
                    per_page=min(args.per_page, 200),
                    min_width=args.min_width,
                    min_height=args.min_height,
                    timeout=args.timeout,
                )
            except Exception as exc:
                print(f"[WARN] Failed search query='{query}' page={page}: {exc}")
                time.sleep(1)
                continue

            if not hits:
                break

            for hit in hits:
                if next_idx > args.target_count:
                    break

                image_url = hit.get("largeImageURL") or hit.get("webformatURL")
                if not image_url:
                    continue

                content = download_file(image_url, timeout=args.timeout, retries=args.retries)
                if not content or not is_valid_image(content):
                    continue

                h = content_hash(content)
                if h in downloaded_hashes:
                    continue

                out_path = output_dir / f"img_{next_idx:04d}.jpg"
                if not save_jpeg(content, out_path):
                    continue

                downloaded_hashes.add(h)
                next_idx += 1
                progress.update(1)

    progress.close()
    total = len(list(output_dir.glob("img_*.jpg")))
    print(f"[DONE] Total images in dataset: {total}")


if __name__ == "__main__":
    main()
