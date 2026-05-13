import argparse
import json
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    import albumentations as A
except Exception:
    A = None


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess road image dataset.")
    parser.add_argument("--raw-dir", default="dataset/raw", help="Raw images path")
    parser.add_argument("--processed-dir", default="dataset/processed", help="Processed image path")
    parser.add_argument("--train-dir", default="dataset/train", help="Train split path")
    parser.add_argument("--val-dir", default="dataset/val", help="Validation split path")
    parser.add_argument("--test-dir", default="dataset/test", help="Test split path")
    parser.add_argument("--image-size", type=int, default=384, help="Output square image size")
    parser.add_argument("--augment", action="store_true", help="Enable augmentation for processed copy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def build_augmentor():
    if A is None:
        return None
    return A.Compose(
        [
            A.RandomBrightnessContrast(p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.HorizontalFlip(p=0.5),
            A.RandomResizedCrop(size=(384, 384), scale=(0.85, 1.0), ratio=(0.9, 1.1), p=0.35),
        ]
    )


def sample_camera_calibration() -> Tuple[np.ndarray, np.ndarray]:
    # Example intrinsic matrix for a mobile/dashcam-like setup.
    camera_matrix = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    # Simulated radial and tangential distortion values.
    distortion_coeffs = np.array([-0.22, 0.09, 0.0, 0.0, 0.0], dtype=np.float32)
    return camera_matrix, distortion_coeffs


def undistort_image(image: np.ndarray, camera_matrix: np.ndarray, distortion_coeffs: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs, (w, h), 1, (w, h))
    return cv2.undistort(image, camera_matrix, distortion_coeffs, None, new_camera_matrix)


def normalize_image(image: np.ndarray) -> np.ndarray:
    image_float = image.astype(np.float32) / 255.0
    return np.clip(image_float, 0.0, 1.0)


def preprocess_image(image_path: Path, image_size: int, augmentor=None) -> Tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    cam_mtx, dist = sample_camera_calibration()
    image = undistort_image(image, cam_mtx, dist)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)

    if augmentor is not None:
        transformed = augmentor(image=image)
        image = transformed["image"]

    norm = normalize_image(image)
    image_uint8 = (norm * 255).astype(np.uint8)
    return image_uint8, norm


def split_paths(paths: List[Path], seed: int):
    train_paths, rem_paths = train_test_split(paths, test_size=0.30, random_state=seed, shuffle=True)
    val_paths, test_paths = train_test_split(rem_paths, test_size=0.50, random_state=seed, shuffle=True)
    return train_paths, val_paths, test_paths


def copy_to_split(paths: List[Path], split_dir: Path):
    ensure_dirs(split_dir)
    for idx, img_path in enumerate(paths, start=1):
        dst = split_dir / f"{split_dir.name}_{idx:04d}.jpg"
        shutil.copy2(img_path, dst)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    test_dir = Path(args.test_dir)

    ensure_dirs(processed_dir, train_dir, val_dir, test_dir)

    image_paths = sorted(list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.png")) + list(raw_dir.glob("*.jpeg")))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {raw_dir.resolve()}")

    augmentor = build_augmentor() if args.augment else None
    stats = {"processed_count": 0, "failed_count": 0, "pixel_mean": [], "pixel_std": []}

    for i, path in enumerate(tqdm(image_paths, desc="Preprocessing"), start=1):
        try:
            image_uint8, image_norm = preprocess_image(path, args.image_size, augmentor=augmentor)
            out_path = processed_dir / f"proc_{i:04d}.jpg"
            cv2.imwrite(str(out_path), image_uint8)

            stats["processed_count"] += 1
            stats["pixel_mean"].append(float(np.mean(image_norm)))
            stats["pixel_std"].append(float(np.std(image_norm)))
        except Exception as exc:
            stats["failed_count"] += 1
            print(f"[WARN] Failed {path.name}: {exc}")

    processed_paths = sorted(processed_dir.glob("*.jpg"))
    train_paths, val_paths, test_paths = split_paths(processed_paths, seed=args.seed)

    copy_to_split(train_paths, train_dir)
    copy_to_split(val_paths, val_dir)
    copy_to_split(test_paths, test_dir)

    summary = {
        "processed_count": stats["processed_count"],
        "failed_count": stats["failed_count"],
        "train_count": len(train_paths),
        "val_count": len(val_paths),
        "test_count": len(test_paths),
        "dataset_mean": float(np.mean(stats["pixel_mean"])) if stats["pixel_mean"] else 0.0,
        "dataset_std": float(np.mean(stats["pixel_std"])) if stats["pixel_std"] else 0.0,
        "image_size": args.image_size,
        "augmentation_enabled": bool(args.augment),
    }

    with open(processed_dir / "preprocess_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[DONE] Preprocessing complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
