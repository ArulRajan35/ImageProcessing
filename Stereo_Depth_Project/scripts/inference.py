import argparse
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

from midas_pipeline import colorize_depth, get_device, load_midas_model, normalize_depth, predict_depth_map
from train import DepthRefiner


def parse_args():
    parser = argparse.ArgumentParser(description="Run MiDaS inference on single image or folder.")
    parser.add_argument("--input", required=True, help="Input image path or folder path")
    parser.add_argument("--output-dir", default="outputs/depth_maps", help="Output directory")
    parser.add_argument("--model-key", default="midas_v31_hybrid", help="MiDaS model key")
    parser.add_argument("--colormap", default="magma", help="Matplotlib colormap")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--use-refiner", action="store_true", help="Use trained refiner model")
    parser.add_argument("--refiner-path", default="model/best_model/depth_refiner_best.pt", help="Refiner checkpoint")
    return parser.parse_args()


def list_inputs(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        imgs.extend(path.glob(ext))
    return sorted(imgs)


def load_refiner(refiner_path: str, device: torch.device) -> Optional[DepthRefiner]:
    ckpt_path = Path(refiner_path)
    if not ckpt_path.exists():
        print(f"[WARN] Refiner checkpoint not found: {refiner_path}")
        return None
    model = DepthRefiner().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def save_comparison(image_bgr: np.ndarray, depth_color_bgr: np.ndarray, out_path: Path):
    comparison = np.hstack([image_bgr, depth_color_bgr])
    cv2.imwrite(str(out_path), comparison)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, transform, device = load_midas_model(model_key=args.model_key, prefer_gpu=True)
    refiner = load_refiner(args.refiner_path, device) if args.use_refiner else None

    input_paths = list_inputs(Path(args.input))
    if not input_paths:
        raise FileNotFoundError(f"No valid images found at: {args.input}")

    for img_path in tqdm(input_paths, desc="Inference"):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"[WARN] Skipping unreadable image: {img_path}")
            continue
        image_resized = cv2.resize(image_bgr, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)

        depth = predict_depth_map(model, transform, device, image_resized)
        depth = normalize_depth(depth)

        if refiner is not None:
            rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            inp = np.concatenate([rgb, depth[..., None]], axis=-1).transpose(2, 0, 1)
            inp_t = torch.from_numpy(inp).unsqueeze(0).float().to(device)
            with torch.no_grad():
                depth = torch.sigmoid(refiner(inp_t)).squeeze().cpu().numpy()
            depth = normalize_depth(depth)

        depth_color = colorize_depth(depth, cmap_name=args.colormap)
        gray_out = out_dir / f"{img_path.stem}_depth_gray.png"
        color_out = out_dir / f"{img_path.stem}_depth_color.png"
        compare_out = out_dir / f"{img_path.stem}_comparison.png"

        cv2.imwrite(str(gray_out), (depth * 255).astype(np.uint8))
        cv2.imwrite(str(color_out), depth_color)
        save_comparison(image_resized, depth_color, compare_out)

    print(f"[DONE] Saved outputs to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
