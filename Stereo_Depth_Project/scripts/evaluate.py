import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from midas_pipeline import get_device, load_midas_model, normalize_depth, predict_depth_map
from train import DepthRefiner


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate depth estimation performance.")
    parser.add_argument("--test-dir", default="dataset/test", help="Test images directory")
    parser.add_argument("--best-model", default="model/best_model/depth_refiner_best.pt", help="Best refiner checkpoint")
    parser.add_argument("--output-metrics", default="outputs/metrics/evaluation_metrics.json", help="Metrics output path")
    parser.add_argument("--viz-dir", default="outputs/metrics/visualizations", help="Comparison visualizations folder")
    parser.add_argument("--model-key", default="midas_v31_hybrid")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--max-viz", type=int, default=15)
    return parser.parse_args()


def rel_error(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs(pred - target) / (np.abs(target) + eps)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def list_images(folder: Path):
    return sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png")))


def main():
    args = parse_args()
    test_paths = list_images(Path(args.test_dir))
    if not test_paths:
        raise FileNotFoundError("No test images found. Run preprocess.py first.")

    Path(args.viz_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_metrics).parent.mkdir(parents=True, exist_ok=True)

    teacher_model, teacher_transform, teacher_device = load_midas_model(args.model_key, prefer_gpu=True)
    device = get_device(prefer_gpu=True)

    refiner = DepthRefiner().to(device)
    ckpt = torch.load(args.best_model, map_location=device)
    refiner.load_state_dict(ckpt["model_state"])
    refiner.eval()

    rmse_vals, mae_vals, rel_vals = [], [], []

    for i, img_path in enumerate(tqdm(test_paths, desc="Evaluating"), start=1):
        image_bgr = cv2.imread(str(img_path))
        image_bgr = cv2.resize(image_bgr, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        teacher_depth = predict_depth_map(teacher_model, teacher_transform, teacher_device, image_bgr)
        teacher_depth = normalize_depth(teacher_depth)

        inp = np.concatenate([image_rgb, teacher_depth[..., None]], axis=-1).transpose(2, 0, 1)
        inp_t = torch.from_numpy(inp).unsqueeze(0).float().to(device)
        with torch.no_grad():
            pred_refined = torch.sigmoid(refiner(inp_t)).squeeze().cpu().numpy()

        rmse_vals.append(rmse(pred_refined, teacher_depth))
        mae_vals.append(mae(pred_refined, teacher_depth))
        rel_vals.append(rel_error(pred_refined, teacher_depth))

        if i <= args.max_viz:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            axes[0].set_title("Input")
            axes[1].imshow(teacher_depth, cmap="magma")
            axes[1].set_title("Pseudo Depth (Teacher)")
            axes[2].imshow(pred_refined, cmap="magma")
            axes[2].set_title("Refined Depth (Student)")
            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(Path(args.viz_dir) / f"comparison_{i:03d}.png", dpi=120)
            plt.close(fig)

    metrics = {
        "num_samples": len(test_paths),
        "rmse_mean": float(np.mean(rmse_vals)),
        "mae_mean": float(np.mean(mae_vals)),
        "rel_error_mean": float(np.mean(rel_vals)),
        "note": "Metrics are against MiDaS pseudo-labels (teacher), not absolute LiDAR ground truth.",
    }

    with open(args.output_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"[DONE] Metrics saved to {args.output_metrics}")


if __name__ == "__main__":
    main()
