import os
from pathlib import Path
from typing import Dict, Tuple

import cv2
import matplotlib.cm as cm
import numpy as np
import torch


MODEL_NAME_MAP: Dict[str, str] = {
    "midas_v31_large": "DPT_BEiT_L_512",
    "midas_v31_hybrid": "DPT_Hybrid",
    "midas_small": "MiDaS_small",
}


def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_midas_model(model_key: str = "midas_v31_hybrid", prefer_gpu: bool = True):
    if model_key not in MODEL_NAME_MAP:
        raise ValueError(f"Unsupported model_key='{model_key}'. Supported: {list(MODEL_NAME_MAP)}")

    model_name = MODEL_NAME_MAP[model_key]
    device = get_device(prefer_gpu=prefer_gpu)

    # MiDaS DPT backbones (e.g. BEiT) require timm. trust_repo skips interactive hub prompts (PyTorch 2+).
    def _hub_load(entrypoint: str):
        try:
            return torch.hub.load("intel-isl/MiDaS", entrypoint, trust_repo=True)
        except TypeError:
            return torch.hub.load("intel-isl/MiDaS", entrypoint)

    model = _hub_load(model_name)
    transforms = _hub_load("transforms")
    transform = transforms.small_transform if model_name == "MiDaS_small" else transforms.dpt_transform

    model.to(device).eval()
    return model, transform, device


@torch.no_grad()
def predict_depth_map(model, transform, device: torch.device, image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr is None:
        raise ValueError("Input image is None.")

    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).to(device)

    prediction = model(input_tensor)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth = prediction.cpu().numpy()
    return depth


def normalize_depth(depth_map: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    depth_min = np.min(depth_map)
    depth_max = np.max(depth_map)
    return (depth_map - depth_min) / (depth_max - depth_min + eps)


def colorize_depth(depth_map: np.ndarray, cmap_name: str = "magma") -> np.ndarray:
    depth_norm = normalize_depth(depth_map)
    colormap = cm.get_cmap(cmap_name)
    depth_color = (colormap(depth_norm)[:, :, :3] * 255).astype(np.uint8)
    return cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR)


def save_depth_outputs(
    depth_map: np.ndarray,
    output_base_path: str,
    cmap_name: str = "magma",
) -> Tuple[str, str]:
    output_base = Path(output_base_path)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    depth_norm = normalize_depth(depth_map)
    gray_path = f"{output_base}_gray.png"
    color_path = f"{output_base}_color.png"

    gray_uint8 = (depth_norm * 255).astype(np.uint8)
    cv2.imwrite(gray_path, gray_uint8)

    depth_color = colorize_depth(depth_map, cmap_name=cmap_name)
    cv2.imwrite(color_path, depth_color)

    return gray_path, color_path


def ensure_checkpoint_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
