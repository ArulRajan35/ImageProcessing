import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

from midas_pipeline import colorize_depth, load_midas_model, normalize_depth, predict_depth_map  # noqa: E402
from train import DepthRefiner  # noqa: E402


st.set_page_config(page_title="Stereo Vision Depth Estimation", page_icon=":camera:", layout="wide")


@st.cache_resource
def cached_midas(model_key: str):
    return load_midas_model(model_key=model_key, prefer_gpu=True)


@st.cache_resource
def cached_refiner(refiner_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthRefiner().to(device)
    ckpt = torch.load(refiner_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, device


def pil_to_bgr(image: Image.Image, target_size: int) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.resize(bgr, (target_size, target_size), interpolation=cv2.INTER_AREA)


def apply_confidence_filter(depth_norm: np.ndarray, threshold: float) -> np.ndarray:
    depth_filtered = depth_norm.copy()
    depth_filtered[depth_filtered < threshold] = 0.0
    return depth_filtered


def render_header():
    st.title("Stereo Vision Depth Estimation")
    st.caption(
        "MiDaS v3.1 depth inference with optional refinement model. "
        "Designed for practical road-scene demo workflows."
    )


def render_road_scene_example_panel():
    """Teaches how to read depth like the car / building / road example (relative depth, not object labels)."""
    with st.expander("Real example: what the depth map is showing (road scene)", expanded=False):
        st.markdown(
            """
**Input (typical road photo)**

- A **nearby car**
- A **distant building**
- **Road** stretching ahead

**How to read the output (intuition)**

| Region | What you usually see in the depth map |
|--------|----------------------------------------|
| **Car (close)** | Pixels on the car differ from the background—typically read as **closer** than the building (warmer/cooler depends on the colormap; compare regions side by side). |
| **Building (far)** | Usually **farther** than the car; smoother or more uniform if the facade is flat. |
| **Road** | Often a **smooth change** along the lane from **near the camera (bottom of image)** toward the **horizon**—a gradual depth ramp, not a single “object class.” |

**Important:** The model predicts **depth per pixel**, not “car” vs “building” labels. You interpret regions by matching the depth colors to objects in the original image. Values are **relative** (good near/far ordering), not guaranteed **meters** unless you add metric calibration.
            """
        )


def vertical_center_profile(depth_norm: np.ndarray) -> pd.DataFrame:
    """One column of normalized depth; index = row from bottom (0 = bottom of image, typical near road)."""
    h, w = depth_norm.shape
    col = depth_norm[:, w // 2].astype(float)
    col_bottom_first = col[::-1]
    return pd.DataFrame({"relative_depth": col_bottom_first}, index=np.arange(h, dtype=int))


def horizontal_band_means(depth_norm: np.ndarray) -> pd.DataFrame:
    """Rough bands: bottom / middle / top of frame (typical dashcam: bottom ≈ road near camera)."""
    h, _ = depth_norm.shape
    t0, t1 = 0, h // 3
    m0, m1 = h // 3, 2 * h // 3
    b0, b1 = 2 * h // 3, h
    bands = [
        ("Top third (often sky / distant)", depth_norm[t0:t1]),
        ("Middle third", depth_norm[m0:m1]),
        ("Bottom third (often road / near)", depth_norm[b0:b1]),
    ]
    rows = []
    for label, patch in bands:
        rows.append({"Image band": label, "Mean relative depth": float(np.mean(patch))})
    return pd.DataFrame(rows)


def main():
    render_header()

    with st.sidebar:
        st.header("Controls")
        model_key = st.selectbox(
            "Model Selection",
            options=["midas_v31_hybrid", "midas_v31_large", "midas_small"],
            index=0,
        )
        colormap = st.selectbox("Depth Colormap", options=["magma", "viridis", "plasma", "inferno"], index=0)
        image_size = st.slider("Input Resolution", min_value=256, max_value=640, value=384, step=32)
        confidence_threshold = st.slider("Confidence / Visibility Threshold", 0.0, 0.9, 0.05, 0.01)
        use_refiner = st.checkbox("Use trained refinement model", value=False)
        refiner_path = st.text_input("Refiner checkpoint path", str(ROOT / "model" / "best_model" / "depth_refiner_best.pt"))
        render_road_scene_example_panel()

    uploaded = st.file_uploader("Upload road/street image", type=["jpg", "jpeg", "png"])
    camera_capture = st.camera_input("Or capture from webcam (optional)")

    image_source = camera_capture if camera_capture is not None else uploaded
    if image_source is None:
        st.info("Upload or capture an image to generate depth.")
        return

    try:
        pil_image = Image.open(image_source)
    except Exception as exc:
        st.error(f"Could not decode image: {exc}")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(pil_image, use_container_width=True)

    with st.spinner("Running depth estimation..."):
        try:
            model, transform, device = cached_midas(model_key)
            image_bgr = pil_to_bgr(pil_image, image_size)
            depth = predict_depth_map(model, transform, device, image_bgr)
            depth = normalize_depth(depth)

            if use_refiner:
                if not Path(refiner_path).exists():
                    st.warning("Refiner checkpoint not found. Continuing with MiDaS only.")
                else:
                    refiner, refiner_device = cached_refiner(refiner_path)
                    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    inp = np.concatenate([rgb, depth[..., None]], axis=-1).transpose(2, 0, 1)
                    inp_t = torch.from_numpy(inp).unsqueeze(0).float().to(refiner_device)
                    with torch.no_grad():
                        depth = torch.sigmoid(refiner(inp_t)).squeeze().cpu().numpy()
                    depth = normalize_depth(depth)

            depth = apply_confidence_filter(depth, confidence_threshold)
            depth_color = colorize_depth(depth, cmap_name=colormap)
            depth_color_rgb = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
        except Exception as exc:
            st.error(f"Inference failed: {exc}")
            return

    with col2:
        st.subheader("Predicted Depth Map")
        st.image(depth_color_rgb, use_container_width=True)

    st.subheader("Road-scene interpretation (from your depth map)")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Vertical profile (image center column)** — road scenes often show a **gradual** change from bottom toward the horizon.")
        prof = vertical_center_profile(depth)
        st.line_chart(prof)
        st.caption("X: pixel row from bottom · Y: normalized relative depth (after your filters).")
    with c4:
        st.markdown("**Mean depth by vertical band** — compare bottom (often near) vs top (often far).")
        st.dataframe(horizontal_band_means(depth), hide_index=True, use_container_width=True)

    with st.expander("Quick reference: car vs building vs road", expanded=True):
        st.markdown(
            """
- **Nearby car** → usually **closer** than distant structures (check the color on the car vs the building).
- **Distant building** → usually **farther** than the car.
- **Road ahead** → often a **smooth depth gradient** along the lane toward the horizon.
            """
        )

    depth_pil = Image.fromarray(depth_color_rgb)
    buffer = io.BytesIO()
    depth_pil.save(buffer, format="PNG")
    st.download_button(
        label="Download Depth Map",
        data=buffer.getvalue(),
        file_name="depth_map.png",
        mime="image/png",
    )

    st.success("Depth map generated successfully.")


if __name__ == "__main__":
    main()
