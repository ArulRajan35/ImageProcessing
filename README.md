# Stereo Vision Depth Estimation using Camera Calibration

A beginner-friendly Python project that estimates depth from KITTI left camera images using the MiDaS v3.1 ONNX model.

## Project structure

- `project/`
  - `main.py` : main script to run depth estimation
  - `images/` : place KITTI left camera images here
  - `model/` : place the MiDaS ONNX model here
  - `output/` : generated depth maps are saved here

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- Matplotlib
- ONNX Runtime

## Setup in VS Code

1. Open this folder in VS Code: `e:\stereo estimation ipcv`
2. Install the Python extension if not already installed.
3. Create a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
4. Install the required packages:
   ```powershell
   python -m pip install --upgrade pip
   pip install opencv-python numpy matplotlib onnxruntime
   ```
5. Download the MiDaS v3.1 ONNX model and save it as `project\model\midas_v3_1.onnx`.
6. Download / copy 20–30 KITTI left camera images into `project\images`.

## Running the project

From the VS Code terminal with the virtual environment activated:

```powershell
cd project
python main.py
```

## Notes

- The script applies a sample KITTI camera intrinsic matrix and basic distortion correction.
- It preprocesses each image to `256x256`, runs the MiDaS model, and saves depth maps in `project/output/`.
- The script also shows the original image next to the depth map visualization.

## How to use

- Put your KITTI left camera images in `project/images/`.
- Put `midas_v3_1.onnx` in `project/model/`.
- Run `python main.py`.

## Output

- Depth results are saved as PNG files in `project/output/`.
- Each output file is named like `KITTI_image_depth.png`.
