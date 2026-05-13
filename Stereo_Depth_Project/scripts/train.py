import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from midas_pipeline import get_device, load_midas_model, normalize_depth, predict_depth_map


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DepthRefiner(nn.Module):
    # Lightweight U-Net style refiner over RGB + MiDaS pseudo depth.
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(4, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(64, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


class PseudoDepthDataset(Dataset):
    def __init__(
        self,
        image_paths: List[Path],
        pseudo_dir: Path,
        teacher_model,
        teacher_transform,
        device: torch.device,
        image_size: int = 384,
    ):
        self.image_paths = image_paths
        self.pseudo_dir = pseudo_dir
        self.teacher_model = teacher_model
        self.teacher_transform = teacher_transform
        self.device = device
        self.image_size = image_size
        self.pseudo_dir.mkdir(parents=True, exist_ok=True)

    def _pseudo_path(self, img_path: Path) -> Path:
        return self.pseudo_dir / f"{img_path.stem}.npy"

    def _create_or_load_pseudo(self, img_path: Path, image_bgr: np.ndarray) -> np.ndarray:
        p_path = self._pseudo_path(img_path)
        if p_path.exists():
            return np.load(p_path)

        depth = predict_depth_map(self.teacher_model, self.teacher_transform, self.device, image_bgr)
        depth = normalize_depth(depth)
        np.save(p_path, depth.astype(np.float32))
        return depth

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        image_bgr = cv2.imread(str(img_path))
        image_bgr = cv2.resize(image_bgr, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        pseudo_depth = self._create_or_load_pseudo(img_path, image_bgr)
        pseudo_depth = cv2.resize(pseudo_depth, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)

        x = np.concatenate([image_rgb, pseudo_depth[..., None]], axis=-1).transpose(2, 0, 1)
        y = pseudo_depth[None, ...]

        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


def parse_args():
    parser = argparse.ArgumentParser(description="Train depth refiner using MiDaS pseudo labels.")
    parser.add_argument("--train-dir", default="dataset/train", help="Training images folder")
    parser.add_argument("--val-dir", default="dataset/val", help="Validation images folder")
    parser.add_argument("--pseudo-root", default="dataset/processed/pseudo_labels", help="Pseudo label cache root")
    parser.add_argument("--checkpoint-dir", default="model/checkpoints", help="Checkpoint output directory")
    parser.add_argument("--best-model-dir", default="model/best_model", help="Best model output directory")
    parser.add_argument("--log-dir", default="outputs/logs", help="Training logs output directory")
    parser.add_argument("--model-key", default="midas_v31_hybrid", help="Teacher model key")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=384)
    return parser.parse_args()


def list_images(folder: Path) -> List[Path]:
    return sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png")))


def run_epoch(model, loader, optimizer, criterion, device: torch.device, train: bool):
    model.train(train)
    losses = []
    for x, y in tqdm(loader, desc="Train" if train else "Val", leave=False):
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            pred = model(x)
            pred = torch.sigmoid(pred)
            loss = criterion(pred, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def save_checkpoint(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.best_model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    train_paths = list_images(Path(args.train_dir))
    val_paths = list_images(Path(args.val_dir))
    if not train_paths or not val_paths:
        raise FileNotFoundError("Train/val images missing. Run preprocess.py first.")

    teacher_model, teacher_transform, teacher_device = load_midas_model(model_key=args.model_key, prefer_gpu=True)
    device = get_device(prefer_gpu=True)

    train_ds = PseudoDepthDataset(
        train_paths,
        Path(args.pseudo_root) / "train",
        teacher_model,
        teacher_transform,
        teacher_device,
        args.image_size,
    )
    val_ds = PseudoDepthDataset(
        val_paths,
        Path(args.pseudo_root) / "val",
        teacher_model,
        teacher_transform,
        teacher_device,
        args.image_size,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = DepthRefiner().to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, criterion, device, train=False)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        ckpt_path = Path(args.checkpoint_dir) / f"epoch_{epoch:03d}.pt"
        save_checkpoint(
            {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "val_loss": val_loss},
            ckpt_path,
        )

        print(f"[EPOCH {epoch}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            save_checkpoint(
                {"epoch": epoch, "model_state": model.state_dict(), "val_loss": val_loss},
                Path(args.best_model_dir) / "depth_refiner_best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("[INFO] Early stopping triggered.")
                break

    log_path = Path(args.log_dir) / "training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"best_val_loss": best_val, "history": history}, f, indent=2)

    print(f"[DONE] Training complete. Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    main()
