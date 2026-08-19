"""
train.py
Reproduces the training procedure used to produce models/best_model.pt
(Analog_Addicts submission, KLA Problem Statement, Hackathon 2026).

Usage:
    python train.py --data_dir <path_to_paired_GT_NoisyLR_data> --out_dir ./checkpoints

This matches the actual v4 training run: Adam optimizer (lr=2e-4),
CosineAnnealingLR (T_max=40), 40 epochs, batch_size=16, and the
combined_loss_v4 objective (L1 + 0.1*SSIM + 0.1*VGG16-perceptual +
0.15*Sobel-edge loss). The original notebook also warm-started ~1/58
layers from an earlier checkpoint (best_model_v3.pt) where tensor
shapes happened to match; this script trains from random initialization
instead, which is the standard/reproducible default — the warm-start
transferred a negligible fraction of weights and is not expected to
materially change results.

NOTE: no explicit random seed was set in the original notebook, so
exact bit-for-bit reproduction isn't guaranteed — results should be
very close but may vary slightly run to run.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Model definition — MUST match run.py exactly (base_ch=64, n_res_blocks=8)
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + identity)


class RestorationNet(nn.Module):
    def __init__(self, base_ch=64, n_res_blocks=8):
        super().__init__()
        self.enc1 = ConvBlock(1, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.res_blocks = nn.Sequential(*[ResBlock(base_ch * 2) for _ in range(n_res_blocks)])

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2 + base_ch * 2, base_ch * 2)

        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec2 = ConvBlock(base_ch + base_ch, base_ch)

        self.final_up = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b = self.res_blocks(p2)
        u1 = self.up1(b)
        d1 = self.dec1(torch.cat([u1, e2], dim=1))
        u2 = self.up2(d1)
        d2 = self.dec2(torch.cat([u2, e1], dim=1))
        return self.final_up(d2)


# ---------------------------------------------------------------------------
# Dataset — TODO: adjust file layout to match your actual GT/NoisyLR folders
# ---------------------------------------------------------------------------
class PairedRestorationDataset(Dataset):
    """Expects <data_dir>/GT/*.npy and <data_dir>/NoisyLR/*.npy with matching filenames."""

    def __init__(self, data_dir, augment=True):
        self.gt_dir = os.path.join(data_dir, "GT")
        self.lr_dir = os.path.join(data_dir, "NoisyLR")
        self.filenames = sorted(f for f in os.listdir(self.gt_dir) if f.endswith(".npy"))
        self.augment = augment

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        gt = np.load(os.path.join(self.gt_dir, fname)).astype(np.float32)
        lr = np.load(os.path.join(self.lr_dir, fname)).astype(np.float32)

        if gt.ndim == 3:
            gt = gt.squeeze(-1)
        if lr.ndim == 3:
            lr = lr.squeeze(-1)

        if self.augment:
            if random.random() < 0.5:
                gt, lr = np.fliplr(gt).copy(), np.fliplr(lr).copy()
            if random.random() < 0.5:
                gt, lr = np.flipud(gt).copy(), np.flipud(lr).copy()
            k = random.choice([0, 1, 2, 3])
            gt, lr = np.rot90(gt, k).copy(), np.rot90(lr, k).copy()

        gt_t = torch.from_numpy(gt).unsqueeze(0)
        lr_t = torch.from_numpy(lr).unsqueeze(0)
        return lr_t, gt_t


# ---------------------------------------------------------------------------
# Loss — combined_loss_v4: L1 + 0.1*SSIM + 0.1*VGG16-perceptual + 0.15*Sobel-edge
# (matches the actual loss used to train the submitted checkpoint)
# ---------------------------------------------------------------------------
import torchvision.models as tv_models
from pytorch_msssim import ssim as ssim_metric  # pip install pytorch-msssim


class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT).features[:16].to(device)
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, pred, target):
        pred3 = pred.repeat(1, 3, 1, 1)
        target3 = target.repeat(1, 3, 1, 1)
        pred3 = (pred3 - self.mean) / self.std
        target3 = (target3 - self.mean) / self.std
        return F.l1_loss(self.vgg(pred3), self.vgg(target3))


def sobel_edges(x):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)


def combined_loss_v4(pred, target, perceptual_loss_fn, ssim_weight=0.1, perceptual_weight=0.1, edge_weight=0.15):
    l1 = F.l1_loss(pred, target)
    ssim_l = 1 - ssim_metric(pred, target, data_range=1.0, size_average=True)
    perceptual = perceptual_loss_fn(pred, target)
    edge_loss = F.l1_loss(sobel_edges(pred), sobel_edges(target))
    return l1 + ssim_weight * ssim_l + perceptual_weight * perceptual + edge_weight * edge_loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=None,
                         help="Not set in the original run; pass a value for your own reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = PairedRestorationDataset(args.data_dir, augment=True)
    val_dataset = PairedRestorationDataset(args.data_dir, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = RestorationNet(base_ch=64, n_res_blocks=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    perceptual_loss_fn = VGGPerceptualLoss(device).to(device)

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for lr_img, gt_img in train_loader:
            lr_img, gt_img = lr_img.to(device), gt_img.to(device)
            optimizer.zero_grad()
            pred = model(lr_img)
            loss = combined_loss_v4(pred, gt_img, perceptual_loss_fn)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * lr_img.size(0)
        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lr_img, gt_img in val_loader:
                lr_img, gt_img = lr_img.to(device), gt_img.to(device)
                pred = model(lr_img)
                loss = combined_loss_v4(pred, gt_img, perceptual_loss_fn)
                val_loss += loss.item() * lr_img.size(0)
        val_loss /= len(val_dataset)
        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        torch.save(model.state_dict(), os.path.join(args.out_dir, "last_epoch.pt"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pt"))
            print(f"  -> new best model saved (val_loss={val_loss:.4f})")

    print(f"Training complete. Best checkpoint saved to {args.out_dir}/best_model.pt")


if __name__ == "__main__":
    main()
