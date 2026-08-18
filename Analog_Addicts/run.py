"""
run.py
Entry point for AI-Based Restoration of Degraded Images (KLA Problem Statement).

Usage:
    python run.py <input-dir> <output-dir>

Reads every .npy file in <input-dir>, restores it (denoise + 2x super-resolution),
and writes one restored .npy file per input to <output-dir> under the same filename.

Requires only a local model checkpoint (models/best_model.pt, included in this repo).
No internet access, API keys, or manual configuration needed at runtime.
"""

import sys
import os
import glob

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Model definition
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
    """Single end-to-end network: joint denoising (speckle + Gaussian) and
    2x super-resolution. Encoder-decoder with skip connections, residual
    bottleneck, PixelShuffle upsampling head, Sigmoid output (keeps values
    in [0,1])."""

    def __init__(self, base_ch=48, n_res_blocks=6):
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
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    os.makedirs(output_dir, exist_ok=True)

    # Locate the checkpoint relative to this script, so it works regardless
    # of the current working directory the script is launched from.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, "models", "best_model.pt")

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = RestorationNet(base_ch=64, n_res_blocks=8).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    input_files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    input_files = [f for f in input_files if "__MACOSX" not in f]
    print(f"Found {len(input_files)} input .npy files in {input_dir}")

    if len(input_files) == 0:
        print("WARNING: no .npy files found. Nothing to do.")
        return

    batch_size = 8
    n_done = 0

    for i in range(0, len(input_files), batch_size):
        batch_files = input_files[i:i + batch_size]

        batch_arrays = []
        for f in batch_files:
            arr = np.load(f).astype(np.float32)
            if arr.ndim == 3:  # (H,W,1) -> (H,W)
                arr = arr.squeeze(-1)
            batch_arrays.append(arr)

        batch_tensor = torch.from_numpy(np.stack(batch_arrays)).unsqueeze(1).to(device)  # (B,1,H,W)

        with torch.no_grad():
            pred = model(batch_tensor)

        # Safety: guarantee valid [0,1] output with no NaN/Inf, even though
        # Sigmoid already bounds the range under normal conditions.
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
        pred = torch.clamp(pred, 0.0, 1.0)

        pred_np = pred.cpu().numpy()

        for j, f in enumerate(batch_files):
            fname = os.path.basename(f)
            out_path = os.path.join(output_dir, fname)
            restored = pred_np[j, 0].astype(np.float32)  # (H,W)
            np.save(out_path, restored)
            n_done += 1

        print(f"Processed {n_done}/{len(input_files)}")

    print(f"\nDone. Restored {n_done} images. Saved to: {output_dir}")


if __name__ == "__main__":
    main()
