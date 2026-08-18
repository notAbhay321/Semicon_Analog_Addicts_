"""
baseline_compare.py
Satisfies the KLA checklist requirement: "Compare at least one baseline
with the final method."

Baseline: plain bicubic upsampling (no denoising) — the simplest possible
2x restoration approach, with zero learned parameters. This establishes
the floor that RestorationNet must beat to justify using a trained model
at all.

Usage:
    python baseline_compare.py --data_dir <path_to_paired_GT_NoisyLR_data> \
        --checkpoint ../models/best_model.pt --out_dir ./comparison_outputs

Requires: numpy, torch, scikit-image (for PSNR/SSIM), lpips (pip install lpips)
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from run import RestorationNet  # reuse the exact submitted architecture


def bicubic_baseline(lr_tensor, scale=2):
    return F.interpolate(lr_tensor, scale_factor=scale, mode="bicubic", align_corners=False).clamp(0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Folder with GT/*.npy and NoisyLR/*.npy")
    parser.add_argument("--checkpoint", type=str, default="../models/best_model.pt")
    parser.add_argument("--out_dir", type=str, default="./comparison_outputs")
    parser.add_argument("--use_lpips", action="store_true",
                         help="Also compute LPIPS (requires `pip install lpips`)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RestorationNet(base_ch=64, n_res_blocks=8).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    lpips_fn = None
    if args.use_lpips:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex").to(device)

    gt_dir = os.path.join(args.data_dir, "GT")
    lr_dir = os.path.join(args.data_dir, "NoisyLR")
    filenames = sorted(f for f in os.listdir(gt_dir) if f.endswith(".npy"))

    results = {"bicubic": {"psnr": [], "ssim": [], "lpips": []},
               "model":   {"psnr": [], "ssim": [], "lpips": []}}

    for fname in filenames:
        gt = np.load(os.path.join(gt_dir, fname)).astype(np.float32)
        lr = np.load(os.path.join(lr_dir, fname)).astype(np.float32)
        if gt.ndim == 3:
            gt = gt.squeeze(-1)
        if lr.ndim == 3:
            lr = lr.squeeze(-1)

        lr_t = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            bicubic_out = bicubic_baseline(lr_t)[0, 0].cpu().numpy()
            model_out = model(lr_t)[0, 0].cpu().numpy()

        for name, out in [("bicubic", bicubic_out), ("model", model_out)]:
            results[name]["psnr"].append(psnr_fn(gt, out, data_range=1.0))
            results[name]["ssim"].append(ssim_fn(gt, out, data_range=1.0))
            if lpips_fn is not None:
                gt_t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
                out_t = torch.from_numpy(out).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
                with torch.no_grad():
                    d = lpips_fn(gt_t * 2 - 1, out_t * 2 - 1).item()
                results[name]["lpips"].append(d)

    print(f"\n{'Method':<12}{'PSNR':<10}{'SSIM':<10}{'LPIPS':<10}")
    for name in ["bicubic", "model"]:
        p = np.mean(results[name]["psnr"])
        s = np.mean(results[name]["ssim"])
        l = np.mean(results[name]["lpips"]) if results[name]["lpips"] else float("nan")
        print(f"{name:<12}{p:<10.4f}{s:<10.4f}{l:<10.4f}")

    with open(os.path.join(args.out_dir, "baseline_comparison.txt"), "w") as f:
        f.write(f"{'Method':<12}{'PSNR':<10}{'SSIM':<10}{'LPIPS':<10}\n")
        for name in ["bicubic", "model"]:
            p = np.mean(results[name]["psnr"])
            s = np.mean(results[name]["ssim"])
            l = np.mean(results[name]["lpips"]) if results[name]["lpips"] else float("nan")
            f.write(f"{name:<12}{p:<10.4f}{s:<10.4f}{l:<10.4f}\n")

    print(f"\nSaved to {args.out_dir}/baseline_comparison.txt")


if __name__ == "__main__":
    main()
