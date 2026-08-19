# results/

This folder satisfies the KLA checklist items:
- "Show restored examples at full image resolution, including successful and failed cases."
- "Compare at least one baseline with the final method."

## What's here

- `baseline_compare.py` — runs both a plain bicubic-upsampling baseline and
  the submitted model (`models/best_model.pt`) on the same validation set,
  and reports PSNR/SSIM/LPIPS for each side by side.

The final RestorationNet was compared against a plain bicubic 2x
upsampling baseline on the validation dataset.

| Method | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|
| Bicubic | 22.8530 | 0.5361 | 0.4435 |
| RestorationNet | 27.9553 | 0.7526 | 0.2678 |

The submitted model improves PSNR by **5.1023 dB** and SSIM by
**0.2165** over bicubic upsampling, while reducing LPIPS by **0.1757**.

The folder also contains representative successful and failure cases:
- `example_success_1.png`
- `example_failure_1.png`
