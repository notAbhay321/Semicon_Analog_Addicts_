# results/

This folder contains the evaluation artifacts for the submitted RestorationNet model and satisfies the KLA checklist requirements:

- Show restored examples at full image resolution, including successful and failed cases.
- Compare at least one baseline with the final method.

## What's here

- `baseline_compare.py` — evaluates both a plain bicubic-upsampling baseline and the submitted model (`models/best_model.pt`) on the same paired dataset and reports PSNR/SSIM/LPIPS.
- `baseline_comparison.txt` — measured comparison between bicubic upsampling and the submitted RestorationNet.
- `example_success_1.png` — representative successful restoration example.
- `example_failure_1.png` — representative failure case showing a limitation of the model.

## Baseline comparison

The final RestorationNet was compared against plain bicubic 2x upsampling.

| Method | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|
| Bicubic | 22.8530 | 0.5361 | 0.4435 |
| RestorationNet | 27.9553 | 0.7526 | 0.2678 |

Compared with bicubic upsampling, RestorationNet improves:

- **PSNR by 5.1023 dB**
- **SSIM by 0.2165**
- **LPIPS decreases by 0.1757**

These results show that the learned restoration model provides a substantial improvement over simple bicubic upsampling on the evaluated paired dataset.

## Visual examples

Two representative examples are included:

- `example_success_1.png` — a successful restoration with low reconstruction error.
- `example_failure_1.png` — a failure case with substantially higher reconstruction error, illustrating that the model does not restore every image equally well.

## Reproducing the comparison

From the project root:

```bash
python results/baseline_compare.py --data_dir <path-to-dataset> --checkpoint models/best_model.pt --out_dir results/comparison_outputs --use_lpips
