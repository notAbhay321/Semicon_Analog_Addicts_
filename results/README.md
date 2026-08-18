# results/

This folder satisfies the KLA checklist items:
- "Show restored examples at full image resolution, including successful and failed cases."
- "Compare at least one baseline with the final method."

## What's here

- `baseline_compare.py` — runs both a plain bicubic-upsampling baseline and
  the submitted model (`models/best_model.pt`) on the same validation set,
  and reports PSNR/SSIM/LPIPS for each side by side.

## What YOU still need to add (can't be generated without your real data)

1. **Run `baseline_compare.py`** against your actual held-out validation set:
   ```bash
   cd results
   python baseline_compare.py --data_dir /path/to/validation_GT_NoisyLR --use_lpips
   ```
   This produces `baseline_comparison.txt` — drop that file in this folder.

2. **Save a handful of visual examples** (recommended: 4-6 images):
   - 2-3 "successful" cases: input / model output / ground truth, side by side
   - 1-2 "failure" cases: where the model over-smooths or misses detail
     (per the README's "Known limitations" section — fine periodic/grating
     structures are a documented weak spot, so a grating-heavy image is a
     good failure-case candidate)
   - Save these as `.png` comparison grids (matplotlib subplot, like the
     v3-vs-v4 comparison image you made earlier) into this folder, e.g.
     `example_success_1.png`, `example_failure_1.png`.

3. **Update this README** with a short paragraph summarizing what the
   comparison shows once you have the numbers — e.g. "Our model improves
   PSNR by X dB and SSIM by Y over the bicubic baseline, confirming the
   learned denoising step is providing real value beyond simple upsampling."
