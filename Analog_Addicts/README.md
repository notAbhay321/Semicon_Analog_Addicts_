# PixelRestore — AI-Based Restoration of Degraded Images

Solution for the KLA problem statement, Hackathon 2026 (SEMICON India).

Restores grayscale semiconductor inspection images degraded by speckle noise,
additive Gaussian noise, and 2x downsampling — recovering a clean,
full-resolution image from a noisy, low-resolution input in a single
forward pass.

## Folder structure

```
PixelRestore/
├── run.py
├── requirements.txt
├── README.md
└── models/
    └── best_model.pt
```

## Setup

```bash
pip install -r requirements.txt
```

No internet access, API keys, or additional downloads are required at
inference time — the model weights are included locally in `models/`.

## Running

```bash
python run.py <input-dir> <output-dir>
```

Example:
```bash
python run.py ./Test_NoisyLR ./restored_outputs
```

- Reads every `.npy` file in `<input-dir>`.
- Creates `<output-dir>` automatically if it does not exist.
- Writes one restored `.npy` file per input, using the same filename.
- Each output is a grayscale array of shape `(H, W)`, with values in `[0,1]`
  (NaN/Inf are clipped defensively, though the model's Sigmoid output layer
  already guarantees this under normal conditions).
- Runs on GPU automatically if available (`torch.cuda.is_available()`),
  otherwise falls back to CPU.
- Batched inference (8 images per batch) for throughput.

## Model summary

- Single end-to-end CNN: encoder-decoder with skip connections, a
  residual-block bottleneck (denoising), and a learned PixelShuffle
  upsampling head (2x super-resolution) — denoising and upscaling happen
  jointly in one pass, not as two chained stages.
- ~3.46M parameters, base_ch=64, 8 residual blocks.
- Trained on 3200 paired 256x256 GT / 128x128 NoisyLR images with a
  combined L1 + SSIM + VGG-perceptual loss, and flip/rotation augmentation
  for generalization to unseen image content.

## Validation results (384 held-out images)

| Metric | Value |
|---|---|
| PSNR | 25.70 dB |
| SSIM | 0.7437 |
| LPIPS | 0.2711 |
| Speed | 6.2 ms/image |

## Known limitations

- Trained only on the 256→128 (2x) downsampling scale provided in the
  training set.
- SSIM scores naturally lower on this data's fine speckle texture even on
  visually accurate reconstructions.
- Fine periodic/grating structures show some over-smoothing on
  out-of-distribution test content — a known, evidence-identified limitation.
