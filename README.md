# Analog_Addicts — AI-Based Restoration of Degraded Images

Solution for the KLA problem statement, Hackathon 2026 (SEMICON India).

Restores grayscale semiconductor inspection images degraded by speckle noise, additive Gaussian noise, and 2x downsampling — recovering a clean, full-resolution image from a noisy, low-resolution input in a single forward pass.

## Folder structure

```text
Analog_Addicts/
├── run.py
├── train.py
├── requirements.txt
├── README.md
├── configs/
│   └── config.yaml
├── models/
│   └── best_model.pt
└── results/
    ├── README.md
    ├── baseline_compare.py
    ├── baseline_comparison.txt
    ├── example_success_1.png
    └── example_failure_1.png


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

## Baseline comparison

The submitted RestorationNet was also compared against plain bicubic 2x upsampling on the available paired GT/NoisyLR dataset.

| Method | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|
| Bicubic | 22.8530 | 0.5361 | 0.4435 |
| RestorationNet | 27.9553 | 0.7526 | 0.2678 |

Compared with bicubic upsampling, RestorationNet improves PSNR by 5.1023 dB and SSIM by 0.2165, while reducing LPIPS by 0.1757.

These results show that the learned restoration model provides a substantial improvement over simple bicubic upsampling on the evaluated paired dataset.

See the `results/` directory for the baseline comparison script, measured results, and representative successful and failure examples.

## Training

The training implementation is included in `train.py`.

The submitted configuration is documented in:

```text
configs/config.yaml

The configuration specifies:

RestorationNet architecture with base_ch=64 and 8 residual blocks.
3200 training pairs.
256x256 ground-truth images and 128x128 noisy low-resolution inputs.
Speckle noise, additive Gaussian noise, and downsampling degradations.
Adam optimizer with learning rate 2e-4.
40 training epochs.
Batch size 16.
Cosine annealing learning-rate scheduling.
Combined L1, SSIM, VGG-perceptual, and Sobel-edge losses.
Random crop, horizontal flip, vertical flip, and 90-degree rotation augmentation.

The checkpoint used for inference is:

models/best_model.pt
Results and evaluation artifacts

The results/ directory contains:

baseline_compare.py — compares the submitted model against bicubic upsampling.
baseline_comparison.txt — measured PSNR, SSIM, and LPIPS comparison.
example_success_1.png — representative successful restoration example.
example_failure_1.png — representative failure example.
README.md — description of the evaluation artifacts.
```

## Known limitations

- Trained only on the 256→128 (2x) downsampling scale provided in the
  training set.
- SSIM scores naturally lower on this data's fine speckle texture even on
  visually accurate reconstructions.
- Fine periodic/grating structures show some over-smoothing on
  out-of-distribution test content — a known, evidence-identified limitation. out-of-distribution test content — a known, evidence-identified limitation.
