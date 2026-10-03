# Target-Aware Dual-Branch HSI Target Detection

This repository implements a compact PyTorch detector for **supervised,
same-source hyperspectral image (HSI) target detection**. Train, validation,
and test pixels are sampled from the same image. The detector combines raw
spectral evidence, full-image PCA evidence, a target-aware spatial gate, and an
explicit center-pixel feature for binary background/target prediction.

The repository intentionally contains only the same-source detection pipeline.
Cross-dataset transfer, cross-scene evaluation, ablation frameworks, and legacy
classification backbones are not part of this release.

## Method overview

For a centered HSI patch, the input consists of the original spectrum followed
by its full-image whitened PCA representation.

1. **Raw-spectral branch** — a 1×1 convolution maps the raw bands to 64
   channels, followed by batch normalization, SiLU activation, residual context
   blocks with dilations 1 and 2, and projection to the PCA feature width.
2. **PCA branch** — a 1×1 convolution normalizes the PCA patch, followed by
   residual context blocks with dilations 1 and 3.
3. **Feature fusion and spatial gating** — learned raw features, learned PCA
   features, and the original PCA patch are concatenated. A 1×1
   convolutional gate maps this fused representation to 16 channels and then to
   one spatial confidence map. Sigmoid outputs attenuate background-like
   locations after multi-scale fusion.
4. **Multi-scale context** — depthwise-separable convolutions with dilations
   1, 2, and 4 operate in parallel on the concatenated spectral evidence. Their
   outputs and the three identity paths are projected by a 1×1 convolution.
5. **Center-aware decision head** — global average pooling captures patch
   context, while the central spatial feature preserves evidence at the
   candidate pixel. Their concatenation is classified by LayerNorm, a linear
   hidden layer, SiLU, dropout (0.1), and a two-logit output layer.

For the reference configuration (89 raw bands, 32 PCA channels, 9×9 patches,
and 64 hidden channels), the detector has **55,603 parameters** and requires
**4,230,960 multiply–accumulate operations (MACs) per patch**. Under the common
two-operations-per-MAC convention, this corresponds to 8,461,920 FLOPs.

## Data format

Place one MATLAB MAT file in the repository root or supply its path with
`--dataset_path`. The file must contain:

- `data`: a numeric array with shape `(height, width, bands)`;
- `map`: an integer array with shape `(height, width)`.

In `map`, background pixels must be `0` and target pixels must be `1`. Both
classes must be represented. The provided project MAT files are not committed
because of their size and data-redistribution restrictions.

The loader performs minimum–maximum normalization, per-band mean centering,
and full-image PCA. PCA uses spectra from the same image but does **not** use
ground-truth labels. Labels are used for the supervised stratified split, loss
weighting, sampler, validation checkpoint selection, and threshold selection.

## Installation

Python 3.9 or newer is required.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

For a CUDA installation, follow the official [PyTorch installation
instructions](https://pytorch.org/get-started/locally/) and select a wheel
matching your CUDA runtime before installing the remaining requirements.

## Usage

Run five-seed supervised detection on one same-source dataset:

```bash
python main_detection.py \
  --dataset_path dataset.mat \
  --output_dir Results_Detection \
  --device cuda:0 \
  --num_runs 5
```

Useful options include:

- `--patch_size`: spatial patch size (default: 9);
- `--pca_num`: number of whitened PCA components (default: 32);
- `--train_ratio`, `--val_ratio`: stratified split ratios (defaults: 0.10 and
  0.10, leaving 0.80 for testing);
- `--epochs`, `--batch_size`, `--learning_rate`: optimization controls;
- `--num_runs`: number of repeated seeds;
- `--seed`: initial seed (default: 202201; run `r` uses `seed + r`);
- `--save_png`: additionally export probability and prediction PNGs.

Training uses AdamW, weighted cross-entropy, a balanced random sampler, random
flips/rotations for training patches, and cosine learning-rate decay. The
checkpoint with the highest validation PR-AUC is restored. Its validation
scores are then used to choose the threshold with maximum F1; that threshold is
applied unchanged to the held-out test pixels.

During inference, every pixel is treated as a candidate center. Boundary
patches use reflect padding, and the target probability is the softmax
probability of class 1.

## Outputs

For run `r`, `Results_Detection/run_r` contains:

| File | Description |
| --- | --- |
| `probability_map.npy` | Target probability for every image pixel |
| `prediction.npy` | Binary prediction using the validation-selected threshold |
| `test_mask.npy` | Boolean mask of held-out test pixels |
| `model_best.pth` | Best validation checkpoint |
| `detector_bundle.pth` | Model state plus threshold and input configuration |
| `metrics.json` | Class counts, confusion matrix, and detection metrics |
| `probability_map.png`, `prediction.png` | Optional visualizations with `--save_png` |

`Results_Detection/summary.json` reports the mean and population standard
deviation over runs for precision, probability of detection (Pd), false-alarm
rate (FAR), F1, ROC-AUC, and PR-AUC.

## Tests

Run the unit tests with:

```bash
python -m unittest discover -s tests -v
```

The tests cover stratified splitting, class weighting, balanced sampling, MAT
loading, metrics and threshold selection, patch construction, the fixed model
architecture, parameter/MAC counts, training inference, and full-map inference.

## Repository structure

```text
.
├── main_detection.py          # Same-source training and evaluation CLI
├── models/
│   └── hsi_detector.py        # Target-aware dual-branch detector
├── utils/
│   ├── dataset.py             # Reflect-padded patch dataset and augmentation
│   ├── detection.py           # MAT loading, splitting, sampling, and metrics
│   └── detection_runtime.py   # Training, validation, and full-map inference
└── tests/
    └── test_detection.py      # Unit tests for the release path
```

## Intended use and limitations

- The pipeline is designed for supervised evaluation where labeled training,
  validation, and test pixels come from the same HSI scene.
- It does not claim cross-scene or cross-dataset generalization.
- Full-image PCA is transductive with respect to unlabeled spectra: it uses all
  spectra from the evaluated image, but never uses their labels.
- Metrics can be optimistic if target pixels are spatially adjacent to training
  target pixels. Users comparing protocols should state the split procedure.

## License

This project is released under the MIT License; see [LICENSE](LICENSE).
