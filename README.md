# The BioMassters

[The
BioMassters](https://www.drivendata.org/competitions/99/biomass-estimation/page/534/)

[1st
place](https://www.drivendata.org/competitions/99/biomass-estimation/leaderboard/)
out of 976 participants with 27.6280 Average RMSE score (top2 27.6779).

![](./assets/private.png)

## Approach

The solution is based on an UNet model with a shared encoder with aggregation
via attention. The inputs to the encoder are 15-band images with a resolution
of 256x256 from joint Sentinel-1 and Sentinel-2 satellite missions. The encoder
is shared for all 12 months. The outputs are aggregated via self-attention.
Finally, a decoder takes as inputs the aggregated features and predicts a
single yearly agbm. We directly optimize `RMSE` using `AdamW` optimizer and
`CosineAnnelingLR` scheduler. We don't compute loss for high agbm values
(>400). We use vertical flips, rotations, and random month dropout as
augmentations. Month dropout simply removes images.

## Highlights

- UNet model
  * Shared `tf_efficientnetv2_l_in21k` encoder for all months. The input is
    15-band images with min-max normalization. Aggregate over all 12 months via
    self-attention at the encoder level
  * Heavy decoder takes aggregated features and predicts single yearly agbm
- `RMSE` loss (ignore high values >400)
- Train 900 epochs on one split (fold), then finetune 200 epochs on full
  dataset, 8 batch size per GPU, mixed precision
  * `AdamW` optimizer with `1e-3` learning rate and `1e-2` weight decay
  * `CosineAnnealingLR` scheduler
- Augmentations: random flips, rotations, and month dropout
- Test time augmentations: left-right and up-down flips

## Prerequisites & Hardware

- GNU/Linux
- [Python 3](https://www.python.org/)
- Nvidia Driver Version: 515.65.01
- CUDA Version: 11.7
- [PyTorch 1.13](https://pytorch.org/)
- 2 x GPU Nvidia A100 40GB VRAM
- 4 x CPU AMD Milan 7413 @ 2.65 GHz 128M cache L3
- 64GB RAM
- 8 days for training
- 5 min for inference

## Setup

Create an environment using Python 3.8. The solution was originally run on
Python 3.8.10. Install the required Python packages

```bash
pip install -r requirements.txt
```

Download the data from the competition page and unzip into `data` folder.

### Windows / RTX 50-series setup

Use the local `.venv` environment with Python 3.11 and CUDA 12.8 PyTorch.
The original PyTorch 1.13 build does not support RTX 50-series GPUs. The
Windows requirements retain the older timm and segmentation-models-pytorch
APIs used by this solution.

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe -r requirements-windows.txt
.\.venv\Scripts\Activate.ps1
```

Alternatively, invoke `.\.venv\Scripts\python.exe` directly without activation.
Select `.venv\Scripts\python.exe` as your editor's Python interpreter.
Run `python src/train.py --help` or `python src/submit.py --help` to inspect
the available options. The `.sh` launchers require a Unix shell, and multi-GPU
training uses NCCL, which requires Linux/WSL2. For native Windows, invoke the
Python scripts directly for a single GPU and use a smaller batch size than
the original A100 settings.

The tracked `data` and `models` entries are placeholder files containing the
author's Linux paths, not datasets or checkpoints. Download those separately
and pass their actual directories using the scripts' command-line options.
When loading trusted original checkpoints with PyTorch 2.7, set
`$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = "1"` before running the scripts;
they deserialize complete model objects using `torch.load`.

### Frozen encoder feature extraction

Run this launcher from Git Bash to extract the first 100 chip/month samples
using the local checkpoint and dataset paths:

```bash
bash ./extract_features.sh
```

Or run directly in PowerShell with explicit paths:

```powershell
.\.venv\Scripts\python.exe src/extract_features.py `
  --checkpoint "models/tf_efficientnetv2_l_in21k_f0_b8x2_e100_nrmse_devscse_attnlin_augs_decplus7_plus800eb_200ft/modelo_best.pth" `
  --train-features-dir "E:/Datasets/Biomassters/train_features/Volumes/Samsung_T5/BIOMASS/BioMasster_Dataset/v1/DrivenData/train_features" `
  --output-dir "E:/Datasets/Biomassters/extracted_features" `
  --batch-size 4 --num-workers 2 --max-samples 100
```

CLI arguments override launcher defaults. Use `--max-samples 0` for all
samples, `--batch-size 1` to reduce GPU memory use, `--num-workers 0` to load
inputs in the main process, or `--device cpu` for CPU execution. On Linux/WSL,
override the paths with mounted Linux paths and use a Linux virtual environment.
Every extractor option is configurable in the launcher's configuration block
or through environment variables: `CHECKPOINT`, `TRAIN_FEATURES_DIR`,
`OUTPUT_DIR`, `BATCH_SIZE`, `NUM_WORKERS`, `DEVICE`, `MAX_SAMPLES`, and
`OVERWRITE`. `PYTHON` overrides the interpreter; otherwise the launcher uses
the local virtual environment or system `python3` (including Colab).
`OVERWRITE=true` enables recomputation and `OVERWRITE=false` preserves resume
behavior. For example:

```bash
BATCH_SIZE=2 NUM_WORKERS=0 MAX_SAMPLES=100 OVERWRITE=false bash ./extract_features.sh
```

The extractor loads the trusted complete checkpoint, maps its legacy timm
class names, and strictly transfers the encoder weights into the current
encoder. It downloads no pretrained weights. Parameters are frozen and all
computation and saved arrays use float32, with autocast and TF32 disabled.
Outputs precede temporal attention and decoding.

Matching S1 and S2 images supply 15 normalized input bands for each chip/month.
Missing S2 images are filled with zeros, as in training. The sample limit
counts combined chip/month inputs, rather than individual sensor TIFFs.
For example, `0003d2eb_S1_00.tif` and `0003d2eb_S2_00.tif` produce:

| Output relative to extracted_features | Shape (channels, height, width) |
| --- | --- |
| `128x128/0003d2eb_00.npy` | `(32, 128, 128)` |
| `64x64/0003d2eb_00.npy` | `(64, 64, 64)` |
| `32x32/0003d2eb_00.npy` | `(96, 32, 32)` |
| `16x16/0003d2eb_00.npy` | `(224, 16, 16)` |
| `8x8/0003d2eb_00.npy` | `(640, 8, 8)` |

```python
import numpy as np
features = np.load(
    "E:/Datasets/Biomassters/extracted_features/128x128/0003d2eb_00.npy",
    allow_pickle=False,
)
```

The manifest records the checkpoint SHA-256, source directory, normalization,
precision, and output shapes. Rerunning skips samples with five valid arrays;
incomplete samples are recomputed. Use `--overwrite` to recompute the selected
samples. A mismatched manifest is rejected even with `--overwrite`; choose a
separate directory for a different checkpoint or extraction configuration.
Keep the input TIFFs unchanged when resuming into an existing output directory.

The 100-sample test writes 500 arrays totaling about 375 MiB. The full 104,268
samples require about 382 GiB, plus file overhead. Free disk space is checked
before inference; the full run requires more space than the current E: drive
has available. The disk-space check rounds each NPY file, including its header,
to the output filesystem's allocation unit. For example, an exFAT drive with
1 MiB clusters uses 8 MiB per chip/month (about 815 GiB for the full dataset),
despite only 3.75 MiB of tensor data. A filesystem with 4 KiB allocation units
uses about 384 GiB plus metadata. Free space can also change during extraction
if other processes write to the same drive.

### Training

`src/train.py` accepts raw images by default. Add `--feature-mode` to train
directly from the cached encoder features:

```bash
python src/train.py \
  --feature-mode \
  --train-features-dir /content/drive/MyDrive/Biomassters/extracted_features \
  --train-df /content/drive/MyDrive/Biomassters/The_BioMassters_-_features_metadata.csv.csv \
  --train-labels-dir /content/drive/MyDrive/Biomassters/train_agbm \
  --checkpoint-dir ./models/features_run \
  --batch-size 1 --num-workers 2 --num-epochs 100 --augs
```

Point the labels argument to the folder directly containing `chip_agbm.tif`.
Each metadata chip needs all 12 months at all five cached scales. The
100-sample extraction preview contains only eight fully extracted chips;
finish extraction for normal training, or supply metadata limited to those
complete chips and reduce `--n-folds` for a small smoke test.

Feature mode constructs no EfficientNet encoder and downloads no encoder
weights. It trains temporal attention, the decoder, and the prediction head
using the existing optimizer, losses, and fold split. `--augs` uses month
dropout only; spatial augmentation and image-flip validation TTA are disabled.
Feature mode honors `--num-workers`, including zero. Backbone identity and
encoder channel counts come from `manifest.json`.

By default the attention and decoder are freshly initialized. To initialize
them from the original winning checkpoint, append:

```bash
--load ./models/tf_efficientnetv2_l_in21k_f0_b8x2_e100_nrmse_devscse_attnlin_augs_decplus7_plus800eb_200ft/modelo_best.pth \
--dec-channels 384 368 352 336 320 --dec-attn-type scse
```

Complete model checkpoints and training state checkpoints are accepted by
`--load`. Feature mode ignores image encoder weights and strictly loads all
attention/decoder/head weights, so decoder settings must match the checkpoint.
Use `--load .../model_last.pth --resume` to resume a feature run including
optimizer and scheduler state. Resuming requires the same feature manifest
(source paths may change) and training mode. Feature-mode checkpoints contain
no image encoder and are intended for cached-feature inputs.

To run training from the command line

```bash
sh ./run.sh
```

It will take about 8 days on 2 A100 40GB GPUs.

### Inference

Download [pretrained](https://disk.yandex.ru/d/01YXhPyiKZifYw) models and
extract into `models` folder.

```bash
unzip models.zip -d models
```

To run inference from the command line

```bash
sh ./submit.sh
```

It takes ~5 minutes on 1 GPU A100 40GB (note on V100 32GB the results are
slightly different).
