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

### Training

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
