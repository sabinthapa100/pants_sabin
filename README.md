# PanTS nnU-Net baseline

This project trains and evaluates **nnU-Net v2** for pancreatic-tumor
segmentation using the [PanTS dataset](https://github.com/MrGiovanni/PanTS).

The model receives a 3D abdominal CT and predicts background plus the 28 PanTS
classes. Pancreatic lesion is label 28. The other anatomical labels provide
useful context around the pancreas.

## Start here

The complete workflow is in
[`notebooks/PanTS_nnUNet_Colab.ipynb`](notebooks/PanTS_nnUNet_Colab.ipynb).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sabinthapa100/pants_sabin/blob/main/notebooks/PanTS_nnUNet_Colab.ipynb)

The notebook walks through:

1. checking the PanTS download;
2. preparing data in nnU-Net format;
3. fingerprinting, planning, and preprocessing;
4. training with checkpoints;
5. validation and prediction;
6. Dice/IoU evaluation, including pancreatic-lesion label 28;
7. exporting the trained nnU-Net model.

## Project layout

```text
pants_sabin/
├── PanTS/                 # original PanTS repository and data (read-only)
├── nnUNet/                # local official nnU-Net clone (reference only)
├── src/data/              # reusable PanTS data functions
├── scripts/               # small data/QC commands
├── notebooks/             # Colab nnU-Net workflow
├── nnunet/                # generated nnU-Net data and results
└── outputs/               # generated figures and predictions
```

`PanTS/`, `nnUNet/`, `nnunet/`, and `outputs/` are ignored by Git. The official
nnU-Net clone is only for reading its documentation and source; this project
uses the installed `nnunetv2` package and does not copy or modify nnU-Net code.

## Local setup

```bash
conda create -n pants_sabin python=3.11 -y
conda activate pants_sabin

# Install the correct PyTorch build for your GPU first.
pip install -r requirements.txt
```

From the project root, set the three nnU-Net directories:

```bash
export nnUNet_raw="$PWD/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$PWD/nnunet/nnUNet_preprocessed"
export nnUNet_results="$PWD/nnunet/nnUNet_results"
```

## Useful commands

Inspect one case:

```bash
python scripts/inspect_data.py --case PanTS_00000001
```

Visualize pancreas and lesion annotations:

```bash
python scripts/visualize_data.py \
  --case PanTS_00000003 \
  --structures pancreas pancreatic_lesion
```

Create the 40-case smoke dataset:

```bash
python scripts/prepare_nnunet.py \
  --dataset-id 501 \
  --name PanTSSmoke \
  --max-cases 40
```

## Smoke test and production training

`Dataset501_PanTSSmoke` contains 40 cases from PanTS-tr. It is used to check
that preprocessing, training, checkpointing, inference, and evaluation all
work. Its one-epoch results are **not benchmark results**.

The final nnU-Net benchmark will use:

- `Dataset500_PanTS` with all 9,000 PanTS-tr cases;
- fixed five-fold cross-validation;
- the standard nnU-Net trainer;
- model selection using PanTS-tr validation predictions only;
- five-fold ensemble inference on the 901 PanTS-te cases after the model and
  evaluation procedure are frozen.

For Colab Pro, use an A100/High-RAM runtime, train from fast `/content` storage,
and keep checkpoints and exported models in Google Drive. If Colab runtime or
storage limits become restrictive, the same nnU-Net commands can run on NERSC.

## Evaluation

nnU-Net reports semantic Dice and IoU for every class. For this project, label
28 must be reported separately because it represents pancreatic lesion.

The PanTS table also includes patient-wise sensitivity, tumor-wise sensitivity,
specificity, and AUC. We will add those metrics only after their exact lesion
matching and threshold definitions are established; until then, nnU-Net results
are reported as internal semantic validation.

## Data rule

Never modify files inside `PanTS/`. Use PanTS-tr for development and reserve
PanTS-te for the final locked evaluation.

## References

- [PanTS dataset](https://github.com/MrGiovanni/PanTS)
- [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet)
- Isensee et al., *nnU-Net: a self-configuring method for deep learning-based
  biomedical image segmentation*, Nature Methods (2021).
