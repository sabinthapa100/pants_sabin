# PanTS pancreatic-tumor segmentation

This repository develops a scientifically controlled pancreatic-tumor
segmentation baseline on the [PanTS dataset](https://github.com/MrGiovanni/PanTS),
starting with [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet).

The primary endpoint is pancreatic-tumor detection, localization, and
segmentation. The current target is one exclusive semantic map containing
background plus all 28 PanTS foreground classes; pancreatic lesion is label
28. The additional anatomy is retained because it provides clinically relevant
context around the pancreas.

## Data policy

- `PanTS/` is immutable upstream data and is ignored by this repository.
- PanTS-tr is used for data understanding, training, cross-validation, model
  selection, and any future threshold or postprocessing decisions.
- PanTS-te is reserved for one locked in-distribution evaluation after the
  model and evaluation protocol are frozen.
- The 40-case `Dataset501_PanTSSmoke` experiment is an integration test, not a
  performance benchmark.
- Full production training will use a separate `Dataset500_PanTS` containing
  all 9,000 PanTS-tr cases on persistent HPC storage.

Nothing inside `PanTS/` is renamed, resampled, overwritten, or deleted.

## Repository layout

```text
pants_sabin/
├── README.md
├── requirements.txt
├── PanTS/                       # ignored immutable upstream data
├── src/data/                    # reusable PanTS data/QC/conversion functions
├── scripts/                     # small command-line entry points
├── notebooks/
│   └── PanTS_nnUNet_Colab.ipynb
├── nnunet/                      # ignored nnU-Net raw/preprocessed/results
└── outputs/                     # ignored generated figures/predictions/logs
```

`src/models/`, `src/training/`, and `src/evaluation/` are intentionally empty.
Standard nnU-Net already provides the architecture, loss, optimizer,
augmentation, checkpointing, sliding-window inference, and generic semantic
evaluation. Project-owned code will be added there only for a genuinely new
method or a fully specified PanTS benchmark evaluator.

## Environment

Create and activate a Python 3.11 Conda environment, install a PyTorch build
compatible with the target GPU, then install the project requirements:

```bash
conda create -n pants_sabin python=3.11 -y
conda activate pants_sabin

# Install PyTorch for this machine first: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

The tested local environment used nnU-Net v2.8.1. PyTorch is deliberately not
pinned in `requirements.txt` because laptop, Colab, and NERSC CUDA builds are
machine-specific. Record the exact PyTorch, CUDA, nnU-Net, and GPU versions for
every experiment.

Set the nnU-Net locations from the repository root:

```bash
export nnUNet_raw="$PWD/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$PWD/nnunet/nnUNet_preprocessed"
export nnUNet_results="$PWD/nnunet/nnUNet_results"
```

## Existing project commands

Inspect a training case numerically:

```bash
python scripts/inspect_data.py --case PanTS_00000001
```

Create mask-guided anatomical QC views:

```bash
python scripts/visualize_data.py \
  --case PanTS_00000003 \
  --structures pancreas pancreatic_lesion
```

Create the deterministic 40-case nnU-Net smoke dataset:

```bash
python scripts/prepare_nnunet.py \
  --dataset-id 501 \
  --name PanTSSmoke \
  --max-cases 40
```

The adapter creates nnU-Net-compatible CT and label links from PanTS-tr only,
checks geometry and integer labels, balances 20 lesion-positive and 20
lesion-negative cases, covers labels 1-28, and writes
`overwrite_image_reader_writer: NibabelIOWithReorient`. Reorientation is an
in-memory permutation/flip to RAS; it does not rewrite the source NIfTIs.

## nnU-Net smoke protocol

The [Colab notebook](notebooks/PanTS_nnUNet_Colab.ipynb) is the executable
runbook. It can also be opened directly in
[Google Colab](https://colab.research.google.com/github/sabinthapa100/pants_sabin/blob/main/notebooks/PanTS_nnUNet_Colab.ipynb).
Its controlled sequence is:

1. verify the raw PanTS download;
2. clone a pinned repository release;
3. audit Python/PyTorch/CUDA/GPU/RAM/disk;
4. create and validate Dataset501;
5. generate an orientation-aware fingerprint and default plans;
6. preprocess only `3d_fullres` with one worker;
7. create one fixed, lesion-stratified five-fold split;
8. run official `nnUNetTrainer_1epoch` on fold 0;
9. require completed full-volume validation and `summary.json`;
10. test the raw-input prediction interface on fold-0 validation cases;
11. evaluate semantic Dice/IoU, explicitly reporting label 28;
12. export a portable nnU-Net model ZIP and checksum.

One epoch still contains 250 optimizer iterations and 50 online patch
validation iterations. It is only a pipeline test. `--val` means
**validation-only** and is not included in the initial training command;
ordinary `nnUNetv2_train` automatically performs full-volume validation after
training.

## Production protocol

A reportable nnU-Net result requires:

- Dataset500 containing all 9,000 PanTS-tr cases;
- one frozen split policy reused by every compared method;
- standard five-fold cross-validation with `nnUNetTrainer` rather than a debug
  trainer;
- configuration/model selection using PanTS-tr out-of-fold predictions only;
- a frozen model, checkpoint rule, inference settings, and postprocessing;
- five-fold ensemble prediction on all 901 PanTS-te cases exactly once;
- one final semantic evaluation plus the official PanTS evaluator if JHU makes
  its complete metric protocol available.

The released metadata does not provide an explicit patient-group identifier,
so case-level cross-validation cannot prove patient independence. This must be
documented or resolved with the dataset authors before publication-grade claims.

nnU-Net's evaluator reports voxel-wise Dice, IoU, TP, FP, FN, and TN. It does
not reproduce PanTS patient-wise sensitivity, tumor-wise sensitivity,
specificity, or AUC because the public PanTS material does not fully specify
the lesion matching, probability score, component filtering, thresholding, and
empty-case aggregation rules. Until those definitions are obtained, these
quantities must not be labeled official PanTS metrics.

## Colab and checkpointing

Google Colab is suitable for Dataset501 smoke runs, but managed runtimes are
ephemeral and cannot be guaranteed not to disconnect. The notebook keeps
active raw/preprocessed data under fast `/content`, stores durable artifacts in
Google Drive, verifies SHA-256 checksums, and exposes explicit `fresh`,
`resume`, and `validate_only` actions. Standard nnU-Net continuation uses
`--c` and the matching dataset/configuration/fold/trainer/plans identity.

Full 9,000-case preprocessing and five-fold training require persistent,
high-throughput storage and should run on NERSC or dedicated cloud
infrastructure rather than ordinary Colab Pro.

## Model export

Use `nnUNetv2_export_model_to_zip` after validation completes. Ship the ZIP
with its checksum, dataset/plans/split identifiers, package versions, hardware,
commands, validation summary, citations, and a research-only model card. Do not
publish PanTS CTs, labels, reports, metadata, or preprocessed arrays. Keep any
Hugging Face repository private until the model-weight redistribution terms are
confirmed with the PanTS authors.

## Citations

- Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. *nnU-Net: a
  self-configuring method for deep learning-based biomedical image
  segmentation.* Nature Methods. 2021;18:203-211.
- Li W, et al. *PanTS: The Pancreatic Tumor Segmentation Dataset.* NeurIPS 2025.
