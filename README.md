# PanTS 3D Pancreatic Tumor Segmentation

A reproducible research pipeline for training, comparing, evaluating, and exporting
3D pancreatic-tumor segmentation models on the **PanTS** dataset.

The project is organized around a common data/evaluation interface so different
model families can be compared fairly and exported through a simple inference
workflow for external testing.

## Study design

Three experiments are planned under the same PanTS development protocol:

| Experiment | Model | Initialization | Purpose |
|---|---|---|---|
| `segresnet_suprem` | 3D SegResNet | public SuPreM checkpoint | transfer-learning model |
| `segresnet_random` | same 3D SegResNet | random | controlled pretraining ablation |
| `nnunet3d` | nnU-Net v2 `3d_fullres` | framework default | independent strong baseline |

The two SegResNet experiments use the **same architecture and downstream
pipeline**. Their initialization is the controlled variable. nnU-Net is kept as
an independent baseline because it is a self-configuring framework with its own
planning, training, and inference machinery.

No public model already trained on PanTS is used as the submitted model.

## Scientific protocol

1. Keep the original PanTS data immutable.
2. Validate NIfTI geometry, labels, and dataset metadata before training.
3. Create a fixed PanTS-tr development split used across all experiments.
4. Use smoke/tiny-overfit experiments to validate each pipeline before large runs.
5. Tune model selection and post-processing using PanTS-tr development data only.
6. Freeze the complete pipeline before evaluating on PanTS-te.
7. Use one common evaluation implementation whenever the prediction format permits.
8. Export one selected final model with a minimal inference interface for external OOD testing.

PanTS-te is treated as a locked in-distribution evaluation set, not as a tuning set.

## Repository layout

```text
pants_sabin/
├── configs/                    # one experiment = one tracked configuration
│   ├── segresnet_suprem.yaml
│   ├── segresnet_random.yaml
│   └── nnunet3d.yaml
│
├── src/
│   ├── data/                   # PanTS paths, labels, I/O, QC, preparation
│   ├── models/                 # in-process model definitions/factory
│   ├── training/               # shared training/checkpoint utilities
│   └── evaluation/             # model-agnostic metrics and inference utilities
│
├── scripts/                    # thin command-line entry points
├── notebooks/                  # Colab orchestration/visualization only
├── tests/                      # fast unit and integration tests
├── requirements.txt
└── README.md
```

The notebooks are **not** the implementation. They are lightweight interfaces for
mounting cloud storage, selecting hardware, launching scripts, and visualizing
results. Reusable logic belongs under `src/`.

## PanTS ontology

The current data layer tracks background plus 28 PanTS foreground classes.
`pancreatic_lesion` is label **28**. Pancreas, pancreas head/body/tail, duct, and
surrounding anatomy are retained as contextual supervision rather than reducing
the task immediately to a binary mask.

## SuPreM usage

The transfer-learning experiment uses the publicly released **SuPreM SegResNet
pretrained weights as initialization only**. The model is then fine-tuned on
PanTS-tr. The same SegResNet is also trained from random initialization to
quantify the effect of supervised 3D pretraining.

The upstream SuPreM pancreatic-tumor example is treated as a methodological
reference; PanTS-specific data handling, experiment control, evaluation,
checkpointing, and export are implemented in this repository.

Set the checkpoint location with an environment variable rather than committing
weights:

```bash
export SUPREM_CHECKPOINT=/path/to/supervised_suprem_segresnet_2100.pth
```

## Data paths

Raw data and model artifacts are never committed. Configure storage through
environment variables:

```bash
export PANTS_DATA_ROOT=/path/to/PanTS/data
export PANTS_OUTPUT_ROOT=/path/to/pants_outputs
```

The official PanTS layout is case-based (`ImageTr/<case>/ct.nii.gz`,
`LabelTr/<case>/...`). Existing utilities under `src/data/` operate on this
case structure.

## Existing data/QC tools

Inspect one case:

```bash
python scripts/inspect_data.py --case PanTS_00000001
```

Visualize selected structures:

```bash
python scripts/visualize_data.py \
  --case PanTS_00000003 \
  --structures pancreas pancreatic_lesion
```

Create the deterministic balanced nnU-Net smoke dataset:

```bash
python scripts/prepare_nnunet.py \
  --dataset-id 501 \
  --name PanTSSmoke \
  --max-cases 40
```

## Reproducibility

SegResNet training checkpoints are designed to preserve model, optimizer,
scheduler, precision-scaler state (when applicable), epoch/global step,
best metric, resolved configuration, RNG states, and the producing Git commit.
Training checkpoints and submission checkpoints are intentionally separate
artifacts.

## Evaluation

The repository currently implements model-agnostic semantic Dice utilities.
PanTS benchmark quantities such as patient-wise sensitivity, tumor-wise
sensitivity, specificity, and AUC will only be labeled as PanTS-compatible after
the exact lesion-matching and threshold protocol has been verified and encoded.
No benchmark results are fabricated or inferred from smoke experiments.

## Current status

This branch is the modular refactor of the original nnU-Net-first prototype.
Implemented so far:

- PanTS label/path/I/O/QC utilities from the original prototype;
- deterministic balanced nnU-Net smoke-data preparation;
- three tracked experiment configurations;
- shared SegResNet model factory with SuPreM or random initialization;
- atomic resumable training-checkpoint utilities;
- shared semantic segmentation metrics and tests.

Next implementation milestone: a correct PanTS manifest/split layer followed by
the shared MONAI SegResNet data pipeline and geometry-preserving inference CLI.

## References

- PanTS: Li et al., NeurIPS 2025 Datasets and Benchmarks Track.
- SuPreM / supervised 3D medical-image pretraining: Li et al.
- nnU-Net: Isensee et al., *Nature Methods* (2021).
- MONAI: Medical Open Network for AI.
