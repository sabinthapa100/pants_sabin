# PanTS 3D Pancreatic Tumor Segmentation

## Goal

Segment 28 abdominal structures plus the pancreatic lesion (class 28) from 3D CT,
and measure how much a supervised 3D pretraining checkpoint (SuPreM) actually
helps compared with training the identical network from random initialization.
The comparison, not the leaderboard number, is the point of the study.

## Dataset

[PanTS](https://github.com/MrGiovanni/PanTS): `ImageTr/<case>/ct.nii.gz` and
`LabelTr/<case>/combined_labels.nii.gz`.

| | |
| --- | --- |
| PanTS-tr | 9,000 cases — **882 lesion-positive**, 8,118 lesion-negative |
| PanTS-te | `PanTS_00009001`+ — **locked** until model selection is frozen |
| Classes | 29 (background + 28 structures); class 28 is `pancreatic_lesion` |
| Split | `pants_cv_v1.json`, seed 317, 5 folds stratified by class-28 presence |
| Fold 0 | 7,199 train / 1,801 validation |

`lesion_present` is defined **only** by class 28 appearing in `combined_labels`.
The metadata spreadsheet's `tumor?` column is carried for QC and never used as
ground truth; the two disagree on 44 cases.

## Experimental design

| Run | Model | Initialization |
| --- | --- | --- |
| `segresnet_random` | 3D SegResNet | seeded random |
| `segresnet_suprem` | the **same** 3D SegResNet | SuPreM checkpoint (81 of 83 tensors) |
| nnU-Net v2 `3d_fullres` | independent baseline | framework default — **paused** |

Architecture, split, preprocessing, sampling, loss, optimizer, schedule and all
training hyperparameters are held fixed; **initialization is the scientific
intervention**. Execution hardware is recorded per run and is *not* assumed
identical — see [Reproducibility](#reproducibility).

nnU-Net is deliberately not forced through our preprocessing. It is
self-configuring, and overriding its fingerprint would remove what makes it an
independent baseline; it shares only the case partition.

## Repository layout

```text
pants_sabin/
├── pants_cv_v1.json      fixed 5-fold split — tracked, defines the experiment
├── src/
│   ├── data/             paths, labels, I/O, QC, manifest, transforms, prepared cache
│   ├── models/           SegResNet + SuPreM transfer
│   ├── training/         trainer + resumable checkpoints
│   └── evaluation/       metrics + inference primitives
├── scripts/              command-line entry points
├── notebooks/            Colab orchestration only
├── tests/
└── TECHNICAL_NOTES.md    data QC evidence behind the preprocessing choices
```

## Setup

Install PyTorch first, matched to your own GPU and CUDA runtime — the repository
deliberately does not pin a CUDA build, because the Colab runtime and a local
laptop do not use the same wheel.

```bash
pip install -r requirements.txt
export PANTS_DATA_ROOT=/path/to/PanTS/data
```

## Data preparation

```bash
python scripts/prepare_data.py --manifest --split --workers 8
python scripts/prepare_data.py --prepare-segresnet --output <prepared-root> --workers 4
```

The first command writes the case manifest and the fixed split. The second runs
the deterministic preprocessing once and stores each case as a compressed `.npz`
holding two plain arrays — `image` float16 `[D,H,W]` in [0,1] and `label` uint8
`[D,H,W]` in 0–28. Training then reads the cache instead of repeating the same
resampling every epoch. Writes are atomic, so an interrupted job resumes.

## Preprocessing

| Step | Choice | Why |
| --- | --- | --- |
| Orientation | `RAS` | seven source orientations occur in PanTS-tr, so canonicalizing is required, not cosmetic |
| Spacing | 1.5 mm isotropic | z-spacing spans 0.4–10 mm; without resampling a fixed patch covers wildly different anatomy per case |
| Patch | 96³ (144 mm/side) | fits the whole pancreas plus peripancreatic vessels in one window |
| Intensity | `[-175, 250] HU → [0,1]`, clipped | the input domain of the SuPreM supervised pretraining, so transfer is tested on its own terms |
| Sampling | tumor-aware, ratios 1:2:3 | the median lesion is ~0.013% of a volume; uniform crops would almost never see one |
| Augmentation | intensity only | PanTS has laterality-paired labels, so a left–right flip would mislabel a mirrored kidney |

Labels are always resampled with **nearest-neighbour**; linear interpolation
would invent fractional class IDs such as 17.4. Reorientation and resampling are
different operations: `Orientationd` permutes and flips axes (lossless index
bookkeeping), `Spacingd` changes the sampling lattice and must interpolate.

Evidence for each choice — the 1.5 mm lesion-preservation study, the degenerate
label-affine repair, and the measured sampler behaviour — is in
[TECHNICAL_NOTES.md](TECHNICAL_NOTES.md).

## Model

MONAI 3D SegResNet: `blocks_down=[1,2,2,4]`, `blocks_up=[1,1,1]`,
`init_filters=16`, 1 input channel, **29 output classes**, GroupNorm.

SuPreM transfer (`supervised_suprem_segresnet_2100.pth`, 56,500,623 bytes,
SHA256 `2db81dc0…89a3`): all 83 parameter names align with our architecture.
**81 transfer.** Only `conv_final.2.conv.{weight,bias}` is excluded — it maps
features to class logits, and SuPreM predicted 32 classes where PanTS needs 29.
`conv_final.0` is a task-independent GroupNorm and *is* transferred. The loader
raises unless exactly 81 tensors load, because a quietly half-initialized
backbone would invalidate the comparison.

## Training

```bash
python scripts/train_segresnet.py --initialization suprem \
  --experiment segresnet_suprem --pretrained-checkpoint $SUPREM_CHECKPOINT \
  --prepared-root <prepared-root> --manifest <prepared-root>/manifest.json \
  --split pants_cv_v1.json --fold 0 --epochs 64 \
  --batch-size 2 --samples-per-case 2 --accumulation 1 \
  --learning-rate 1e-4 --weight-decay 1e-5 --num-workers 4 --seed 317 \
  --save-every-epochs 1 --validate-every-epochs 5 --monitoring-negatives 50 \
  --output-root outputs/runs --persistent-output-root <durable-dir>
```

Production settings, identical for both arms:

| | |
| --- | --- |
| Epochs | **64** |
| Batch | 2 cases × 2 patches = 4 patches per forward |
| Accumulation | 1 |
| Optimizer | AdamW, lr 1e-4, weight decay 1e-5, cosine annealing (`T_max` = epochs) |
| Loss | `DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)` |
| Precision | AMP with `GradScaler` |
| Seed | 317, applied **before** the model is constructed |

Vocabulary, for this exact run:

- **case** — one patient CT plus its label map, resampled to RAS/1.5 mm
- **patch** — a 96³ crop; 2 are drawn per case per epoch
- **batch** — the tensor entering one forward pass, `[4,1,96,96,96]`
- **iteration** — one forward + backward over one batch; **3,600 per epoch**
- **optimizer update** — one `optimizer.step()`; also 3,600 per epoch, since
  accumulation is 1

One epoch = 7,199 cases = **14,398 patch presentations**. (Nominal 3,600 × 4 =
14,400; 7,199 is odd, so the last batch carries one case and two patches.) Over
64 epochs: **230,400 optimizer updates**, the same order as the nnU-Net baseline
this study compares against, whose default schedule is 250,000.

Background is excluded from the Dice term because it fills most of every patch
and would dominate the overlap signal; cross-entropy still sees it.

## Validation and model selection

Two signals with different jobs:

- **Patch-validation DiceCE** — every epoch, on random crops with augmentation
  off. Cheap, stochastic, **diagnostic only**. It moves between calls on an
  unchanged model and never selects anything.
- **Deterministic whole-volume monitoring** — every 5 epochs on a fixed subset
  of fold-0 validation: all **177 lesion-positive** cases plus **50** stride-
  sampled negatives = **227 cases**. No random cropping, so re-running it on an
  unchanged model returns the identical number.

`best.pt` is written only when the mean class-28 Dice over the lesion-positive
monitoring cases improves. `latest.pt` is written every epoch for resume, and
both are mirrored to durable storage from inside the training process, so a dead
runtime costs at most one epoch.

## Evaluation

```bash
python scripts/evaluate_segresnet.py --checkpoint <run>/best.pt \
  --prepared-root <prepared-root> --manifest <prepared-root>/manifest.json \
  --split pants_cv_v1.json --fold 0 --output evaluation/suprem/

python scripts/analyze_segresnet.py \
  --suprem-run <runs>/segresnet_suprem --random-run <runs>/segresnet_random \
  --suprem-eval evaluation/suprem --random-eval evaluation/random \
  --output figures/
```

The evaluator scores **all 1,801** fold-0 validation cases with whole-volume
sliding-window inference — no crops, no augmentation, no sampling — and writes
`evaluation_cases.csv` (per case: lesion voxels and volume for target and
prediction, class-28 Dice, internal detection and false-positive flags, seconds,
Dice for every class 1–28) plus `evaluation_summary.json` (aggregates,
checkpoint SHA256, sliding-window settings, software versions).

**These are internal development metrics.** The PanTS repository publishes no
evaluator, and patient-wise / tumor-wise sensitivity are defined only in prose:

| Benchmark column | Status |
| --- | --- |
| DSC | computed — plain class-28 Dice |
| P-Sen | an **internal** case-detection rate (any predicted class-28 voxel) |
| Spe | an **internal** `1 − FP rate` under the same criterion |
| T-Sen | **not computed** — needs an unpublished component-matching rule |
| AUC | **not computed** — needs an unpublished patient-scoring convention |

Empty-versus-empty Dice returns NaN, not 1.0: with 8,118 lesion-negative cases,
scoring them as perfect would report near-perfect tumor Dice for a model that
never predicts a tumor. Lesion Dice is averaged over lesion-positive cases only.
`scripts/infer_segresnet.py` emits the continuous class-28 probability, so
whoever holds the real protocol can compute their own operating points.

Case IDs above `PanTS_00009000` are refused unless `--allow-test-split` is
passed, so PanTS-te cannot be read by accident.

## Inference on an unseen CT

```bash
python scripts/infer_segresnet.py \
    --input scan.nii.gz \
    --output predictions/ \
    --checkpoint best.pt \
    --lesion-probability
```

Needs **only** this repository, a trained checkpoint, and a raw CT. No training
labels, no manifest, no split, no prepared cache, no Google Drive, no SuPreM
pretraining file, and no PanTS naming convention. `--input` also accepts a
directory of NIfTIs; for nested trees use a shell loop rather than a recursive
scan, which would silently pick up label files in a mixed directory.

```text
raw CT (native grid)
  → RAS → 1.5 mm → [-175,250] HU → [0,1]     image-only, same rules as training
  → sliding-window inference, CPU stitching   ~0.45 GB VRAM
  → argmax → labels 0..28, softmax → class-28 probability
  → inverse transform to the SOURCE grid      nearest for labels, linear for probability
  → combined_labels.nii.gz (uint8)
  → pancreatic_lesion_probability.nii.gz (float32, optional)
```

Outputs carry the source affine, shape and orientation. After training, `best.pt`
is simply a PanTS model — the initialization it started from is provenance, not a
runtime dependency.

## Reproducibility

Recorded in every checkpoint's provenance block and in
`evaluation_summary.json`: git commit, split SHA256, manifest SHA256, seed, fold,
class count, selection metric and value, monitoring-subset fingerprint, torch
version, device, AMP flag. Checkpoint SHA256 is recorded at evaluation time.

| | |
| --- | --- |
| Production code | tag `segresnet-production-v1`, commit `afdb75f3` |
| Split | `pants_cv_v1.json`, SHA256 `a4559f41…24ea91` |
| Manifest | SHA256 `f45e5b42…8faf15` |
| Cache contract | RAS, 1.5 mm, `[-175,250]→[0,1]`, float16 image / uint8 label, 29 classes |
| Seed | 317 |

**Experiment reproducibility** — fixing code, split, preprocessing, seed and
hyperparameters gives a controlled repeat. It does **not** promise bit-for-bit
identical weights across different GPUs, CUDA or cuDNN versions: algorithm
selection, reduction order and TF32 behaviour differ by hardware.

**Inference reproducibility** — given the same checkpoint and fixed
sliding-window settings, inference is deterministic within one software/hardware
stack. Across stacks small floating-point differences can appear; the semantic
contract (shape, affine, orientation, integer labels 0–28, finite probabilities
in [0,1]) is identical everywhere.

**Hardware is not held constant between the two arms.** `segresnet_random` is
training on a Colab Tesla T4; `segresnet_suprem` is planned for a local RTX 4070
Laptop. Each run records its own GPU, VRAM, Python, PyTorch, MONAI and CUDA
versions. The scientific protocol is controlled; the execution stack is
documented rather than claimed identical.

## Results

**Production training is in progress. No performance numbers are reported yet.**
They will be added from the actual `evaluation_summary.json` files once both arms
and the full 1,801-case fold-0 evaluation have completed. Nothing here is a
placeholder, a projection, or an illustrative value.

## Status

Complete: portable data root; validated 9,000-case manifest; fixed split; shared
preprocessing with verified tumor-aware sampling; the portable prepared cache and
its transport; strict SuPreM transfer; one trainer for both arms; checkpoint and
resume; whole-volume inference with source-geometry restoration; the full-fold
evaluator and the analysis figures; the test suite.

In progress: the fold-0 production runs, 64 epochs each.

Not started: full 1,801-case evaluation, any PanTS-te evaluation, the external
submission package.

nnU-Net is **paused, not abandoned** — the dataset definition, the 9,000-case
fingerprint and the `3d_fullres` plan are complete and preserved; production
preprocessing resumes after the SegResNet study.

## Data rule

Never modify anything under the PanTS data root. Develop on PanTS-tr; PanTS-te
stays locked until the pipeline is frozen.

## References

- PanTS: Li et al., NeurIPS 2025 Datasets and Benchmarks Track.
- SuPreM: Li et al., [arXiv:2501.11253](https://arxiv.org/abs/2501.11253).
- nnU-Net: Isensee et al., *Nature Methods* (2021).
- MONAI: Medical Open Network for AI.
