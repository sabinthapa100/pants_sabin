# PanTS SegResNet Study

## 1. Purpose

Does supervised 3D pretraining help pancreatic lesion segmentation?

Two identical 3D SegResNets are trained on PanTS, one from random initialization
and one from the SuPreM supervised abdominal-CT checkpoint. Architecture, split,
preprocessing, sampling, loss, optimizer, schedule and every hyperparameter are
held fixed; **initialization is the intended intervention**. The comparison, not
a leaderboard number, is the point of the study.

A third arm — nnU-Net v2 `3d_fullres` as an independent baseline — is **paused**.
Its dataset definition, 9,000-case fingerprint and plan are complete and
preserved.

## 2. Dataset and split policy

[PanTS](https://github.com/MrGiovanni/PanTS): `ImageTr/<case>/ct.nii.gz` and
`LabelTr/<case>/combined_labels.nii.gz`.

| | |
| --- | --- |
| PanTS-tr | 9,000 cases — 882 lesion-positive, 8,118 lesion-negative |
| PanTS-te | `PanTS_00009001`+ — **locked, never read** (see §12) |
| Classes | 29 = background + 28 structures; class 28 is `pancreatic_lesion` |
| Split | `pants_cv_v1.json` — tracked, seed 317, 5 folds stratified by class-28 presence |
| Fold 0 | 7,199 train / 1,801 validation, of which 177 are lesion-positive |

`lesion_present` is defined **only** by class 28 appearing in
`combined_labels`. The metadata spreadsheet's `tumor?` column is carried for QC
and never used as ground truth; the two disagree on 44 cases.

The split is **case-level, not patient-level**: the released metadata contains no
patient identifier, so patient-level grouping is impossible. Case IDs above
`PanTS_00009000` are refused by the evaluator unless `--allow-test-split` is
passed explicitly.

## 3. Model

MONAI 3D SegResNet — `blocks_down=[1,2,2,4]`, `blocks_up=[1,1,1]`,
`init_filters=16`, 1 input channel, **29 output classes**, GroupNorm.

SuPreM transfer (`supervised_suprem_segresnet_2100.pth`, SHA256 `2db81dc0…89a3`):
all 83 parameter names align. **81 tensors transfer.** Only
`conv_final.2.conv.{weight,bias}` is excluded — it maps features to class logits,
and SuPreM predicted 32 classes where PanTS needs 29. The loader raises unless
exactly 81 load, because a quietly half-initialized backbone would invalidate the
comparison.

## 4. Preprocessing

| Step | Choice |
| --- | --- |
| Orientation | `RAS` (seven source orientations occur in PanTS-tr) |
| Spacing | 1.5 mm isotropic (native z-spacing spans 0.4–10 mm) |
| Intensity | `[-175, 250] HU → [0,1]`, clipped — the SuPreM pretraining domain |
| Patch | 96³ = 144 mm per side |
| Sampling | tumor-aware, other : pancreas : lesion = 1 : 2 : 3 |
| Augmentation | intensity only |

Labels always resample with **nearest-neighbour**; linear interpolation would
invent fractional class IDs such as 17.4. No spatial flips: PanTS has
laterality-paired labels, so mirroring would mislabel a flipped kidney.

Evidence behind each choice — the 1.5 mm lesion-preservation study, the
degenerate label-affine repair, the measured sampler behaviour — is in
[TECHNICAL_NOTES.md](TECHNICAL_NOTES.md).

## 5. Training protocol

Identical for both arms: 64 epochs, batch 2 cases × 2 patches, accumulation 1,
AdamW lr 1e-4 / weight decay 1e-5, cosine annealing `T_max=64`, AMP, seed 317,
`DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)`.
3,600 iterations per epoch, **230,400 optimizer updates** total.

`best.pt` is selected by **deterministic whole-volume** mean class-28 Dice over a
fixed monitoring subset (all 177 fold-0 lesion-positive cases + 50 stride-sampled
negatives = 227), evaluated every 5 epochs. Per-epoch patch-validation loss is
diagnostic only and selects nothing. **Both arms selected epoch 59.**

```bash
python scripts/train_segresnet.py --initialization suprem \
  --experiment segresnet_suprem --pretrained-checkpoint $SUPREM_CHECKPOINT \
  --prepared-root <prepared-root> --manifest <prepared-root>/manifest.json \
  --split pants_cv_v1.json --fold 0 --epochs 64 \
  --batch-size 2 --samples-per-case 2 --accumulation 1 \
  --learning-rate 1e-4 --weight-decay 1e-5 --num-workers 4 --seed 317
```

## 6. Development result — fold 0, 1,801 cases

**These are INTERNAL DEVELOPMENT quantities on PanTS-tr fold 0.**

| | Random | SuPreM | SuPreM + frozen rule |
| --- | ---: | ---: | ---: |
| mean class-28 Dice (177 positives) | 0.1614 | 0.2440 | **0.2437** |
| positive overlap (C) / 177 | 60 | 95 | **90** |
| predicted but zero overlap (B) | 3 | 19 | **10** |
| no prediction (A) | 114 | 63 | **77** |
| false-positive patients / 1,624 | 7 | 201 | **109** |
| internal specificity | 0.9957 | 0.8762 | **0.9329** |
| macro anatomy Dice (classes 1–27) | 0.6852 | 0.6917 | **0.6917** |

**SuPreM raises lesion sensitivity substantially and costs specificity.** Mean
Dice rises 51%, positive-overlap cases go 60 → 95, and small-lesion detection
rises roughly fivefold — while false-positive patients rise 7 → 201. The frozen
component rule recovers most of that specificity (201 → 109) for 5 overlap cases,
all of which had Dice below 0.036, and leaves anatomy unchanged to 7 decimals.

**What these numbers are not.** They are not official PanTS **P-Sen**, **T-Sen**,
**Spe**, **AUC** or benchmark **DSC**, and must not be compared to the published
PanTS benchmark table. The PanTS repository contains no evaluator; patient-wise
and tumor-wise sensitivity are defined in prose only, with no overlap threshold,
component-matching rule, or patient-scoring convention published. "Detection"
here means our own criterion — any predicted class-28 voxel. Empty-versus-empty
Dice returns NaN rather than 1.0, and lesion Dice is averaged over
lesion-positive cases only.

## 7. Frozen final inference rule

```text
model            SuPreM SegResNet, best.pt, epoch 59
                 SHA256 54bbcf0ceb530fd929d352be11bc8d7b18d22c3925deb62d54fa3d6cfb4cef50
preprocessing    RAS, 1.5 mm isotropic, HU [-175,250] → [0,1]
inference        96³ windows, overlap 0.5, gaussian blending, CPU stitching
hard prediction  argmax over 29 classes
lesion filter    26-connectivity components of class 28;
                 keep a component iff max class-28 softmax inside it >= 0.6
rejected voxels  argmax over channels 0..27  (NOT forced to background)
probability map  raw class-28 softmax, never postprocessed
```

The 0.6 threshold was selected from nine predeclared one-parameter candidates in
an offline fold-0 study (`scripts/study_component_filters.py`). **It is a softmax
threshold, not a calibrated 60% probability of malignancy**, and it is not
claimed to be globally optimal.

Rejected voxels take the best non-lesion class because this is an exclusive
29-class segmenter: forcing them to background would assert "outside body" in
tissue the model believes is pancreas. On the full fold, 57.6% of rejected voxels
became a real anatomical structure — a third pancreas-family — and only 42%
background.

Filtering is applied in the canonical 1.5 mm frame **before** inversion to source
geometry, because component identity and physical size are only well defined on
the isotropic grid the model actually saw.

**Two different defaults, deliberately.** `infer_segresnet.py` defaults to 0.6 —
it is the deployment pipeline. `evaluate_segresnet.py` defaults to *no filtering*
— it is a research instrument that must keep reproducing the unfiltered baseline.
**The held-out evaluation must pass `--lesion-peak-probability 0.6` explicitly
and must not rely on a default.**

## 8. Inference on an unseen CT

```bash
python scripts/infer_segresnet.py \
    --checkpoint PanTS_run/segresnet_suprem/best.pt \
    --input scan.nii.gz \
    --output predictions/ \
    --lesion-probability
```

This is the submission pipeline: the frozen 0.6 rule is the default. The explicit
equivalent, preferred when the rule must be visible in the command record:

```bash
python scripts/infer_segresnet.py \
    --checkpoint PanTS_run/segresnet_suprem/best.pt \
    --input scan.nii.gz --output predictions/ \
    --lesion-peak-probability 0.6 --lesion-probability
```

Needs **only** this repository, a checkpoint, and a raw CT — no labels, manifest,
split, prepared cache, SuPreM pretraining file, or PanTS naming convention.
`--input` also accepts a directory of NIfTIs.

## 9. Outputs

```text
combined_labels.nii.gz                  uint8, integer labels 0..28
pancreatic_lesion_probability.nii.gz    float32 in [0,1]  (with --lesion-probability)
```

Outputs are restored to the source CT shape, affine and orientation. Measured on
one 493×282×117 CT: 15 s wall clock, 0.45 GB peak VRAM, 3.4 GB peak host RAM.

The probability map is the raw model output and is deliberately *not* filtered,
so whoever holds the official protocol can compute their own operating points.

## 10. Reproducibility and environment

Tested: Python 3.11.15, PyTorch 2.13.0+cu126, MONAI 1.5.1, NumPy 2.4.6,
SciPy 1.17.1, nibabel 5.4.2, NVIDIA RTX 4070 Laptop (8 GB).

Install PyTorch first, matched to your GPU and CUDA runtime — this repository
deliberately does not pin a CUDA build.

```bash
pip install -r requirements.txt
export PANTS_DATA_ROOT=/path/to/PanTS/data
python scripts/prepare_data.py --manifest --split --workers 8
python scripts/prepare_data.py --prepare-segresnet --output <prepared-root> --workers 4
python -m pytest tests/ -q
```

Every checkpoint and every `evaluation_summary.json` records git commit, split
and manifest SHA256, seed, fold, selection metric and value, monitoring-subset
fingerprint, sliding-window settings, torch version and device. Checkpoint SHA256
is recorded at evaluation time.

Fixing code, split, preprocessing, seed and hyperparameters gives a controlled
repeat; it does **not** promise bit-identical weights across GPUs, CUDA or cuDNN
versions. The semantic output contract — shape, affine, orientation, integer
labels 0–28, finite probabilities in [0,1] — holds everywhere.

## 11. Limitations

- **Not a hardware-controlled ablation.** Random trained on a Colab Tesla T4
  (torch 2.11.0+cu128), SuPreM on a local RTX 4070 (torch 2.13.0+cu126).
  The scientific protocol is controlled; the execution stack is recorded, not
  claimed identical. Evaluation hardware and code were identical for both arms.
- **The postprocessing threshold was tuned on fold 0**, the same fold that
  selected the checkpoint, so fold-0 numbers are optimistic by an unmeasured
  amount. Nine predeclared candidates on one axis is a small search, but not zero.
- **Case-level split**, because no patient identifier is released. If a patient
  contributed two scans they could fall on both sides of the split.
- **1.5 mm resampling costs small lesions.** The smallest lesions fall to ~46
  voxels; small-lesion Dice remains near 0.04 and is the weakest part of the model.
- **Single fold, single seed.** No cross-fold variance or seed variance is measured.
- **Class 21 (`pancreatic_duct`) scores 0.0** and classes 9/10 (femurs) score
  below 0.2 — the intensity window saturates bone, an accepted trade for matching
  SuPreM's pretraining domain.
- **No calibration.** Softmax values are relative class preferences, not
  calibrated probabilities.

## 12. PanTS-te status

**PanTS-te has NOT been evaluated. No held-out results exist in this repository.**

No file under `ImageTe/` or `LabelTe/` has been opened. The raw held-out
evaluation path is implemented and tested against synthetic NIfTI fixtures only
(`tests/test_raw_evaluation.py`), which prove `--data-split test` resolves
`ImageTe`/`LabelTe` without touching the real dataset.

The one-shot held-out evaluation will run only after the checkpoint,
preprocessing, postprocessing rule, inference implementation and metric
definitions are frozen, committed, and tagged — and never more than once.

## References

- PanTS: Li et al., NeurIPS 2025 Datasets and Benchmarks Track.
- SuPreM: Li et al., [arXiv:2501.11253](https://arxiv.org/abs/2501.11253).
- nnU-Net: Isensee et al., *Nature Methods* (2021).
- MONAI: Medical Open Network for AI.
