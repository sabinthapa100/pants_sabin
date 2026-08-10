# PanTS 3D Pancreatic Tumor Segmentation

A reproducible study of 3D pancreatic-tumor segmentation on the
[PanTS dataset](https://github.com/MrGiovanni/PanTS) (9,000 training CTs,
background plus 28 anatomical classes, pancreatic lesion is class 28).

## Study design

Three experiments share one data layer, one case split, and one evaluation
implementation:

| Experiment | Model | Initialization | Role |
| --- | --- | --- | --- |
| `segresnet_suprem` | 3D SegResNet | official SuPreM checkpoint | transfer learning |
| `segresnet_random` | the **same** 3D SegResNet | random | controlled ablation |
| `nnunet3d` | nnU-Net v2 `3d_fullres` | framework default | independent baseline |

The two SegResNet arms are identical in architecture, preprocessing, sampling,
optimizer, patch size, and split. **Initialization is the only variable**, which
is what makes the effect of supervised 3D pretraining measurable rather than
assumed.

nnU-Net is deliberately *not* forced through our preprocessing: it is
self-configuring, and overriding its fingerprint would remove the very thing
that makes it a strong independent baseline. It shares the **case partition**
only.

## Repository layout

```text
pants_sabin/
├── pants_cv_v1.json         # authoritative fixed 5-fold split (tracked, defines the experiment)
├── src/
│   ├── data/                # paths, labels, I/O, QC, manifest, transforms, prepared cache
│   ├── models/              # SegResNet + SuPreM transfer
│   ├── training/            # trainer + resumable checkpoints
│   └── evaluation/          # metrics and inference primitives
├── scripts/                 # thin command-line entry points
├── notebooks/               # Colab orchestration only
└── tests/
```

Generated data never lives in the repository. The prepared training cache, the
case manifest and every nnU-Net artifact are written to gitignored paths and
are rebuildable from the raw NIfTIs plus this code.

## Setup

Install PyTorch first, matched to your GPU and CUDA runtime, then:

```bash
pip install -r requirements.txt
```

Verified combination: Python 3.11.15, PyTorch 2.13.0+cu126, MONAI 1.5.1,
nnU-Net v2 2.8.1.

## Data root

All code resolves the dataset through one variable, in this order: an explicit
function argument, then `PANTS_DATA_ROOT`, then `PROJECT_ROOT/PanTS/data`.
No Google Drive path appears anywhere in reusable code.

```bash
export PANTS_DATA_ROOT=/path/to/PanTS/data
```

The environment is re-read on every call, so a notebook that sets the variable
after importing the package still works.

Expected layout: `ImageTr/<case>/ct.nii.gz` and
`LabelTr/<case>/combined_labels.nii.gz`. PanTS-tr is `PanTS_00000001` to
`PanTS_00009000`; PanTS-te starts at `PanTS_00009001` and is **never** read
during development.

## Manifest and split

```bash
python scripts/prepare_data.py --manifest --split --workers 8
```

This writes `pants_tr_manifest.json` (one row per case: relative paths, shape,
spacing, orientation, lesion presence and voxel count) and `pants_cv_v1.json`
(deterministic 5-fold, seed 317, stratified by class-28 presence).

The manifest is **derived** — exactly reproducible from the raw data by the
command above — so it is gitignored. The split is **source truth** for the
experiment and is tracked.

Manifest paths are relative to `PANTS_DATA_ROOT`, so the same file is valid on
every machine. The build fails loudly on a missing file, a shape or spacing
disagreement, a label value outside 0–28, a wrong case count, or any PanTS-te
identifier.

Measured result: **882 lesion-positive and 8,118 lesion-negative** cases.

`lesion_present` is defined **only** by the presence of class 28 in
`combined_labels.nii.gz`. The metadata spreadsheet's `tumor?` column is
recorded alongside it for QC but is never treated as ground truth. The two
disagree on **44 cases**, all in the same direction: metadata marks the patient
as tumor-positive while `combined_labels` contains no class-28 voxel. Those
cases are treated as lesion-negative for stratification, because a voxel-level
segmentation task can only learn from voxel-level annotation.

To give nnU-Net the identical partition:

```bash
python scripts/prepare_data.py --emit-nnunet-splits \
  "$nnUNet_preprocessed/Dataset500_PanTS/splits_final.json"
```

### Split honesty

The released PanTS metadata exposes **no patient or study identifier** — all
9,901 `PanTS ID` values are unique, one per scan. Patient-level grouping
therefore cannot be enforced, and the split is case-level. This is recorded in
the split file rather than left implicit. The `site` column was considered and
rejected: its values (`"1 Site"`, `"15 Sites"`, `"I"`, …, plus 594 blanks)
describe source-cohort aggregation, not individual institutions.

## Preprocessing

Shared byte-for-byte by both SegResNet arms (`src/data/transforms.py`):

| Step | Choice |
| --- | --- |
| Orientation | canonical `RAS` |
| Target spacing | **1.5 mm isotropic** |
| Patch size | **96³** (= 144 mm per side) |
| Intensity | `[-175, 250] HU → [0, 1]`, clipped |
| Augmentation | intensity only (shift, noise, smoothing) |

Each choice was measured, not inherited:

- PanTS-tr z-spacing spans **0.4–10 mm** and only 18% of cases are isotropic,
  so training on native grids would make a fixed patch cover wildly different
  anatomy per case. Resampling is mandatory. (SuPreM's own pancreas
  application does no resampling, which does not transfer to this dataset.)
- 96³ at 1.5 mm covers 144 mm per side — the whole pancreas plus
  peripancreatic vessels in one window, which suits the 28-class context task.
- `[-175, 250] HU` is the input domain of the SuPreM **supervised-pretraining**
  pipeline that produced the checkpoint we initialize from, so transfer is
  tested on its own terms. (The narrower `[-100, 200]` window belongs to a
  later JHH pancreatic downstream experiment, not to pretraining.) The wider
  window also preserves more contrast for the bone and lung context classes.
- Seven distinct source orientations occur across PanTS-tr (`LAI` 4,291,
  `RAS` 1,916, `LPS` 1,381, `LAS` 989, `IPL` 417, `PRS` 4, `ALS` 2), so
  canonicalizing to `RAS` is required, not cosmetic.

### 1.5 mm lesion-preservation QC

Before locking the spacing we ran the **real** label path
(affine repair → `RAS` → 1.5 mm nearest-neighbour) over every one of the 882
lesion-positive cases and compared the lesion before and after:

| Quantity | Result |
| --- | --- |
| Lesion-positive before resampling | 882 |
| Lesion-positive after resampling | **880** |
| Lesions becoming class-28 empty | **2** (0.23%) |
| Lesion voxels, median | 3,166 → 1,382 |
| Lesion **physical volume**, median | 4,614 → 4,664 mm³ |
| Relative volume error, median | **0.000** |
| Relative volume error, mean absolute | 0.037 |

The two lost lesions are `PanTS_00005043` (4 voxels, **1.8 mm³**) and
`PanTS_00000044` (16 voxels, **7.1 mm³**) — roughly 1.5 mm and 2.4 mm across,
well below the size at which a pancreatic lesion is clinically actionable or
reliably visible on CT. No clinically meaningful set of lesions disappears, so
**1.5 mm is kept**.

Physical volume is preserved essentially exactly at the median, which is the
reassuring part: nearest-neighbour resampling changes the voxel count roughly in
proportion to the voxel-size ratio without systematically eroding or dilating
the structure.

Residual exposure: 14 lesions (1.6%) end below a 3³ voxel cube and 44 (5.0%)
below 100 voxels. Small-tumor sensitivity is the metric most exposed to this
choice, and 1.0 mm isotropic is the first thing to revisit if it underperforms.
Note also that those 2 cases are `lesion_present=True` in the manifest but have
an empty target after preprocessing; they are effectively negatives for the
SegResNet arms.

**Reorientation and resampling are different operations.** `Orientationd`
permutes and flips axes so they align with Right/Anterior/Superior — pure index
bookkeeping, lossless. `Spacingd` changes the physical sampling lattice and must
interpolate. Labels are always resampled with nearest-neighbour; linear
interpolation would invent fractional class IDs such as 17.4.

### No spatial flips

PanTS contains four laterality-paired class sets (adrenal 1/2, femur 9/10,
kidney 12/13, lung 15/16). A left–right mirror leaves a flipped left kidney
still labelled `kidney_left`, which is inconsistent supervision. Correcting it
would require paired-label swapping, so flips and 90° rotations are omitted and
only intensity augmentation is used.

### Tumor-aware sampling

Generic foreground sampling is unsuitable here: PanTS foreground includes femur,
lung and spleen, so a foreground-centred crop usually contains no pancreas. At
1.5 mm the median lesion occupies ~865 voxels out of ~6.8 M — about **0.013%** —
so uniform random crops would almost never see a tumor.

A three-value sampling map (0 = other, 1 = pancreas family and duct,
2 = lesion) drives `RandCropByLabelClassesd` with ratios `1 : 2 : 3`. On a
lesion-positive case that is 50% tumor-centred, 33% pancreas, 17% general
anatomy; on a lesion-negative case MONAI zeroes the absent class and
renormalizes to 67% / 33%. Tumor crops supply detection signal, pancreas-only
crops teach the model not to fire on normal pancreas, and general crops preserve
the 28-class context task.

**Verified empirically** by drawing 400 real crops from each group:

| Group | centre = lesion | centre = pancreas | centre = other | patches containing lesion |
| --- | --- | --- | --- | --- |
| Lesion-positive | 26.0% | 32.8% | 41.2% | **75.8%** |
| Lesion-negative | 0.0% | 40.5% | 59.5% | 0.0% |

The pancreas fraction matches the design (32.8% vs 33%) and lesion-negative
cases degrade exactly as intended. The lesion centre fraction reads lower than
the nominal 50% for a benign reason: MONAI clamps a crop centre so the patch
fits inside the volume, and 36% of lesion voxels lie outside that valid centre
band, mostly in volumes thinner than 96 voxels along z. `SpatialPadd` has
already padded those to 96, so the patch still covers the whole axis and still
contains the lesion — which is why three quarters of positive-case patches carry
tumor. The centre-voxel statistic is a lower bound, not a defect; the sampler
was left unchanged.

## Known dataset quirk: degenerate label affines

In **80 of the 9,000** PanTS-tr cases the `combined_labels.nii.gz` affine is
degenerate — translation zeroed and direction reset to LPS — while the CT
carries a correct affine. In 50 of those the axis codes differ outright. Shape
and voxel spacing agree in **all 9,000** cases, so the arrays correspond
one-to-one by index.

Left uncorrected, `Orientationd` would reorient image and label by *different*
affines and silently mirror the annotation on those cases. Which array is
authoritative was determined empirically, not assumed: index-aligned labels put
liver at 15–143 HU and lung near −761 HU in **100%** of the affected cases,
whereas the mirrored interpretation gives a liver median of −113 HU, places
femur in air, and is plausible for only 6% of them.

`AlignLabelGeometryd` therefore copies the CT affine onto the label before any
reorientation. It is a no-op for the other 8,920 cases and is applied
unconditionally. The manifest records `label_affine_matches_ct` per case, and
`tests/test_manifest.py` asserts that the correction actually prevents the
mirroring.

### Direct annotation evidence for the repair

HU plausibility motivated the hypothesis; the repair is justified by the
annotations themselves. For all **80** affected cases we compared
`combined_labels` against the standalone masks in `LabelTr/<case>/segmentations/`
in voxel-index space. Because `combined_labels` is an *exclusive* semantic map
while standalone masks may overlap, the correct test is containment, not
equality: every voxel labelled class *c* must lie inside standalone mask *c*.

| Check | Result |
| --- | --- |
| CT vs combined shape mismatches | 0 / 80 |
| Class comparisons performed | 1,852 |
| Fully contained in the standalone mask | **1,852 / 1,852** |
| Containment violations | 0 |
| Standalone files absent (tolerated) | 0 |
| Cases with ≥1 standalone mask whose affine equals the CT affine | **80 / 80** |

The last row closes the chain. Each affected case has at least one standalone
mask carrying the CT's own affine — that mask is therefore registered to the CT
grid — and `combined_labels` is voxel-identical with it. Conclusion:

> The combined-label voxel arrays are index-aligned with the CT; only their
> NIfTI affine metadata is defective. Assigning CT geometry before common
> orientation and resampling is therefore justified.

Note that many standalone masks share the same defect (1,039 of 1,852
comparisons had a CT-matching affine), so this is a systematic export artifact
rather than per-file corruption.

## SuPreM transfer

```bash
export SUPREM_CHECKPOINT=/path/to/supervised_suprem_segresnet_2100.pth
```

Source: <https://huggingface.co/MrGiovanni/SuPreM>, 56,500,623 bytes,
SHA256 `2db81dc05cd9ea7234ca75e921e53e32b8716dc4cba88a6710742bfc282589a3`
(verified locally against the published object hash).

The architecture matches the SuPreM backbone exactly — all 83 parameter names
align. Only the class-specific output convolution differs, because SuPreM
predicted 32 classes and PanTS needs 29.

**81 of 83 parameters transfer.** Only `conv_final.2.conv.{weight,bias}` is
excluded and left randomly initialized. `conv_final.0` is a task-independent
GroupNorm and *is* transferred — excluding the whole `conv_final` block would
needlessly discard it. The loader reports expected, loaded, excluded, missing,
unexpected, and shape-mismatched keys, and **raises** if the backbone transfers
only partially, since a quietly half-initialized backbone would invalidate the
comparison.

## Training

One trainer, `src/training/trainer.py`, runs both arms. Nothing branches on the
initialization except model construction, which is what keeps the comparison
controlled.

```bash
python scripts/train_segresnet.py --initialization random --epochs 3
python scripts/train_segresnet.py --initialization suprem --epochs 3   # needs $SUPREM_CHECKPOINT
```

Loss is `DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)`
with equal Dice and cross-entropy weight. Background is excluded from the Dice
term because it occupies most of every patch and would otherwise dominate the
overlap signal; cross-entropy still sees it, so background remains supervised.
Optimizer is AdamW (lr 1e-4, weight decay 1e-5) with cosine annealing and
gradient accumulation of 4. AMP is enabled automatically on CUDA and disabled
on CPU.

### Checkpoints and resume

Colab sessions disconnect, so resume is a correctness requirement rather than a
convenience. `latest.pt` is written periodically and `best.pt` on validation
improvement; each stores model, optimizer, scheduler, AMP scaler, epoch, global
step, best metric, resolved config, Git commit, and Python/NumPy/Torch/CUDA RNG
state, via an atomic temp-file rename.

`tests/test_resume.py` proves the round trip: weights, optimizer, scheduler and
counters are restored exactly, and training continues with a finite loss.
`fit()` deliberately does **not** call `manual_seed` — doing so would overwrite
the RNG state just restored and silently change the data stream after a resume.

## Whole-volume inference

```python
predict_case_in_source_geometry(model, ct_path, want_lesion_probability=True)
```

Patches execute on the GPU while the stitched volume accumulates on the **host**
(`sw_device` vs `device` in MONAI's sliding-window inferer). This is not a
micro-optimization: a 512×512×300 volume at 29 classes in float32 is about 9 GB,
which previously exhausted VRAM. Measured peak on an 8 GB RTX 4070 Laptop:
**0.45 GB** with CPU stitching, versus 2.41 GB when accumulating on the GPU for
a much smaller volume. `sw_batch_size` starts at 1 and cases are processed one
at a time, with large tensors released before returning.

The prediction is then mapped back onto the source CT grid. Labels are inverted
with **nearest-neighbour** interpolation and the probability map with linear —
inverting labels linearly produces fractional class identifiers (measured: 5.6
million distinct values instead of 29). `tests/test_geometry_roundtrip.py`
verifies shape, affine, orientation and value range against the source CT for
`RAS`, `LPS`, `LAI` and `IPL` sources.

## Evaluation

The PanTS benchmark reports P-Sen, T-Sen, specificity, AUC and DSC. The PanTS
repository contains **no evaluation code**, and the README defines patient-wise
and tumor-wise sensitivity only in prose — no overlap threshold, minimum lesion
size, connected-component rule, or AUC scoring definition is published.
Submission is by email to Dr. Zhou.

This repository therefore implements **internal development metrics only** and
does not label them official. Models expose the three outputs any future
protocol would need: full semantic labels 0–28, a class-28 mask, and a class-28
probability map.

Empty-versus-empty Dice returns NaN, not 1.0. Since **8,118 of 9,000** cases are
lesion-negative, scoring them as perfect and averaging would report near-perfect
tumor Dice for a model that never predicts a tumor. Lesion Dice is averaged over
lesion-positive cases only; lesion-negative cases are reported separately as a
false-positive rate. Connected-component filtering and size thresholds are
deliberately absent, since inventing them would amount to inventing the
unpublished matching protocol.

## Prepared training cache

The expensive deterministic preprocessing runs **once, locally**, and the result
is a compact portable dataset that both SegResNet arms share byte-for-byte.

```bash
python scripts/prepare_data.py --prepare-segresnet \
  --output /path/outside/the/repo/PanTS_prepared/segresnet --workers 4
```

Each case becomes one `np.savez_compressed` archive holding exactly two plain
arrays — `image` float16 `[D,H,W]` in [0,1] and `label` uint8 `[D,H,W]` with
values 0–28. No pickles, no objects, no affines: the cache is already in the
training coordinate system, and inference recovers source geometry from the
original NIfTI instead.

```text
<prepared-root>/
├── cases/PanTS_XXXXXXXX.npz
├── manifest.json
└── preprocessing.json      # orientation, spacing, window, dtypes, commit, hashes
```

Writes are atomic and power-loss safe: temporary file → `fsync` → reopen with
`allow_pickle=False` → validate both arrays → `os.replace` → `fsync` the
directory. Re-running skips completed cases, so an interrupted 9,000-case job
resumes instead of restarting.

`tests/test_prepared.py` asserts that the cache path and the raw-NIfTI path
produce the same patches from the same seed, so the two can never drift apart.

### Transport to Google Drive

`npz` is already compressed, so the shards are **uncompressed** `tar` — tar
aggregates files, it does not compress them. Bundling avoids thousands of tiny
Drive objects.

```bash
cd <prepared-root>
ls cases/*.npz | sort > /tmp/all && split -d -l <N> /tmp/all /tmp/shard_
for part in /tmp/shard_*; do
  tar --create --file "shards/segresnet_shard_${part##*_}.tar" \
      --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner --files-from "$part"
done
sha256sum shards/*.tar manifest.json preprocessing.json > SHA256SUMS
sha256sum -c SHA256SUMS

rclone copy . gdrive:PanTS_prepared/segresnet --exclude "cases/**" --dry-run
rclone copy . gdrive:PanTS_prepared/segresnet --exclude "cases/**" --transfers 8 --progress
rclone check . gdrive:PanTS_prepared/segresnet --exclude "cases/**"
```

The fixed metadata makes shards byte-reproducible. Re-running `rclone copy`
transfers only what is missing, which is how an interrupted upload resumes.

## Colab

`notebooks/PanTS_SegResNet_Colab.ipynb` is the cloud-training interface. It
orchestrates only — every model, transform and checkpoint comes from `src/` at
a pinned commit.

Drive is **persistent transport**. `/content` is the **ephemeral fast disk** that
training actually reads; a FUSE-mounted Drive cannot sustain the random reads a
dataloader issues. Per shard: copy → verify SHA256 → extract → delete the
archive → next, so peak disk is *cache + one shard* rather than *cache + all
archives*.

A disconnect destroys `/content`: the repo, the staged cache and any checkpoint
not yet copied to Drive. Recovery is to re-run the staging sections at the same
pinned commit and resume from `latest.pt`, **using the original `--epochs`** —
the cosine schedule stored in the checkpoint was built for that horizon.

> `notebooks/PanTS_nnUNet_Colab.ipynb` pins the older tag `nnunet-colab-v1`
> (commit `c659518`) and reproduces the earlier nnU-Net-only smoke experiment.
> It does **not** use the code described here.

## Commands

```bash
export PANTS_DATA_ROOT=/path/to/PanTS/data
export SUPREM_CHECKPOINT=/path/to/supervised_suprem_segresnet_2100.pth

# inspect one case
python scripts/inspect_data.py --case PanTS_00000003

# manifest + split
python scripts/prepare_data.py --manifest --split --workers 8

# prepared training cache: 100-case representative pilot, then all 9,000
python scripts/prepare_data.py --prepare-segresnet --pilot 100 --output <prepared-root> --workers 4
python scripts/prepare_data.py --prepare-segresnet --output <prepared-root> --workers 4

# nnU-Net production dataset (symlinks) + the same split in native format
python scripts/prepare_nnunet.py --production

# whole suite; `python -m pytest` puts the project root on sys.path
python -m pytest tests/ -q

# tiny-overfit pipeline check, both arms (~3 min)
python -m pytest tests/test_overfit.py -q -s -m slow

# training from the cache, both arms — identical but for --initialization
python scripts/train_segresnet.py --initialization random \
  --prepared-root <prepared-root> --manifest <prepared-root>/manifest.json --fold 0
python scripts/train_segresnet.py --initialization suprem --pretrained-checkpoint $SUPREM_CHECKPOINT \
  --prepared-root <prepared-root> --manifest <prepared-root>/manifest.json --fold 0

# resume after an interruption (use the SAME --epochs as the original run)
python scripts/train_segresnet.py --initialization random --epochs <original> \
  --resume outputs/runs/segresnet_random/latest.pt
```

## Status

Implemented: portable data root; authoritative 9,000-case manifest with
validation; fixed deterministic split shared by all three experiments; shared
preprocessing and verified tumor-aware sampling; the portable prepared cache and
its Drive transport; strict SuPreM transfer; one minimal trainer running both
arms from either the cache or raw NIfTI; proven checkpoint/resume; memory-safe
whole-volume inference with geometry restoration; internal evaluation metrics;
unit and integration tests.

Not yet implemented: production-scale training, any PanTS-te evaluation, and the
submission inference package.

### nnU-Net: paused, not abandoned

The nnU-Net production data definition, the full 9,000-case fingerprint and the
experiment planning are **complete**. Production preprocessing is **paused** and
will resume after the SegResNet/SuPreM study.

Preserved and unchanged: `src/data/nnunet.py`, `scripts/prepare_nnunet.py`, the
`Dataset500_PanTS` symlink dataset, `dataset_fingerprint.json`,
`nnUNetPlans.json`, and `splits_final.json` derived from `pants_cv_v1.json`.
The production `3d_fullres` plan is spacing `[1.25, 0.793, 0.793]`, patch
`[64, 160, 192]`, batch 2, `CTNormalization`, `NibabelIOWithReorient`. That plan
is not to be modified to fit any hardware budget.

### Engineering smoke run, not a result

A 3-epoch run over a 40-case lesion-balanced cohort (32 train / 8 val) on an
8 GB RTX 4070 Laptop:

| Arm | train loss | val loss | steps | runtime | peak VRAM |
| --- | --- | --- | --- | --- | --- |
| random | 4.119 → 3.894 | 3.979 → 3.913 | 96 | 79 s | 2.90 GB |
| SuPreM | 4.337 → 3.698 | 3.982 → **3.633** | 96 | 83 s | 2.90 GB |

Both arms train, checkpoint, resume, and predict whole volumes. After 3 epochs
neither predicts any lesion voxel (max class-28 probability ≈ 0.09), so internal
lesion Dice is 0.0 on positives and the false-positive rate is 0.0 on negatives.

**These numbers are engineering evidence only.** Three epochs on 32 cases is far
short of convergence for the rarest class, the cohort is tiny, and no
hyperparameter was tuned. They must not be read as a comparison between
initializations.

The `configs/*.yaml` files are declarative provenance records. No code reads
them yet; the trainer arrives in the next milestone.

## Data rule

Never modify anything under the PanTS data root. Develop on PanTS-tr; PanTS-te
stays locked until the pipeline is frozen.

## References

- PanTS: Li et al., NeurIPS 2025 Datasets and Benchmarks Track.
- SuPreM: Li et al., [arXiv:2501.11253](https://arxiv.org/abs/2501.11253).
- nnU-Net: Isensee et al., *Nature Methods* (2021).
- MONAI: Medical Open Network for AI.
