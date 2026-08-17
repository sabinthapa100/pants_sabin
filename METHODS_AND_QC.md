# Methods and QC

Evidence behind the preprocessing decisions summarised in the README. Kept
separate because it is needed to *justify* the pipeline, not to *use* it.

## 0. Data provenance and label ontology

The image and label files used in this study came from the **PanTSMini**
distribution (<https://huggingface.co/datasets/BodyMaps/PanTSMini>), fetched with
the upstream `download_PanTS_data.sh`. All credit for the dataset belongs to the
PanTS project (<https://github.com/MrGiovanni/PanTS>).

`combined_labels.nii.gz` is an **exclusive** semantic map assembled from
per-structure annotations that can otherwise be overlapping or hierarchical
(`LabelTr/<case>/segmentations/` ships those separately). A voxel can therefore
carry only one identifier even where two annotations legitimately overlap.
Consequently **class 17 should not automatically be interpreted as whole-pancreas
Dice**: whatever the assembly rule assigns to the head, body, tail and duct
classes is unavailable to class 17.

## 1. Why 1.5 mm isotropic

PanTS-tr in-plane spacing is ~0.8 mm, but z-spacing spans **0.4–10 mm** and only
18% of cases are isotropic. At native spacing a fixed 96³ patch would span 38 mm
to 960 mm along z depending on the case, so resampling is mandatory. (SuPreM's
own pancreas application does no resampling; that does not transfer here.)

### Lesion-preservation QC

Before locking the spacing, the real label path (affine repair → RAS → 1.5 mm
nearest-neighbour) was run over all 882 lesion-positive cases:

| Quantity | Result |
| --- | --- |
| Lesion-positive before resampling | 882 |
| Lesion-positive after resampling | **880** |
| Lesions becoming class-28 empty | **2** (0.23%) |
| Lesion voxels, median | 3,166 → 1,382 |
| Lesion **physical volume**, median | 4,614 → 4,664 mm³ |
| Relative volume error, median | **0.000** |
| Relative volume error, mean absolute | 0.037 |

The two lost lesions are `PanTS_00005043` (4 voxels, 1.8 mm³) and
`PanTS_00000044` (16 voxels, 7.1 mm³) — roughly 1.5 mm and 2.4 mm across. Both
lesions were below the effective resolution of the 1.5-mm representation and were
vulnerable to disappearance during resampling. Physical
volume is preserved essentially exactly at the median, which is the reassuring
part: nearest-neighbour resampling rescales the voxel count in proportion to the
voxel-size ratio without systematically eroding or dilating the structure.

**Residual exposure:** 14 lesions (1.6%) end below a 3³ voxel cube and 44 (5.0%)
below 100 voxels. Small-tumor sensitivity is the metric most exposed to this
choice, and 1.0 mm isotropic is the first thing to revisit if it underperforms.
The 2 emptied cases remain `lesion_present=True` in the manifest but have an
empty target after preprocessing, so they act as negatives for the SegResNet
arms.

## 2. Why `[-175, 250]` HU

This is the input domain of the SuPreM **supervised-pretraining** pipeline that
produced the checkpoint we initialize from, so transfer is tested on its own
terms. The narrower `[-100, 200]` window in the SuPreM repository belongs to a
later JHH pancreatic downstream experiment, not to pretraining. The wider window
also preserves more contrast for the bone and lung context classes.

## 3. Source orientations

Seven distinct orientations occur across PanTS-tr: `LAI` 4,291, `RAS` 1,916,
`LPS` 1,381, `LAS` 989, `IPL` 417, `PRS` 4, `ALS` 2. Canonicalizing to RAS is
required, not cosmetic.

## 4. Degenerate label affines

In **80 of 9,000** PanTS-tr cases the `combined_labels.nii.gz` affine is
degenerate — translation zeroed, direction reset to LPS — while the CT carries a
correct affine. In 50 of those the axis codes differ outright. Shape and voxel
spacing agree in all 9,000 cases, so the arrays correspond one-to-one by index.

Left uncorrected, `Orientationd` would reorient image and label by *different*
affines and silently mirror the annotation on those cases.

### Evidence that the voxel arrays, not the affines, are authoritative

HU plausibility motivated the hypothesis: index-aligned labels put liver at
15–143 HU and lung near −761 HU in 100% of the affected cases, whereas the
mirrored interpretation gives a liver median of −113 HU and places femur in air.

The repair is justified by the annotations themselves. For all 80 affected cases,
`combined_labels` was compared against the standalone masks in
`LabelTr/<case>/segmentations/` in voxel-index space. Because `combined_labels`
is an *exclusive* semantic map while standalone masks may overlap, the correct
test is containment, not equality.

| Check | Result |
| --- | --- |
| CT vs combined shape mismatches | 0 / 80 |
| Class comparisons performed | 1,852 |
| Fully contained in the standalone mask | **1,852 / 1,852** |
| Containment violations | 0 |
| Cases with ≥1 standalone mask whose affine equals the CT affine | **80 / 80** |

The last row closes the chain: each affected case has at least one standalone
mask carrying the CT's own affine, and `combined_labels` is voxel-identical with
it. The combined-label arrays are therefore index-aligned with the CT and only
the NIfTI affine metadata is defective.

`AlignLabelGeometryd` copies the CT affine onto the label before any
reorientation. It is a no-op for the other 8,920 cases and is applied
unconditionally. Many standalone masks share the same defect (1,039 of 1,852
comparisons), so this is a systematic export artifact rather than per-file
corruption.

## 5. Tumor-aware sampling, measured

Generic foreground sampling is unsuitable: PanTS foreground includes femur, lung
and spleen, so a foreground-centred crop usually contains no pancreas. At 1.5 mm
the median lesion occupies ~865 voxels of ~6.8 M — about **0.013%**.

A three-value sampling map (0 = other, 1 = pancreas family and duct, 2 = lesion)
drives `RandCropByLabelClassesd` with ratios `1:2:3`. On a lesion-positive case
that is 50% tumor-centred, 33% pancreas, 17% general anatomy; on a lesion-negative
case MONAI zeroes the absent class and renormalizes to 67% / 33%.

Verified by drawing 400 real crops from each group:

| Group | centre = lesion | centre = pancreas | centre = other | patches containing lesion |
| --- | --- | --- | --- | --- |
| Lesion-positive | 26.0% | 32.8% | 41.2% | **75.8%** |
| Lesion-negative | 0.0% | 40.5% | 59.5% | 0.0% |

The pancreas fraction matches the design (32.8% vs 33%). The lesion-centre
fraction reads lower than the nominal 50% for a benign reason: MONAI clamps a
crop centre so the patch fits inside the volume, and 36% of lesion voxels lie
outside that valid centre band, mostly in volumes thinner than 96 voxels along z.
`SpatialPadd` has already padded those to 96, so the patch still covers the whole
axis and still contains the lesion — which is why three quarters of positive-case
patches carry tumor. The centre-voxel statistic is a lower bound, not a defect.

## 6. Split honesty

The released PanTS metadata exposes **no patient or study identifier** — all
9,901 `PanTS ID` values are unique, one per scan. Patient-level grouping cannot
be enforced, so the split is case-level, and this is recorded in the split file
rather than left implicit. The `site` column was considered and rejected: its
values (`"1 Site"`, `"15 Sites"`, `"I"`, …, plus 594 blanks) describe
source-cohort aggregation, not individual institutions.

## 7. Whole-volume inference memory

Patches execute on the GPU while the stitched volume accumulates on the **host**
(`sw_device` vs `device` in MONAI's sliding-window inferer). A 512×512×300 volume
at 29 classes in float32 is about 9 GB, which previously exhausted VRAM. Measured
peak on an 8 GB RTX 4070 Laptop: **0.45 GB** with CPU stitching.

Labels are inverted with nearest-neighbour and the probability map with linear.
Inverting labels linearly produces fractional class identifiers — measured: 5.6
million distinct values instead of 29.

Mean sliding-window geometry over real prepared volumes: ~56 windows per case at
96³ with 0.5 overlap, moving ~5.3 GiB device-to-host per case when accumulating
on CPU. That transfer, not GPU compute, dominates whole-volume validation cost.
