# PanTS Technical Submission

Sabin Thapa
Kent State University
<sthapa3@kent.edu>

Repository: <https://github.com/sabinthapa100/pants_sabin>
Final package tag: **`pants-submission-v2`**
Inference code tag: `pants-submission-v1` (`72813b28`)
Metric code tag: `pants-metrics-v1` (`b3fa3a9e`)
Checkpoint: `best.pt`, SHA256 `54bbcf0ceb530fd929d352be11bc8d7b18d22c3925deb62d54fa3d6cfb4cef50`

## Model

MONAI 3D SegResNet — `blocks_down=[1,2,2,4]`, `blocks_up=[1,1,1]`, `init_filters=16`,
1 input channel, 29 output classes. Initialized from SuPreM
`supervised_suprem_segresnet_2100.pth` (81 of 83 tensors; the 29-class output head is not
transferable from SuPreM's 32-class head). Trained on PanTS-tr fold 0 for 64 epochs with
DiceCE, AdamW and a cosine schedule; checkpoint selected at epoch 59 by deterministic
whole-volume class-28 Dice on a fixed 227-case monitoring subset.

## Preprocessing

RAS orientation, 1.5 mm isotropic spacing, `[-175, 250] HU → [0,1]` clipped. Sliding-window
inference at 96³, overlap 0.5, gaussian blending.

Lesion postprocessing: class-28 voxels are grouped into 26-connected components; a component
is retained only if its peak class-28 softmax is >= 0.6; voxels in rejected components are
reassigned to their best class among 0..27. The continuous probability map is not filtered.

## Inference

```bash
python scripts/infer_segresnet.py \
    --checkpoint best.pt \
    --input unseen_ct.nii.gz \
    --output output/ \
    --lesion-peak-probability 0.6 \
    --lesion-probability
```

**Input:** one 3D CT NIfTI in Hounsfield units with a valid affine. No labels, manifest,
split, prepared data or pretraining checkpoint are required.

**Output:**

```text
combined_labels.nii.gz                  uint8, integer labels 0..28
pancreatic_lesion_probability.nii.gz    float32 in [0,1], class-28 softmax
```

Both are restored to the source CT shape, affine and orientation. Peak VRAM ~0.45 GB;
~8 s per case on an RTX 4070 Laptop.

## PanTS-te result

901 held-out scans: 151 lesion-positive, 750 lesion-negative, 0 failures.

| Model | P-Sen | T-Sen | Spe | AUC | DSC |
| --- | ---: | ---: | ---: | ---: | ---: |
| SegResNet, SuPreM initialization | 68.9% | 57.8% | 93.6% | 0.862 | 30.1% |

AUC 95% CI [0.821, 0.899], from 2000 stratified patient-level bootstrap resamples, seed 317.

- **P-Sen / Spe** — frozen pmax >= 0.6 hard operating point. 104/151 and 702/750.
- **T-Sen** — 26-connected, maximum-cardinality one-to-one any-overlap matching. 93/161.
- **AUC** — maximum class-28 softmax from the source-restored probability map, resampled to
  1-mm isotropic.
- **DSC** — macro mean over lesion-positive scans.

Supplementary: micro (pooled) DSC 50.2%; positive spatial overlap 92/151; zero-Dice scans
59/151.

The PanTS paper defines DSC, sensitivity, specificity and AUC but not P-Sen, T-Sen or any
tumor-matching rule, and we found no public PanTS implementation specifying individual-tumor
matching. The T-Sen matching rule and the AUC patient score are therefore ours, stated in
full above so they can be reproduced or replaced with the official ones.

## Environment

Python 3.11.15, PyTorch 2.13.0+cu126, MONAI 1.5.1, NumPy 2.4.6, SciPy 1.17.1,
nibabel 5.4.2. Single NVIDIA RTX 4070 Laptop GPU (8 GB).

PanTS-te was evaluated from tag `pants-submission-v1`, after the checkpoint, preprocessing,
postprocessing rule and inference implementation were frozen. A later measurement-only pass
from tag `pants-metrics-v1` added T-Sen and AUC; it changed no prediction and reproduced the
original hard counts (104/151, 48/750, 92 overlap cases, DSC 0.300721) exactly.
