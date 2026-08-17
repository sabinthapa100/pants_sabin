# PanTS Technical Submission

Sabin Thapa
Kent State University
<sthapa3@kent.edu>

Repository: <https://github.com/sabinthapa100/pants_sabin>
Evaluated release tag: **`pants-submission-v1`**
Source SHA: `72813b288db26dd0f887fe9d29a007fa46bcc764`

## Model

| | |
| --- | --- |
| Architecture | MONAI 3D SegResNet — `blocks_down=[1,2,2,4]`, `blocks_up=[1,1,1]`, `init_filters=16`, 1 input channel, 29 output classes |
| Initialization | SuPreM `supervised_suprem_segresnet_2100.pth`; 81 of 83 tensors transferred, output head excluded |
| Training | PanTS-tr fold 0 (7,199 cases), 64 epochs, DiceCE, AdamW, cosine schedule |
| Selected checkpoint | epoch 59, by deterministic whole-volume class-28 Dice on a fixed 227-case monitoring subset |
| Checkpoint SHA256 | `54bbcf0ceb530fd929d352be11bc8d7b18d22c3925deb62d54fa3d6cfb4cef50` |

Preprocessing: RAS orientation, 1.5 mm isotropic spacing, `[-175, 250] HU → [0,1]` clipped.

## Inference

```bash
python scripts/infer_segresnet.py \
    --checkpoint best.pt \
    --input <raw_ct.nii.gz> \
    --output <output_dir>/ \
    --lesion-peak-probability 0.6 \
    --lesion-probability
```

Requires only a raw CT and the checkpoint — no labels, manifest, split, prepared cache, or
pretraining file. Sliding window 96³, overlap 0.5, gaussian blending, ~0.45 GB VRAM.

Lesion postprocessing: 26-connectivity connected components of the hard class-28 argmax; a
component is kept only if its peak class-28 softmax is >= 0.6; rejected voxels are reassigned
by argmax over channels 0..27 rather than forced to background. The continuous probability
map is **not** postprocessed.

## Output

```text
combined_labels.nii.gz                  uint8, integer labels 0..28
pancreatic_lesion_probability.nii.gz    float32 in [0,1], raw class-28 softmax
```

Both are restored to the source CT shape, affine and orientation.

## PanTS-te result — our frozen internal evaluation protocol

| | |
| --- | ---: |
| Cases evaluated | 901 (0 failures) |
| Lesion-positive | 151 |
| Lesion-negative | 750 |
| Mean positive-case class-28 Dice | 0.3007 |
| Median positive-case Dice | 0.1764 |
| Positive spatial overlap | 92/151 (60.9%) |
| Predicted but zero overlap | 12/151 |
| No lesion predicted | 47/151 |
| Internal patient-wise prediction rate | 104/151 (68.9%) |
| False-positive patients | 48/750 (6.4%) |
| Internal specificity | 93.6% |
| Macro anatomy Dice (classes 1–27) | 0.7136 |

## Metric note

These are **internal** quantities computed by our own evaluator. The public PanTS repository
defines P-Sen and T-Sen in prose but does not publish the evaluator, the lesion
component-matching rule, the patient-level scoring convention, or the AUC definition, so we
have not attempted to reconstruct them. Our "patient-wise prediction rate" counts any
predicted class-28 voxel; our "specificity" is `1 − FP rate` under the same criterion. Dice
on a case where the class is absent from both prediction and ground truth is returned as NaN
rather than 1.0, and lesion Dice is averaged over lesion-positive cases only.

The checkpoint and the probability-producing inference script are provided so the official
PanTS evaluation can be applied directly.

## Reproducibility

The held-out result was produced from tag `pants-submission-v1` with the command above
(`--lesion-peak-probability 0.6` passed explicitly), on a single NVIDIA RTX 4070 Laptop GPU:
901 cases in 124.7 min, 8.31 s/case mean, 0.45 GB peak VRAM.

Environment: Python 3.11.15, PyTorch 2.13.0+cu126, MONAI 1.5.1, NumPy 2.4.6, SciPy 1.17.1,
nibabel 5.4.2.

PanTS-te was read exactly once, after the checkpoint, preprocessing, postprocessing rule,
inference implementation and metric definitions were frozen and tagged.
