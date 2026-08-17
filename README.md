# PanTS Pancreatic Lesion Segmentation

Sabin Thapa
Kent State University
<sthapa3@kent.edu>

A 3D SegResNet study on PanTS comparing random initialization against supervised SuPreM
initialization, followed by one frozen held-out PanTS-te evaluation.

- PanTS dataset and benchmark: <https://github.com/MrGiovanni/PanTS>
- PanTS paper: <https://www.cs.jhu.edu/~zongwei/publication/li2025pants.pdf>
- PanTSMini distribution: <https://huggingface.co/datasets/BodyMaps/PanTSMini>
- SuPreM code and weights: <https://github.com/MrGiovanni/SuPreM>

## Data

PanTS-tr 9,000 cases; PanTS-te 901 cases; 29 exclusive semantic classes; class 28 is the
pancreatic lesion. The image and label files used here came from the PanTSMini distribution,
fetched with the upstream `download_PanTS_data.sh`.

The tracked five-fold split (`pants_cv_v1.json`, seed 317) is **case-level**, because the
released metadata provides no patient-group identifier. Fold 0 is 7,199 train / 1,801
validation.

## Model and preprocessing

| | |
| --- | --- |
| Architecture | MONAI 3D SegResNet, `blocks_down=[1,2,2,4]`, `init_filters=16`, 29 outputs |
| Initialization | SuPreM `supervised_suprem_segresnet_2100.pth` — 81 of 83 tensors transferred |
| Orientation | RAS |
| Spacing | 1.5 mm isotropic |
| Intensity | `[-175, 250] HU → [0,1]`, clipped |
| Patch | 96³ |

Only the output head `conv_final.2.conv.{weight,bias}` is excluded from transfer: SuPreM
predicted 32 classes where PanTS needs 29. All transferred weights remain trainable.

## Training

64 epochs, 3,600 iterations per epoch, 230,400 optimizer updates. `DiceCELoss` (background
excluded from the Dice term), AdamW lr 1e-4 / weight decay 1e-5, cosine learning-rate
schedule, mixed precision. Best checkpoint at **epoch 59**, selected by deterministic
whole-volume mean class-28 Dice over a fixed 227-case monitoring subset.

Random and SuPreM used the same scientific training protocol. Random was trained on a Colab
T4 and SuPreM on an RTX 4070 with different PyTorch/CUDA environments, so the comparison is
**protocol-matched but not hardware-controlled**.

## Development result

PanTS-tr fold-0 development result:

| | Random | SuPreM | SuPreM + 0.6 filter |
| --- | ---: | ---: | ---: |
| mean class-28 Dice (177 positives) | 0.1614 | 0.2440 | 0.2437 |
| positive spatial overlap / 177 | 60 | 95 | 90 |
| false-positive scans / 1,624 | 7 | 201 | 109 |
| specificity | 0.9957 | 0.8762 | 0.9329 |
| macro anatomy Dice (classes 1–27) | 0.6852 | 0.6917 | 0.6917 |

Single fold, single seed.

## Frozen inference rule

29-class argmax produces the hard label map. Class-28 voxels are grouped into 26-connected
components; a component is retained only if its peak class-28 softmax is >= 0.6. Voxels in
rejected components are reassigned to their best class among 0..27 rather than forced to
background. The continuous class-28 probability map is left unfiltered.

0.6 is a fixed operating threshold selected on fold 0.

## PanTS-te result

| | |
| --- | ---: |
| Cases | 901 |
| Lesion-positive | 151 |
| Lesion-negative | 750 |
| P-Sen | 104/151 = 68.9% |
| Specificity | 702/750 = 93.6% |
| Mean lesion DSC on positive scans | 30.1% |
| Median lesion DSC | 17.6% |
| Positive spatial overlap | 92/151 = 60.9% |
| Prediction but zero overlap | 12/151 |
| No lesion prediction | 47/151 |

Definitions used here:

- **P-Sen** — fraction of lesion-positive scans with at least one retained class-28
  component, regardless of location.
- **Specificity** — fraction of lesion-negative scans with no retained class-28 component.
- **Mean lesion DSC** — mean class-28 Dice over lesion-positive scans.

T-Sen and AUC are not reported because this evaluation did not implement individual-tumor
matching or a continuous patient-level score. The inference script outputs a source-geometry
segmentation and continuous class-28 probability map for further evaluation.

![held-out outcomes](docs/figures/02_heldout_failure_modes.png)

## Inference

```bash
python scripts/infer_segresnet.py \
    --checkpoint best.pt \
    --input unseen_ct.nii.gz \
    --output output/ \
    --lesion-peak-probability 0.6 \
    --lesion-probability
```

Input contract: one 3D CT NIfTI in Hounsfield units with a valid affine.

Outputs `combined_labels.nii.gz` and `pancreatic_lesion_probability.nii.gz`, both restored to
the source CT shape, affine and orientation.

Inference does not need labels, a manifest, a split, prepared training data, or the original
SuPreM checkpoint.

## Reproducibility

| | |
| --- | --- |
| Evaluated tag | `pants-submission-v1` |
| Evaluated source SHA | `72813b288db26dd0f887fe9d29a007fa46bcc764` |
| Final checkpoint SHA256 | `54bbcf0ceb530fd929d352be11bc8d7b18d22c3925deb62d54fa3d6cfb4cef50` (epoch 59) |
| Environment | Python 3.11.15, PyTorch 2.13.0+cu126, MONAI 1.5.1 |

Methods detail and data QC evidence: [METHODS_AND_QC.md](METHODS_AND_QC.md).

## Limitations

- One fold and one seed.
- Different hardware/software environments between the two arms.
- The 0.6 threshold was selected on fold 0.
- Small-lesion performance remains weak.
- No T-Sen or AUC calculation.
- Probabilities are not calibrated.

## References

- PanTS — Li et al., NeurIPS 2025 Datasets and Benchmarks Track.
- SuPreM — Li et al., [arXiv:2501.11253](https://arxiv.org/abs/2501.11253).
- MONAI — Medical Open Network for AI.
