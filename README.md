# Hierarchical Patch Sampling for 3D CT

**Adaptive patch sampling through hierarchical conditional probabilities.**

[Paper](paper/manuscript_kiit_2026.pdf) · [Method](#method) · [Experiments](#experimental-setup) · [Results](#results) · [Conference](#conference) · [Study notebooks](#study-notebooks)

Yongmin Park (박용민) · Computer Science, Kyonggi University · [Laplace-tech](https://github.com/Laplace-tech)

This sole-author study investigates patch selection for **3D abdominal CT
multi-organ segmentation**. Organ-wise learning states and error types guide
sampling while the nnU-Net architecture stays unchanged.

## Research question

**Can a hierarchy of organ- and error-conditioned sampling decisions improve
segmentation at a fixed update budget?**

## Method

![CT, organ selection, and three error-type candidate pools in hierarchical patch sampling](paper/figures/fig01_hierarchical_sampling.jpg)

The guided branch uses **hierarchical conditional probabilities**: select an organ
$k$, an error type $r$ within that organ, and a candidate center $c$ within that pool.

$$
p_t(k,r,c)=
\underbrace{p_t(k)}_{\text{organ}}
\underbrace{p_t(r\mid k)}_{\text{error type}}
\underbrace{p_t(c\mid k,r)}_{\text{patch center}}.
$$

Organ selection uses a smoothed Dice deficit; error-type selection combines current
candidate counts with historical error shares. A 3D patch is cropped around the
selected center. These probabilities describe the guided branch, not every patch.

<details>
<summary>Sampling equations · learning states and conditional distributions</summary>

For the current training case, $K_t$ contains organs with candidates,
$R_t(k)$ contains their non-empty error types, and $C_t(k,r)$ is a candidate pool.
States are updated from training-only observer patches, not validation labels.

**1. Organ learning state and selection**

Let $D_k^{(t)}$ be the observed hard Dice for organ $k$. Its deficit is smoothed
with an exponential moving average (EMA):

$$e_t(k)=1-D_k^{(t)},\qquad d_t(k)=\beta d_{t-1}(k)+(1-\beta)e_t(k).$$

$$p_t(k)=\frac{1-\lambda_k}{|K_t|}+\lambda_k\frac{d_t(k)}{\sum_{j\in K_t}d_t(j)}.$$

Here $\beta=0.9$ and $\lambda_k=0.5$. The uniform component keeps each eligible
organ selectable: $p_t(k)\geq(1-\lambda_k)/|K_t|$.

**2. Error type conditioned on the selected organ**

$$b_t(r\mid k)=\frac{|C_t(k,r)|}{\sum_{r'\in R_t(k)}|C_t(k,r')|}.$$

$$p_t(r\mid k)=(1-\lambda_r)b_t(r\mid k)+\lambda_r a_t(r\mid k).$$

$a_t(r\mid k)$ is the EMA of observed error-type shares, renormalized over
available types; $\lambda_r=0.5$. Those shares use raw observer error counts,
whereas $b_t$ uses bounded candidate-pool counts.

**3. Center conditioned on organ and error type**

$$p_t(c\mid k,r)=\frac{1}{|C_t(k,r)|},\qquad c\in C_t(k,r).$$

Each EMA starts from its first valid observation. Both-empty Dice observations
are skipped. Zero total organ difficulty falls back to uniform selection;
unavailable error-state mass falls back to the candidate-count distribution.
The hierarchy and mixtures follow the manuscript; the probability lower bound
above is a direct consequence, not an additional method.

</details>

| Candidate type | What the model is getting wrong | What the patch targets |
| :--- | :--- | :--- |
| Interior miss | Missed voxels inside the reference organ, away from its boundary | Organ recognition |
| Boundary disagreement | Prediction–label disagreement near the organ surface | Boundary delineation |
| Exterior false positive | That organ predicted outside the reference boundary band | Confusion with surrounding tissue |

![3D pancreas surface, error candidates, and conditional error-type probabilities](paper/figures/fig02_error_candidate_topology.png)

*Training case `s0004` · pancreas · P / seed 55254 / 10K updates.*
Markers show bounded candidate reservoirs, not every error. The highlighted center
is illustrative, not a recorded training draw; surface smoothing is for display.

### From conditional probabilities to a 3D patch

![Saved organ probabilities, conditional error-type probabilities and an illustrated 3D CT crop](paper/figures/fig04_conditional_probability_hierarchy.png)

The saved state links eight eligible organs to their error-type distributions and
an illustrative pancreas-centered crop; the gallbladder pool is empty in this snapshot.
These are sampling probabilities, not prediction confidence.
[Editable SVG](paper/figures/fig04_conditional_probability_hierarchy.svg) · [Source values](paper/results/figure04_probability_snapshot.json)

## Experimental setup

| Setting | Configuration |
| :--- | :--- |
| Dataset | TotalSegmentator v2.0.1 |
| Cohort | 602 eligible cases with all nine target masks non-empty |
| Official split retained | 525 train / 28 validation / 49 test |
| Framework | nnU-Net v2.8.1 · `3d_fullres` |
| Spacing / patch / batch | 1.5 mm isotropic / `160 × 112 × 128` / 2 |
| Training budget | 30,000 optimizer updates per run |
| Main metric | Organ macro Dice within each case, then the mean across cases |

Targets: spleen, right/left kidneys, gallbladder, liver, stomach, pancreas,
and right/left adrenal glands. Non-empty labels do **not** establish complete
anatomical coverage. Cohort filtering narrows the population to which results apply.

The comparisons keep the network, preprocessing, loss, augmentation, and update
budget matched while varying sampling. B1/A1/P share the online candidate mechanism.

| Policy | Online error candidates | Organ adaptation | Error-type adaptation |
| :--- | :---: | :---: | :---: |
| **B0** · nnU-Net foreground oversampling | — | — | — |
| **B1** · static candidate allocation | ✓ | — | — |
| **A1** · organ-adaptive allocation | ✓ | ✓ | — |
| **P** · hierarchical conditional sampling | ✓ | ✓ | ✓ |

For policy $m$ at update $u$, case-first macro Dice and the P–B0 difference are

$$
S_m(u)=\frac{1}{N}\sum_{i=1}^{N}
\left(\frac{1}{9}\sum_{k=1}^{9}\mathrm{Dice}_{i,k}^{(m,u)}\right),
$$

$$
\Delta_{\mathrm{pp}}(u)=100\left[S_P(u)-S_{B0}(u)\right].
$$

Each case and organ has equal weight. These are full-volume evaluation scores,
distinct from the observer-patch Dice used for sampling.

## Results

![Discovery learning curves and independent-seed replication of the 10K P-minus-B0 difference](paper/figures/fig03_learning_dynamics.png)

*Left: the four-policy discovery comparison. Right: 10K P − B0 in independent
training seeds; the mean error bar is a paired seed × case bootstrap 95% interval.
All evaluations use the same 28 validation cases.*

### Discovery: early improvement in one seed

Seed **55254**, **28 validation cases**.

| Policy | 10K Dice | 20K Dice | 30K Dice |
| :--- | ---: | ---: | ---: |
| B0 | 0.745542 | 0.898442 | 0.923552 |
| B1 | 0.711985 | 0.896431 | 0.922776 |
| A1 | 0.733086 | 0.897731 | 0.923548 |
| P | **0.796770** | **0.908046** | 0.923413 |
| P − B0 | **+5.1228 pp** | **+0.9603 pp** | −0.0139 pp |

[Download the discovery table](paper/results/table01_discovery.csv).

The early P–B0 difference narrowed at 20K and was nearly absent at 30K.

### Replication: the improvement was not consistent

The primary check compared P and B0 at **10K updates** in three independent training
seeds, using the same 28 validation cases.

| Seed | P − B0 Dice (pp) |
| :--- | ---: |
| 55255 | +4.3806 |
| 55256 | −1.4315 |
| 55257 | −3.8317 |
| **Mean** | **−0.2942** |

The mean difference had a paired seed × case bootstrap 95% interval of
**[−3.8562, +4.2047] pp**. Only one seed was positive: the confirmatory criterion
was **not met**.

**The experiments do not establish consistent early-learning or final-performance
superiority.** Wall-clock speedup is not claimed because cloud runtime conditions varied.

[Replication CSV](paper/results/table02_replication.csv) · [Surface metrics CSV](paper/results/table04_surface_metrics.csv) · [Exploratory test CSV](paper/results/table03_exploratory_test.csv)

<a id="detailed-results"></a>
<details>
<summary>Evaluation details, surface metrics and exploratory test</summary>

Exported from saved evaluations on 2026-09-30; no retraining for this release.

### Experimental scope

Patient independence beyond the official split was not independently established.
The filtered cohort does not represent every field of view in the 1,228-case release.
Repeated-seed summaries weight seeds equally. An empty prediction against a
non-empty reference mask has Dice zero.

| Stage | Seeds | Policies | Full-volume evaluation checkpoints |
| :--- | :--- | :--- | :--- |
| Discovery | 55254 | B0, B1, A1, P | 10K / 20K / 30K |
| Replication | 55255, 55256 | B0, B1, A1, P | 5K / 10K / 15K / 20K / 25K / 30K |
| Additional replication | 55257 | B0, P only | 5K / 10K / 15K / 20K / 25K / 30K |
| Post-primary exploratory test | 55254, 55255 | B0, P only | 10K / 30K |

### Primary check and uncertainty

All three replication seeds enter the primary summary; discovery seed 55254 does not.
At 10K, mean Dice was **0.786975 for B0** and **0.784033 for P**.
The criterion required positive differences in every replication seed and a positive
lower confidence bound. The paired seed × case percentile bootstrap used 100,000
resamples. Three seeds still give limited seed-level precision; resampling cases
does not create additional training runs.

[All six checkpoints and per-seed differences (CSV)](paper/results/table02_replication.csv).

### Surface checks at 30K

Discovery seed 55254; 28 validation cases; technical NSD tolerance 3 mm.
This tolerance is not a clinical acceptability threshold.

| Policy | NSD @ 3 mm ↑ | HD95 (mm) ↓ | Empty predictions / 252 case–organ pairs |
| :--- | ---: | ---: | ---: |
| B0 | 0.965922 | 4.667 | 1 |
| B1 | 0.966945 | 6.053 | 0 |
| A1 | 0.965281 | 6.921 | 0 |
| P | 0.964456 | 6.121 | 1 |

NSD is zero for an empty prediction; its HD95 is undefined, not zero.
HD95 summaries average available organs within each case and then across cases.
Missing entries and different availability therefore matter: subtraction of these
displayed means is **not** the paired common-nonempty-organ contrast.
The audit does not support a final surface-quality superiority claim.

[Unrounded surface summaries (CSV)](paper/results/table04_surface_metrics.csv).

### Held-out test: explicitly exploratory

49 held-out cases, evaluated **after** the failed validation primary using the
selected seeds 55254 and 55255. These are not a new random sample of training seeds;
55254 is the discovery seed. Seed selection and post-primary analysis prevent this
result from replacing the failed confirmatory check.

| Updates | B0 mean Dice | P mean Dice | P − B0 (pp) | Bootstrap 95% CI (pp) |
| :--- | ---: | ---: | ---: | :--- |
| 10K | 0.705304 | 0.767734 | +6.2430 | [+4.4458, +8.2161] |
| 30K | 0.905157 | 0.906698 | +0.1541 | [−0.3884, +0.6086] |

The intervals describe variability within this two-seed, 49-case evaluation;
they do not account for the preceding seed-selection decision.
[Unrounded test summaries (CSV)](paper/results/table03_exploratory_test.csv).

### Evidence and release scope

The manuscript focuses on the discovery comparison; this repository also includes
replication and exploratory-test evidence. Neither clinical benefit nor formal
equivalence is established.

The [source manifest](paper/results/source_manifest.json) records source-artifact
names and SHA-256 hashes for the aggregate numbers, not a full reproduction package.

The results graph is available as an [editable SVG](paper/figures/fig03_learning_dynamics.svg).
The supplied CT/3D figures derive from TotalSegmentator, not generative images.
Candidate snapshots may combine observations from different refresh times.

</details>

## Study notebooks

The study archive follows the data flow of a segmentation system, from Tensor
contracts to physical-space evaluation and nnU-Net's training pipeline.

| Part | Focus | Notebooks |
| :--- | :--- | ---: |
| [1 · Segmentation fundamentals](studies/prerequisites/part01_segmentation_fundamentals) | Tensor contracts, softmax, cross-entropy, Dice / IoU | 3 |
| [2 · U-Net from scratch](studies/prerequisites/part02_unet_from_scratch) | Convolution, skip connections, tiny overfit | 3 |
| [3 · Volumetric learning](studies/prerequisites/part03_volumetric_learning) | Conv3D memory, patch sampling, sliding-window inference | 3 |
| [4 · Medical-image geometry](studies/prerequisites/part04_medical_image_geometry_ct) | NIfTI, affine, orientation, resampling, CT windowing | 4 |
| [5 · Losses & evaluation](studies/prerequisites/part05_losses_physical_space_evaluation) | Soft Dice, surface metrics, empty masks, case aggregation | 3 |
| [6 · nnU-Net literacy](studies/prerequisites/part06_nnunet_v2_literacy) | Fingerprints, planning, oversampling, deep supervision | 3 |
| [7 · Research hygiene](studies/prerequisites/part07_research_hygiene_statistics) | Splits, leakage, reproducibility | 1 |

<details>
<summary>Full notebook index · 20 lessons</summary>

| Part / Lesson | Notebook | Cells | 난도 | 복습의 핵심 |
| --- | --- | ---: | --- | --- |
| 1.1 | [Tensor Contract](studies/prerequisites/part01_segmentation_fundamentals/01_tensor_contracts.ipynb) | 5 | ██░░░ | input·logits·target·prediction Shape |
| 1.2 | [Softmax & Cross-Entropy](studies/prerequisites/part01_segmentation_fundamentals/02_softmax_cross_entropy.ipynb) | 5 | ███░░ | 수치 안정성과 voxel-wise loss |
| 1.3 | [Dice & IoU](studies/prerequisites/part01_segmentation_fundamentals/03_dice_iou.ipynb) | 5 | ███░░ | TP/FP/FN·overlap·empty masks |
| 2.1 | [Convolution & Receptive Field](studies/prerequisites/part02_unet_from_scratch/01_convolution_shapes.ipynb) | 6 | ███░░ | convolution Shape·receptive field 계산 |
| 2.2 | [Encoder–Decoder & Skip](studies/prerequisites/part02_unet_from_scratch/02_encoder_decoder_skip.ipynb) | 5 | ███░░ | down/up sampling·feature 결합 |
| 2.3 | [Minimal U-Net & Tiny Overfit](studies/prerequisites/part02_unet_from_scratch/03_minimal_2d_unet_tiny_overfit.ipynb) | 6 | ████░ | 작은 synthetic data의 end-to-end 학습 |
| 3.1 | [Conv3D & Memory](studies/prerequisites/part03_volumetric_learning/01_conv3d_tensor_flow_memory.ipynb) | 6 | ███░░ | 3D activation·학습 peak memory |
| 3.2 | [Crop, Padding & Sampling](studies/prerequisites/part03_volumetric_learning/02_crop_padding_patch_sampling.ipynb) | 6 | ███░░ | patch 경계·uniform/foreground center |
| 3.3 | [Sliding-Window Inference](studies/prerequisites/part03_volumetric_learning/03_sliding_window_inference.ipynb) | 5 | ████░ | overlap accumulation·normalization |
| 4.1 | [NIfTI Array & Affine](studies/prerequisites/part04_medical_image_geometry_ct/01_nifti_array_affine.ipynb) | 5 | ███░░ | voxel index에서 physical coordinate로 변환 |
| 4.2 | [Orientation & Three-Plane Viewing](studies/prerequisites/part04_medical_image_geometry_ct/02_orientation_three_plane_viewing.ipynb) | 5 | ███░░ | array 축·방향·canonical RAS |
| 4.3 | [Spacing-Aware Resampling](studies/prerequisites/part04_medical_image_geometry_ct/03_spacing_aware_resampling.ipynb) | 5 | ██░░░ | Shape·spacing·image/label interpolation |
| 4.4 | [CT HU, Windowing & Anatomy](studies/prerequisites/part04_medical_image_geometry_ct/04_ct_hu_windowing_abdominal_anatomy.ipynb) | 6 | ███░░ | intensity·laterality·partial FOV |
| 5.1 | [Cross-Entropy + Soft Dice](studies/prerequisites/part05_losses_physical_space_evaluation/01_cross_entropy_soft_dice.ipynb) | 6 | ██░░░ | loss 결합·gradient 확인 |
| 5.2 | [Surface Distance, NSD & HD95](studies/prerequisites/part05_losses_physical_space_evaluation/02_surface_distance_nsd_hd95.ipynb) | 6 | ████░ | surface distance·mm tolerance |
| 5.3 | [Empty Masks & Case Aggregation](studies/prerequisites/part05_losses_physical_space_evaluation/03_empty_masks_case_aggregation.ipynb) | 5 | ███░░ | 예외 처리·case/class 집계 순서 |
| 6.1 | [Fingerprint, Plans & Preprocessing](studies/prerequisites/part06_nnunet_v2_literacy/01_dataset_fingerprint_plans_preprocessing.ipynb) | 6 | ████░ | nnU-Net planning·preprocessing flow |
| 6.2 | [Default Foreground Oversampling](studies/prerequisites/part06_nnunet_v2_literacy/02_default_foreground_oversampling.ipynb) | 6 | ███░░ | case·foreground·class·center 선택 |
| 6.3 | [Deep Supervision & Predictor](studies/prerequisites/part06_nnunet_v2_literacy/03_deep_supervision_predictor.ipynb) | 6 | ████░ | multi-scale Tensor·inference flow |
| 7.1 | [Split, Leakage & Reproducibility](studies/prerequisites/part07_research_hygiene_statistics/01_split_leakage_reproducible_runs.ipynb) | 6 | ██░░░ | statistical unit·manifest·seed |

</details>

The notebooks are educational implementations, not the experiment runners.

<details>
<summary>Running the notebooks and reading their execution record</summary>

Open a notebook, install its imports in your own Python environment, and run cells
from top to bottom. CUDA-memory exercises require a CUDA-capable GPU. Saved kernel
names describe the original workstation and may need to be reselected elsewhere.

The archive contains 111 non-empty code cells. A static check found no syntax errors
or saved exceptions; nine cells in Parts 1–3 have no execution count. This is not a
claim that every notebook has been rerun or that educational examples validate
performance on patient data.

</details>

<a id="conference"></a>
## Manuscript & conference

**Hierarchical Conditional-Probability-Based Adaptive Patch Sampling with
Organ-Wise Learning States and Error Types for 3D Abdominal CT Multi-Organ Segmentation**

*3차원 복부 CT 다장기 분할을 위한 계층적 조건부 확률 기반의 장기별 학습 상태 및 오류 유형 적응형 패치 샘플링*

[Read the manuscript (PDF)](paper/manuscript_kiit_2026.pdf)

Prepared for the **KIIT 2026 Fall Conference — Undergraduate Paper Competition**
(2026 한국정보기술학회 추계종합학술대회 및 대학생논문경진대회).

| Item | Details |
| :--- | :--- |
| Organizer | Korean Institute of Information Technology (한국정보기술학회) |
| Conference | November 26–28, 2026 · Maison Glad Jeju, South Korea |
| Participation | Undergraduate paper competition · sole author Yongmin Park |
| Current status | Preparing to participate; conference has not yet taken place as of September 30, 2026 |

Dates and venue: [official conference website](https://ki-it.or.kr/conference/fallconf26).
Presentation slides and any award documentation will be added when available.

## Repository scope

```text
hierarchical-patch-sampling-3d/
├── README.md                Research overview, results, notebook index
├── paper/
│   ├── manuscript_kiit_2026.pdf
│   ├── figures/             fig01–fig04: method and experimental results
│   └── results/             Aggregate CSVs, sampling probabilities, source hashes
└── studies/prerequisites/   20 educational notebooks
```

**Published code is limited to `studies/`.** Training/evaluation implementation,
infrastructure scripts, raw CT/masks, checkpoints, and internal reports are not
included in the current public tree. This is a research portfolio and study archive,
not a complete experiment-reproduction release. No clinical-use claim is made.

The work builds on [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
and [nnU-Net](https://github.com/MIC-DKFZ/nnUNet).
The dataset release is [TotalSegmentator v2.0.1](https://doi.org/10.5281/zenodo.10047292)
(CC BY 4.0); the displayed medical-image figures are derived from that dataset.
Full references appear in the manuscript.
