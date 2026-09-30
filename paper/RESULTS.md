# OLES3D · Results and evidence notes

[Research overview](../README.md) · [Manuscript](<2026_KIIT_추계(대학생)_박용민_논문.hwp.pdf>)

These tables summarize saved full-volume evaluation artifacts. They do not use
training-time patch pseudo Dice. Numbers are shown on the Dice 0–1 scale;
**pp** means percentage points, computed as `100 × (P − B0)`.
Publication export: 2026-09-30. No models were retrained for this export.

## 1. Experimental scope

TotalSegmentator v2.0.1 was filtered to 602 eligible cases with non-empty masks for
all nine target organs. The official split roles were retained: 525 train, 28
validation, 49 test. Patient independence beyond those official split assignments
was not independently established. Eligibility does not guarantee complete organ
coverage or represent every field of view in the original 1,228-case release.

Each case contributes the mean of its nine organ Dice scores, and cases are then
averaged with equal weight. For repeated-seed summaries, those case-first means
are averaged across seeds. A non-empty ground truth with an empty prediction has
Dice zero.

| Stage | Seeds | Policies | Full-volume evaluation checkpoints |
| :--- | :--- | :--- | :--- |
| Discovery | 55254 | B0, B1, A1, P | 10K / 20K / 30K |
| Replication | 55255, 55256 | B0, B1, A1, P | 5K / 10K / 15K / 20K / 25K / 30K |
| Additional replication | 55257 | B0, P only | 5K / 10K / 15K / 20K / 25K / 30K |
| Post-primary exploratory test | 55254, 55255 | B0, P only | 10K / 30K |

The B0/P replication summary below uses **all three** replication seeds.
Seed 55254 is discovery and is excluded from that primary summary.

## 2. Discovery: four-policy comparison

Seed 55254; 28 validation cases. [Unrounded values (CSV)](assets/discovery.csv).

| Policy | 10K | 20K | 30K |
| :--- | ---: | ---: | ---: |
| B0 | 0.745542 | 0.898442 | 0.923552 |
| B1 | 0.711985 | 0.896431 | 0.922776 |
| A1 | 0.733086 | 0.897731 | 0.923548 |
| P | 0.796770 | 0.908046 | 0.923413 |
| P − B0 (pp) | +5.1228 | +0.9603 | −0.0139 |

An early advantage was observed in this run. Similar 30K Dice is not a formal
equivalence result, and a single discovery seed cannot establish stable superiority.

## 3. Independent replication: the primary check

At 10K updates on the same 28 validation cases:

| Replication seed | P − B0 (pp) |
| :--- | ---: |
| 55255 | +4.3806 |
| 55256 | −1.4315 |
| 55257 | −3.8317 |
| **Mean** | **−0.2942** |

The mean Dice was **0.786975 for B0** and **0.784033 for P**.
The paired seed × case percentile-bootstrap 95% interval for P − B0 was
**[−3.8562, +4.2047] pp** (100,000 resamples).
The prespecified criterion required positive differences in all replication seeds
and a positive lower confidence bound. It was **not met**.
Three seeds provide limited precision for seed-level uncertainty; resampling
patients/cases does not create additional independent training runs.

[All six checkpoints and per-seed differences (CSV)](assets/replication.csv).

## 4. Surface checks at 30K

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

[Unrounded surface summaries (CSV)](assets/surface_checks.csv).

## 5. Held-out test: explicitly exploratory

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
[Unrounded test summaries (CSV)](assets/exploratory_test.csv).

## 6. Interpretation and release scope

- The contribution is an implemented, evaluated hierarchical organ/error-type
  sampling design and its controlled comparison with nnU-Net sampling.
- The discovery run and selected-seed test show early gains, but independent-seed
  replication does not establish a consistent improvement.
- The experiment does not demonstrate final-Dice superiority, clinical benefit,
  formal equivalence, or robust wall-clock savings. Cloud host variability makes
  cross-host runtime claims inappropriate.
- The manuscript focuses on the discovery comparison; this companion includes
  the broader replication and exploratory-test evidence so that the public
  overview is not limited to the favorable run.

The [source manifest](assets/sources.json) records local source-artifact names and
SHA-256 hashes for the exported aggregate numbers. Source artifacts and research
implementation are not included in the public tree. These hashes document
provenance; they are not a substitute for a full reproducibility release.

The [overview figure](assets/evidence_overview.png) is also available as an
[editable vector SVG](assets/evidence_overview.svg). The CT and 3D method figures
are supplied manuscript assets derived from TotalSegmentator, not synthetic
generative images. The illustrated training case is `s0004`; error candidates
come from a bounded observer snapshot and need not represent all voxel errors
at a single simultaneous prediction time.
