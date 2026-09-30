# OLES3D Prerequisite Course

**과정 종료: 2026-09-11 — Part 1–7.1, 20개 lesson.**

2D classification 경험을 3D medical image segmentation의 Tensor/data flow,
geometry, sampling, evaluation으로 연결한 scratch 학습 기록이다.
과정 종료는 독립 숙련 인증이나 실제 nnU-Net 학습 성공을 의미하지 않는다.
7.2 paired evaluation·bootstrap은 실제 연구 결과 분석 단계로 이관했다.

[연구 소개](../../README.md) · [논문·실험 결과](../../paper/RESULTS.md)

이 공개 archive는 연구에 앞서 진행한 학습 목차와 실행 근거를 담는다.
실제 OLES3D training/evaluation pipeline은 공개 범위에 포함하지 않는다.
Notebook의 교육용 구현과 논문의 실험 구현을 구분해서 참고한다.

## 실행 근거

- 종료 당시 변경·신규 notebook 7개, non-empty code cell 41개를 notebook별
  fresh project kernel에서 순차 실행해 통과했다. 나머지 13개는 당시 재실행하지 않았다.
- 6.3과 7.1의 미실행 cell에 검증 결과를 저장했고, 기존 learner code·output은 보존했다.
- 2026-09-12 read-only 점검: 20개 notebook, non-empty code cell 111개,
  저장된 exception output 0개. 모든 code cell의 Python 구문 검사 통과.
- Part 1–3의 non-empty cell 9개에는 execution count가 없다. 저장된 오류가
  없다는 사실은 전체 cell의 실행 성공 증명이 아니다. 이번 문서 정리에서는
  notebook을 재실행하거나 변경하지 않았다.

## 전체 목차·학습량

Cell 수는 `Cell 0 — Project Imports`를 포함한 non-empty code cell 기준이다.
5.3의 빈 trailing cell은 제외했다. 난도는 학습 설계상의 추정치이며
개인의 능력이나 숙련도 점수가 아니다.

| Part / Lesson | Notebook | Cells | 난도 | 복습의 핵심 |
| --- | --- | ---: | --- | --- |
| 1.1 | [Tensor Contract](part01_segmentation_fundamentals/01_tensor_contracts.ipynb) | 5 | ██░░░ | input·logits·target·prediction Shape |
| 1.2 | [Softmax & Cross-Entropy](part01_segmentation_fundamentals/02_softmax_cross_entropy.ipynb) | 5 | ███░░ | 수치 안정성과 voxel-wise loss |
| 1.3 | [Dice & IoU](part01_segmentation_fundamentals/03_dice_iou.ipynb) | 5 | ███░░ | TP/FP/FN·overlap·empty masks |
| 2.1 | [Convolution & Receptive Field](part02_unet_from_scratch/01_convolution_shapes.ipynb) | 6 | ███░░ | convolution Shape·receptive field 계산 |
| 2.2 | [Encoder–Decoder & Skip](part02_unet_from_scratch/02_encoder_decoder_skip.ipynb) | 5 | ███░░ | down/up sampling·feature 결합 |
| 2.3 | [Minimal U-Net & Tiny Overfit](part02_unet_from_scratch/03_minimal_2d_unet_tiny_overfit.ipynb) | 6 | ████░ | 작은 synthetic data의 end-to-end 학습 |
| 3.1 | [Conv3D & Memory](part03_volumetric_learning/01_conv3d_tensor_flow_memory.ipynb) | 6 | ███░░ | 3D activation·학습 peak memory |
| 3.2 | [Crop, Padding & Sampling](part03_volumetric_learning/02_crop_padding_patch_sampling.ipynb) | 6 | ███░░ | patch 경계·uniform/foreground center |
| 3.3 | [Sliding-Window Inference](part03_volumetric_learning/03_sliding_window_inference.ipynb) | 5 | ████░ | overlap accumulation·normalization |
| 4.1 | [NIfTI Array & Affine](part04_medical_image_geometry_ct/01_nifti_array_affine.ipynb) | 5 | ███░░ | voxel index에서 physical coordinate로 변환 |
| 4.2 | [Orientation & Three-Plane Viewing](part04_medical_image_geometry_ct/02_orientation_three_plane_viewing.ipynb) | 5 | ███░░ | array 축·방향·canonical RAS |
| 4.3 | [Spacing-Aware Resampling](part04_medical_image_geometry_ct/03_spacing_aware_resampling.ipynb) | 5 | ██░░░ | Shape·spacing·image/label interpolation |
| 4.4 | [CT HU, Windowing & Anatomy](part04_medical_image_geometry_ct/04_ct_hu_windowing_abdominal_anatomy.ipynb) | 6 | ███░░ | intensity·laterality·partial FOV |
| 5.1 | [Cross-Entropy + Soft Dice](part05_losses_physical_space_evaluation/01_cross_entropy_soft_dice.ipynb) | 6 | ██░░░ | loss 결합·gradient 확인 |
| 5.2 | [Surface Distance, NSD & HD95](part05_losses_physical_space_evaluation/02_surface_distance_nsd_hd95.ipynb) | 6 | ████░ | surface distance·mm tolerance |
| 5.3 | [Empty Masks & Case Aggregation](part05_losses_physical_space_evaluation/03_empty_masks_case_aggregation.ipynb) | 5 | ███░░ | 예외 처리·case/class 집계 순서 |
| 6.1 | [Fingerprint, Plans & Preprocessing](part06_nnunet_v2_literacy/01_dataset_fingerprint_plans_preprocessing.ipynb) | 6 | ████░ | nnU-Net planning·preprocessing flow |
| 6.2 | [Default Foreground Oversampling](part06_nnunet_v2_literacy/02_default_foreground_oversampling.ipynb) | 6 | ███░░ | case·foreground·class·center 선택 |
| 6.3 | [Deep Supervision & Predictor](part06_nnunet_v2_literacy/03_deep_supervision_predictor.ipynb) | 6 | ████░ | multi-scale Tensor·inference flow |
| 7.1 | [Split, Leakage & Reproducibility](part07_research_hygiene_statistics/01_split_leakage_reproducible_runs.ipynb) | 6 | ██░░░ | statistical unit·manifest·seed |

```text
과정 종료       [████████████████████] 20/20 lessons
코드 산출물     111 cells = 20 import + 91 learning
난도 ██░░░      22 cells — 기초 개념 연결
난도 ███░░      60 cells — 여러 개념·Tensor/data flow 추적
난도 ████░      29 cells — 수학·framework 내부 동작의 주요 고비
```

## 연구에서 다시 사용할 핵심

| 배운 부분 | 실제 연구에서 연결할 판단 |
| --- | --- |
| Part 1 | `[B,C,D,H,W]` input, `[B,K,D,H,W]` logits, `[B,D,H,W]` target/prediction 구분; background accuracy에 의존하지 않기 |
| Part 2 | Encoder/decoder·skip의 Shape 추적; tiny overfit은 pipeline 점검이지 일반화 증명이 아님 |
| Part 3 | Case 선택과 patch-center 선택 분리; full inference의 overlap 정규화; raw tensor와 peak VRAM 구분 |
| Part 4 | 같은 Shape라도 affine 비교; image와 label interpolation 구분; non-empty와 장기 전체 포함 구분 |
| Part 5 | 학습 loss와 보고 metric 분리; mm 단위 거리·empty 규칙·case-first 집계 명시 |
| Part 6 | nnU-Net이 정하는 plan과 우리가 바꿀 sampler 경계 구분; 교육용 simulation과 실제 framework 실행 구분 |
| Part 7 | Case/patient와 slice/voxel의 통계 단위 구분; validation 선택과 locked test의 경계 유지 |

5.2의 작은 surface-point 예제는 개념 학습용이다. 연구 평가에서는 surface
가중 방식, 양방향 distance 집계, percentile 정의, empty-mask 처리까지
선택한 evaluator와 대조해야 한다. Notebook 함수를 그대로 복사했다고
표준 구현과 동등한 metric이 되는 것은 아니다.

## 복습 방식

각 notebook을 열어 설명·Tensor shape·주석을 읽고 위에서 아래로 실행한다.
장기 학습 및 CUDA memory 측정 cell은 GPU와 실행 시간이 필요할 수 있다.
핵심 package는 각 notebook의 `Cell 0 — Project Imports`에 명시되어 있다.
저장된 kernel 표시명은 작성 당시 환경 정보이며, 다른 컴퓨터에서는 해당
package가 설치된 자신의 Jupyter kernel을 선택해야 한다.

교육용 synthetic example의 결과를 실제 환자에 대한 성능으로 해석하지 않는다.
연구 적용 결과와 한계는 [프로젝트 소개](../../README.md)에서 확인할 수 있다.
