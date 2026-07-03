# 프레임워크 개요도 — 박스 내용 상세 정리

전체 파이프라인(개요도)과 각 모듈의 동작원리도(상세도)에 들어갈 내용을 단계별로 정리한 문서다.
각 항목은 **박스 라벨(짧게)** + **부제(동작원리 한 줄)** + **상세 설명(캡션·발표용)** + **주요 파라미터** 순으로 구성한다.
짧은 것은 박스 안에, 상세 설명은 그림 캡션이나 본문에 배치하는 것을 전제로 한다.

---

## 0. 전역 설정 / 시각 언어

| 항목 | 내용 |
|------|------|
| 입력 | 직물 배접(textile-mounted) 한국 전통 회화의 고해상도 스캔 |
| 출력 | 객체별 독립 3D 텍스처 메시 |
| 단계 수 | 4-Stage (Restoration → Segmentation → Inpainting → 3D Reconstruction) |
| Upscaling | **제외** (과거 "Image Enhancement"는 업스케일과 결합돼 있었으나 분리·삭제) |
| 명명 규칙 | "Stage"는 4개 모듈에만 사용. 모듈 내부 단계는 "Step" |
| 강조 태그 | Restoration = *newly developed* / Inpainting(SSEI) = *core contribution* |
| 위계 표현 | 선결조건 = 점선·회색 / 핵심 메커니즘 = 실선·컬러 |
| 모듈 색상 | amber(Stage 1) · green(Stage 2) · blue(Stage 3) · pink(Stage 4) |

**단계 간 데이터 흐름(화살표 위 산출물).** 스캔 → (de-weaved 이미지) → (객체 마스크 + RGBA) → (완성된 객체 이미지) → (3D 메시). 단계 순서는 데이터 의존성으로 강제된다: 격자 제거가 분할 경계와 자기참조 매칭을 방해하므로 Stage 1이 선행하고, 분할 마스크 없이는 폐색 영역 식별 자체가 불가능하므로 Stage 2가 Stage 3의 전제가 된다.

---

## Stage 1 · Restoration  〔태그: newly developed〕

**모듈 역할.** 직물 배접 특유의 주기적 격자(weave) 패턴을 제거하고, 그 과정에서 약화된 객체 윤곽선을 회복한다. 출력은 격자가 제거된(de-weaved) 이미지다. 학습 데이터 없이 주파수·공간 도메인 분석만으로 동작하는 추론 전용 모듈이다.

**파이프라인상 위치.** 격자 패턴은 객체의 edge와 유사한 주파수 대역에 위치해 후속 단계 전부에 간섭하므로 가장 먼저 수행된다.

### Step 1 — Split Radius (Spectral Interpolation)
- **박스 라벨**: `Split radius`  ·  **부제**: `FFT notch + spectral interp`
- **상세**: 직물 격자는 약 9×7픽셀 주기를 가지므로 주파수 도메인(FFT)에서 (±W/9, 0), (0, ±H/7) 위치에 뚜렷한 peak로 나타난다. 각 peak를 중심으로 일정 반경을 "구멍"으로 만든 뒤(notch), 그 자리를 주변 주파수 값으로 보간(spectral interpolation)하고 inverse FFT로 공간 영역을 복원한다. 배경 스펙트럼 추정은 peak를 중심으로 한 환형(annulus, 내측 2 / 외측 5) 영역의 산술 평균을 사용한다.
- **Split Radius 핵심**: 축 정렬 peak에는 넓은 반경(r_a), 대각선의 교차 고조파(cross-harmonic) peak에는 보수적 반경(r_c)을 *분리* 적용한다. 이로써 균일하게 공격적인 보간이 손상시키는 에지 디테일을 보존하면서 추가 격자 성분을 제거한다.
- **역할/한계**: 평탄 영역에서 매우 효과적이나, 객체 윤곽선 근처·좁은 영역(잎맥·머리카락 등)에서는 격자가 잔존한다 → Step 2가 보완.
- **주요 파라미터**: 격자 주기 ≈ 9×7px, axis peak_radius=3, cross peak_radius=1, 환형 보간 [2, 5]

### Step 2 — Spatial-Adaptive NLM Blending  〔★ 신규 기여〕
- **박스 라벨**: `Adaptive NLM`  ·  **부제**: `narrow-region blend`
- **문제 의식**: 표준 NL-Means는 공간적으로 균일한 강도로 작동한다. 강하게 걸면 평탄 영역의 디테일이 뭉개지고, 약하게 걸면 좁은 영역의 격자가 남는다. 즉 단일 강도로는 모든 영역을 동시에 만족시킬 수 없다.
- **해결(동작원리)**:
  1. K-means(5 clusters)로 색 기반 영역 분할
  2. 각 클러스터마다 `distanceTransform`으로 클러스터 경계까지의 거리 측정
  3. 거리 < 8px인 픽셀을 "좁은 영역"으로 보고 연속값(0~1)의 `narrow_mask` 생성
  4. Strong NLM(h=15)을 **원본 이미지**에 적용 (Split 출력이 아니라 원본 — 잔존 격자까지 포함해 강하게 평활)
  5. 블렌딩: `out = (1 − m) · split_out + m · strong_nlm`
- **핵심 의의**: 한 알고리즘을 전 영역에 균일 적용하지 않고 **이미지 내 위치별로 다른 처리**를 한다(평탄 영역은 Split 결과 유지, 좁은 영역은 Strong NLM 사용). distance transform + medial-axis 기반의 *공간 적응형 주기 아티팩트 제거*가 이 모듈의 신규성이다.
- **⚠ 다이어그램 주의**: "Strong NLM은 원본에 적용 → 블렌딩"임을 화살표로 분명히 한다. Split 결과에 NLM을 다시 거는 것으로 오해되기 쉽다.
- **주요 파라미터**: h=15 / h_color=15, narrow_threshold=8px, n_clusters=5, template_window=7, search_window=21, blur_sigma=2.0

### Step 3 — Contour Enhancement
- **박스 라벨**: `Contour boost`  ·  **부제**: `LAB L-channel · Scharr`
- **상세**: Step 1~2의 스펙트럼 처리에서 격자 주파수와 인접한 윤곽선 주파수가 함께 감쇠되어 객체 윤곽의 명도 대비가 약화된다. 이를 보정하기 위해 BGR→LAB 변환 후 **L 채널만** 처리한다(a*, b* 보존 → 색감 변화 최소화). adaptive threshold(block_size=15, C=6.0)로 윤곽선 마스크를, Scharr gradient(Sobel보다 정확한 1차 미분)로 윤곽선 강도를 추출한 뒤 `L_new = L_old + boost × edge_signal`로 선택적으로 어둡게 강화하고 LAB→BGR로 되돌린다.
- **주요 파라미터**: block_size=15, thresh_C=6.0, boost=10.0

> **보조 근거(선택).** 격자 채색성 실증 — 격자 신호의 평균 변화량이 L 대비 a* 39.5%, b* 76%로 측정되어 "격자가 채색되어 있다"는 관찰을 정량 확증했다. 향후 chroma-aware 필터링의 정당화 근거이며, 별도 막대 그래프로 제시 가능.

---

## Stage 2 · Object Segmentation

**모듈 역할.** 회화 내 개별 유물을 식별하고 각각을 독립적인 RGBA 이미지로 추출한다. 출력은 객체별 이진 마스크 + RGBA 이미지이며, 이후 모든 처리가 객체 단위로 진행된다(per-object processing).

### Step 1 — LabelMe Polygon  〔선결조건: 수동 레이블〕
- **박스 라벨**: `LabelMe`  ·  **부제**: `polygon labels`
- **상세**: LabelMe 형식의 폴리곤 주석으로 각 객체를 지정한다. 조잡한 폴리곤 경계는 정밀 에지를 포착하지 못하므로 Step 2의 정제 소스로만 사용된다.

### Step 2 — SAM Refinement
- **박스 라벨**: `SAM refine`  ·  **부제**: `2-pass · shrink-only`
- **상세(동작원리)**:
  - **2-Pass 예측**: 침식된 마스크 내부의 양성 점, 팽창된 외부 링의 음성 점, 패딩된 바운딩 박스를 구조화 프롬프트로 생성. 1차 패스의 조잡한 로짓 맵을 2차 패스의 `mask_input`으로 피드백해 경계를 정제.
  - **적응적 파라미터 스케일링**: 마진·침식·팽창 반경을 512px 기준으로 보정한 뒤 각 객체 바운딩 박스에 비례 스케일링.
  - **제약 기반 정제**: 축소 전용(`M_refined = M_SAM ∩ M_rough`, 배경 확장 방지) + 내부 보존(`∪ erode(M_rough)`, 내부 구멍 생성 방지).
  - **안전 검사**: 정제 마스크 면적이 원본 폴리곤의 30% 미만이면 정제를 폐기하고 원본 유지.
- **모델**: SAM vit_h

### Step 3 — Object Extraction
- **박스 라벨**: `RGBA extract`  ·  **부제**: `edge feathering`
- **상세**: 정제된 마스크로 각 객체를 RGBA로 추출. 알파 채널에 distance transform 기반 에지 페더링을 적용해 경계부를 부드럽게 전환.

---

## Stage 3 · Inpainting (SSEI)  〔태그: core contribution〕

**모듈 역할.** 레이블 기반으로 폐색 영역을 탐지하고, 학습 불필요 자기참조 접근으로 원본 회화 고유의 시각 양식을 보존하며 가려진 부분을 복원한다. SSEI = Style-consistent Self-Exemplar Inpainting. 본 논문의 핵심 기술 기여다.

> **위계.** Layer Order와 Occlusion Detection은 "어디를, 어떤 순서로 채울지"를 정하는 **선결조건(준비 단계)**이며, 실제 알고리즘 기여는 아래 두 핵심 메커니즘에 있다. 다이어그램에서 선결조건은 한 묶음으로 압축하고 핵심 2개에 지면을 몰아준다.

### 선결조건 (압축)
- **Layer Order** 〔점선·회색〕 — LabelMe 주석의 `layer_order` 정수 필드로 객체 전후 순서를 수동 지정(작을수록 전경). 산점투시 회화에서 자동 깊이 추정이 부정확할 수 있어 수동 레이블을 채택. *(향후 MiDaS·VLM 기반 자동화 가능)*
- **Occlusion Detection** 〔점선·회색〕 — 인스턴스 마스크와 레이어 순서로 가려진 픽셀을 식별. `O_j = M_j ∩ dilate(M_i)` (단, L_i ≺ L_j). 객체별 가시 마스크 `V_j = M_j \ O_j`, 폐색 마스크 `O_j` 산출.

### 핵심 1 — Constrained Self-Exemplar Completion 〔실선·blue〕
- **박스 라벨**: `Self-exemplar`  ·  **부제**: `visible patches only`
- **공식화**: 객체 j의 가시 영역 `V_j`, 폐색 영역 `O_j`가 주어질 때, 모든 `p ∈ O_j`에 대해 `I_p = argmin_{q ∈ E_j} ‖p − q‖`를 풀어 채운다.
- **제약 3종(동작원리)**:
  1. **참조원 제한** — `q ∈ E_j ⊆ V_j`. 같은 객체의 *가시 영역 패치만* 참조 → 합성 콘텐츠가 모두 입력 이미지 자체에서 기원하므로 스타일 보존이 알고리즘의 구조적 성질로 보장됨.
  2. **채움 순서** — 폐색 경계 `∂O_j`까지의 거리가 작은 픽셀부터(경계 → 안쪽, onion-peel).
  3. **색 일관성** — 분할 경계 근처 픽셀이 폐색 객체의 색을 포함해 참조 통계를 왜곡하는 것을 차단.
- **해결하는 3대 실패(naive PatchMatch 대비)**: ① 무제한 탐색 오염(타 객체·배경 패치 유입), ② 채움 순서 아티팩트(내부 색 불연속), ③ 경계 색 오염.

### 핵심 2 — Dynamic Edge Generation (Algorithm 5) 〔실선·blue〕
- **박스 라벨**: `Edge render`  ·  **부제**: `skeleton width transfer`
- **상세(동작원리, 4단계)**:
  1. **Skeleton + Width map** — 가시 영역의 Canny 엣지(∩ 가시 마스크)에서 morphological close → distance transform → Zhang-Suen 세선화로 skeleton 추출. 각 skeleton 픽셀의 폭 `w = 2·D`를 [p_min, p_max]로 clamp.
  2. **Color profile** — skeleton의 법선 방향으로 색을 샘플링해 `(c_center, c_boundary)` 프로파일 추출.
  3. **KD-tree transfer** — 가시 skeleton 위치로 KD-tree를 구성하고, inpaint 경계의 각 centerline 픽셀에 대해 KNN(거리가중 평균)으로 폭·프로파일을 전이.
  4. **Radial gradient 렌더** — 전이된 폭에 걸쳐 `c_center → c_boundary` 방사형 그래디언트로 윤곽선을 렌더.
  - **Fallback**: skeleton이 부족하면 고정 2px 에지 렌더링으로 자동 전환(강건성 보장).
- **의의**: 윤곽선이 지배적인 한국 회화에서, 패치 매칭만으로는 흐려지는 객체 윤곽을 복원. ablation에서 고정 폭 대비 **+3.25 dB PSNR** 기여.
- **주요 파라미터**: Canny τ_low=50 / τ_high=150, k_nn=10, p_min=1, p_max=8

---

## Stage 4 · 3D Reconstruction

- **현재: 비워둠** (요청에 따라 상세 미기재)
- (향후 채울 내용) Wonder3D를 주 모델로, cross-domain diffusion으로 다시점 normal map + color를 동시 생성 → normal fusion으로 텍스처 메시 재구성(객체당 약 2~3분). 대안으로 InstantMesh, LGM.

---

## 다이어그램 제작 시 체크리스트

- [ ] "Stage"는 4개 모듈에만, 내부는 "Step"으로 — 두 층위 혼동 방지
- [ ] Restoration Step 2: "Strong NLM은 **원본**에 적용 후 블렌딩" 경로를 명확히
- [ ] SSEI: Layer Order·Occlusion은 점선·회색(선결조건)으로 압축, 핵심 2개를 크게
- [ ] 단계 사이 화살표 위/근처에 산출물 표기(de-weaved → 마스크+RGBA → 완성본 → 메시)
- [ ] 강조는 두 곳만: Restoration(newly developed), SSEI(core contribution)
- [ ] 본문 동기화: 업스케일 분리에 맞춰 §Sandwich Architecture·Real-ESRGAN·Figure 4-E1 캡션 재서술
