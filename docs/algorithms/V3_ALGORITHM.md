# V3 Algorithm: Spatial-Adaptive Grid Removal with Multi-Stage Restoration

**버전**: 1.0
**날짜**: 2026-05-22
**대상**: 한국화(비단/마(麻) 마운트) 디지털 스캔의 격자 패턴 제거

---

## 목차

1. [문제 정의](#1-문제-정의)
2. [전체 파이프라인](#2-전체-파이프라인)
3. [Stage 1: Split Radius](#3-stage-1-split-radius)
4. [Stage 2: Spatial-Adaptive NLM](#4-stage-2-spatial-adaptive-nlm)
5. [Stage 3: Contour Enhancement](#5-stage-3-contour-enhancement)
6. [전체 Pseudo-code](#6-전체-pseudo-code)
7. [파라미터 레퍼런스](#7-파라미터-레퍼런스)
8. [Computational Complexity](#8-computational-complexity)
9. [구현 매핑](#9-구현-매핑)

---

## 1. 문제 정의

### 1.1 관측 모델

한국화는 비단/마(麻) 위에 그려져 있으며, 디지털 스캔 시 직물의 격자 패턴이 함께 포착됩니다.

```
I_observed(x, y) = I_clean(x, y) · G_grid(x, y) + ε
```

- `I_observed`: 관측된 (격자 포함) BGR 이미지
- `I_clean`: 격자가 제거된 원본 그림
- `G_grid`: 곱셈성 격자 패턴 (주기 9×7 픽셀)
- `ε`: 잡음

### 1.2 핵심 도전

격자 신호의 공간적 분포가 **이미지 내 위치별로 다른 처리 난이도**를 요구:

| 영역 유형 | 특성 | 격자 제거 난이도 |
|----------|------|----------------|
| 평탄 배경 | 균일한 색, 격자만 변동 | 쉬움 (FFT 만으로 충분) |
| 객체 윤곽선 근처 | 격자 + 객체 엣지 혼재 | 어려움 |
| 좁은 영역 (잎맥, 머리카락) | 패치 매칭 통계 부족 | 매우 어려움 |

→ **단일 알고리즘으로 모든 영역을 동등하게 처리할 수 없다** 는 관찰이 V3 의 동기.

### 1.3 평가 메트릭

| 약어 | 의미 | 방향 |
|------|------|------|
| `GridE` | FFT 격자 주파수 대역 에너지 | ↓ 낮을수록 좋음 |
| `EdgePres` | Scharr gradient 보존율 | ↑ 높을수록 좋음 |
| `BandSNR` | 격자 대역 신호 대 잡음비 | ↑ 높을수록 좋음 |
| `EdgeGrid` | 객체 엣지 5px 이내 격자 잔존 | ↓ 낮을수록 좋음 |
| `NarrowGrid` | 좁은 영역 (distance < 10px) 격자 잔존 | ↓ 낮을수록 좋음 |
| `ColorShift` | LAB a*, b* 평균 변화량 | ↓ 낮을수록 좋음 |

---

## 2. 전체 파이프라인

```
                ┌─────────────────────────────────────┐
                │   입력 BGR 이미지 (격자 포함)         │
                └────────────────┬────────────────────┘
                                 │
                                 ▼
        ┌────────────────────────────────────────────────┐
        │  STAGE 1: Split Radius                          │
        │  (Spectral Interpolation in FFT Domain)         │
        │                                                  │
        │  • FFT → 격자 주파수 peak 검출                   │
        │  • Peak ± radius 영역 보간                       │
        │  • IFFT                                          │
        │                                                  │
        │  강점: 평탄 영역의 주기 격자 효과적 제거          │
        │  약점: 좁은 영역 / 엣지 근처에서 잔존            │
        └────────────────────┬───────────────────────────┘
                             │
                             │  split_out
                             ▼
        ┌────────────────────────────────────────────────┐
        │  STAGE 2: Spatial-Adaptive NLM Blending         │
        │                                                  │
        │  • 원본에 Strong NLM 적용 (h=15)                 │
        │  • K-means(5) → distance transform → narrow_mask │
        │  • Blend: (1-w)·split_out + w·strong_nlm        │
        │                                                  │
        │  강점: 영역별로 적응형 강도 적용                 │
        │  핵심: narrow_mask 가 0~1 연속값                 │
        └────────────────────┬───────────────────────────┘
                             │
                             │  denoised
                             ▼
        ┌────────────────────────────────────────────────┐
        │  STAGE 3: Contour Enhancement                   │
        │                                                  │
        │  • BGR → LAB                                     │
        │  • L 채널만 처리 (색감 보존)                     │
        │  • Adaptive threshold + Scharr gradient          │
        │  • L_new = L + boost · edge_signal               │
        │  • LAB → BGR                                     │
        │                                                  │
        │  강점: Stage 1-2 에서 손실된 엣지 회복           │
        └────────────────────┬───────────────────────────┘
                             │
                             ▼
                ┌──────────────────────────────┐
                │   출력 BGR 이미지 (최종)      │
                └──────────────────────────────┘
```

---

## 3. Stage 1: Split Radius

### 3.1 핵심 아이디어

격자는 주기 T_x = 9, T_y = 7 픽셀의 곱셈성 패턴 → **FFT 에서 특정 주파수에 강한 peak** 를 형성:

```
주파수 도메인 peak 위치 (DC 중심):
  Horizontal peaks:  (±k · W/T_x, 0)   for k = 1, 2, ...
  Vertical peaks:    (0, ±k · H/T_y)   for k = 1, 2, ...
  Cross harmonics:   (±k · W/T_x, ±k · H/T_y)
```

여기서 W, H 는 이미지 너비/높이, k 는 harmonic 차수.

### 3.2 알고리즘

```
INPUT:  I (BGR image)
        T_x, T_y (grid periods)
        r (peak_radius, default=3)
        c (cross_peak_radius, default=1)

FOR each channel c in [B, G, R]:
    F ← FFT2D(I[:, :, c])
    F_shifted ← fftshift(F)

    H ← create_notch_mask(shape, T_x, T_y, r, c)
    F_filtered ← F_shifted · H

    # 보간: peak 위치를 주변 평균으로 대체
    F_filtered ← interpolate_notches(F_filtered, notch_positions)

    I_out[:, :, c] ← Real(IFFT2D(ifftshift(F_filtered)))

RETURN I_out
```

### 3.3 Notch Mask 생성

```python
def create_notch_mask(shape, T_x, T_y, peak_radius, cross_radius):
    H, W = shape
    cy, cx = H // 2, W // 2
    mask = np.ones((H, W), dtype=np.float32)

    # 가로 harmonics
    for k in range(1, n_harmonics + 1):
        fx = k * W / T_x
        for dx in [fx, -fx]:
            mask = suppress_disk(mask, (cx + dx, cy), peak_radius)

    # 세로 harmonics
    for k in range(1, n_harmonics + 1):
        fy = k * H / T_y
        for dy in [fy, -fy]:
            mask = suppress_disk(mask, (cx, cy + dy), peak_radius)

    # 교차 harmonics (작은 반경)
    for kx in range(1, n_harmonics + 1):
        for ky in range(1, n_harmonics + 1):
            fx, fy = kx * W / T_x, ky * H / T_y
            for dx, dy in product([fx, -fx], [fy, -fy]):
                mask = suppress_disk(mask, (cx + dx, cy + dy), cross_radius)

    return mask
```

### 3.4 한계

- **객체 윤곽선**: 객체 신호도 격자와 같은 주파수에 일부 에너지 → 함께 손실
- **좁은 영역**: 주파수 신호가 약함 → 격자 신호 분리 불충분
- → Stage 2 가 이 잔존을 보완

---

## 4. Stage 2: Spatial-Adaptive NLM

### 4.1 핵심 아이디어

**관찰**: 격자 잔존이 공간적으로 균일하지 않음 → 공간별 다른 강도의 후처리 필요.

**해결**: 좁은 영역(narrow region) 을 자동 검출하여 그 영역에만 Strong NLM 을 blending.

### 4.2 Narrow Region Mask 계산

#### 4.2.1 K-means 색 분할

```python
def compute_color_clusters(image_bgr, n_clusters=5):
    """원본 이미지를 색 기반 5개 클러스터로 분할."""
    pixels = image_bgr.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, _ = cv2.kmeans(
        pixels, n_clusters, None, criteria,
        attempts=3, flags=cv2.KMEANS_PP_CENTERS,
    )
    return labels.reshape(image_bgr.shape[:2])  # (H, W)
```

색 클러스터는 의미적 영역을 근사 (배경, 잎, 줄기, 꽃잎 등).

#### 4.2.2 Distance Transform per Cluster

```python
def compute_narrow_region_mask(image_bgr, config):
    """좁은 영역 [0, 1] 가중치 맵 생성."""
    H, W = image_bgr.shape[:2]
    labels = compute_color_clusters(image_bgr, config.n_clusters)

    narrow_mask = np.zeros((H, W), dtype=np.float32)

    for cluster_id in range(config.n_clusters):
        cluster_mask = (labels == cluster_id).astype(np.uint8)

        # 작은 클러스터는 노이즈로 간주, 무시
        if cluster_mask.sum() < config.min_cluster_area:
            continue

        # 클러스터 내부의 각 픽셀에 대해 경계까지 거리 측정
        dist = cv2.distanceTransform(cluster_mask, cv2.DIST_L2, maskSize=5)

        # 좁은 영역 (경계로부터 가까움) → 1 에 가까움
        narrow_contrib = np.clip(
            1.0 - dist / config.narrow_threshold,  # threshold=8.0
            0.0, 1.0,
        ) * cluster_mask  # 클러스터 내부에만 기여

        narrow_mask = np.maximum(narrow_mask, narrow_contrib)

    # 부드럽게 (sharp edge 방지)
    narrow_mask = cv2.GaussianBlur(
        narrow_mask, (0, 0), sigmaX=config.blur_sigma,  # sigma=2.0
    )

    return narrow_mask  # shape (H, W), values in [0, 1]
```

**직관**:
- 큰 평탄 영역 중심: distance 큼 → `narrow_contrib ≈ 0`
- 경계 근처 또는 좁은 영역: distance 작음 → `narrow_contrib ≈ 1`
- Gaussian blur 로 자연스러운 그라데이션

### 4.3 Strong NLM 적용

```python
def apply_strong_nlm(image_bgr, config):
    """원본 이미지에 강한 NLM 적용."""
    return cv2.fastNlMeansDenoisingColored(
        image_bgr,
        None,
        h=config.h_max,                          # luminance strength, default=15
        hColor=config.h_color_max,               # chrominance strength, default=15
        templateWindowSize=config.template_window,  # default=7
        searchWindowSize=config.search_window,      # default=21
    )
```

**왜 원본에 적용?**
- Split Radius 출력에 NLM 을 또 적용하면 디테일 손실 가중
- 원본에 강한 NLM 을 적용 → 격자가 균질화되어 사라짐
- narrow region 에서만 사용하므로 평탄 영역 손실 영향 없음

### 4.4 Blending

```python
def spatial_adaptive_nlm(image_bgr, base_processed_bgr, config):
    """Stage 2 메인 함수: narrow region 에 strong NLM blending."""

    # 1. Narrow mask 계산 (원본 기준)
    narrow_mask = compute_narrow_region_mask(image_bgr, config)

    # 2. Strong NLM (원본에 적용)
    strong_nlm = apply_strong_nlm(image_bgr, config)

    # 3. Blending
    w = narrow_mask[..., np.newaxis]  # (H, W, 1) for broadcasting
    base = base_processed_bgr.astype(np.float32)
    nlm = strong_nlm.astype(np.float32)

    result = (1.0 - w) * base + w * nlm

    return np.clip(result, 0, 255).astype(np.uint8)
```

**Blending 공식의 의미**:

| narrow_mask 값 | 의미 | 결과 |
|---------------|------|------|
| 0.0 | 완전 평탄 영역 | Split 결과 그대로 |
| 0.5 | 중간 (경계 근처) | 두 결과의 평균 |
| 1.0 | 좁은 영역 핵심 | Strong NLM 결과 그대로 |

### 4.5 알고리즘 흐름 요약

```
spatial_adaptive_nlm(image, split_out):
    1. labels ← KMeans(image, k=5)
    2. narrow_mask ← zero map
    3. FOR each cluster:
         IF cluster too small: SKIP
         dist ← distanceTransform(cluster_mask)
         narrow_mask ← max(narrow_mask, clip(1 - dist/8, 0, 1) · cluster_mask)
    4. narrow_mask ← GaussianBlur(narrow_mask, sigma=2)
    5. strong_nlm ← NLM(image, h=15)
    6. result ← (1 - narrow_mask) · split_out + narrow_mask · strong_nlm
    7. RETURN clip(result, 0, 255).uint8
```

---

## 5. Stage 3: Contour Enhancement

### 5.1 핵심 아이디어

Stage 1-2 의 격자 제거는 객체 윤곽선을 일부 흐리게 만듦 (저주파 손실).
**LAB 의 L 채널만** 강화하여 윤곽선을 회복하면서 **색감 (a*, b*) 은 보존**.

### 5.2 알고리즘

```python
def enhance_contours(original_bgr, denoised_bgr, boost=10.0,
                     block_size=15, thresh_c=6.0):
    """
    Args:
        original_bgr: 원본 (윤곽선 정보 추출용)
        denoised_bgr: Stage 2 출력 (강화 적용 대상)
        boost: 강화 강도
        block_size: adaptive threshold 블록 크기
        thresh_c: adaptive threshold 상수

    Returns:
        Contour 강화된 BGR 이미지
    """
    # 1. LAB 변환
    lab_orig = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2LAB)
    lab_denoised = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2LAB)
    L_orig = lab_orig[:, :, 0]
    L_denoised = lab_denoised[:, :, 0]

    # 2. 윤곽선 마스크 (adaptive threshold)
    contour_mask = cv2.adaptiveThreshold(
        L_orig, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, thresh_c,
    ) / 255.0  # → [0, 1]

    # 3. Scharr gradient (정확한 1차 미분)
    gx = cv2.Scharr(L_orig.astype(np.float32), cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(L_orig.astype(np.float32), cv2.CV_32F, 0, 1)
    gradient_magnitude = np.sqrt(gx**2 + gy**2)

    # 4. 정규화
    gradient_normalized = gradient_magnitude / (gradient_magnitude.max() + 1e-8)

    # 5. 윤곽선 신호 = 마스크 × 그래디언트
    edge_signal = contour_mask * gradient_normalized

    # 6. L 채널 강화
    L_new = L_denoised.astype(np.float32) + boost * edge_signal
    L_new = np.clip(L_new, 0, 255).astype(np.uint8)

    # 7. LAB → BGR
    lab_result = lab_denoised.copy()
    lab_result[:, :, 0] = L_new
    result_bgr = cv2.cvtColor(lab_result, cv2.COLOR_LAB2BGR)

    return result_bgr
```

### 5.3 디자인 선택의 이유

| 선택 | 이유 |
|------|------|
| L 채널만 강화 | a*, b* 보존 → 색감 변화 최소화 |
| Adaptive threshold | 평탄 영역 제외 (전역 임계값보다 강건) |
| Scharr (vs Sobel) | 더 정확한 1차 미분 (rotation-invariant) |
| 원본에서 그래디언트 추출 | 더 선명한 윤곽선 (denoised 보다 정확) |

---

## 6. 전체 Pseudo-code

### 6.1 메인 함수

```
ALGORITHM: V3_Grid_Removal
INPUT:  I (BGR image, shape HxWx3)
        config (WeaveRemovalConfig)
OUTPUT: I_out (BGR image, shape HxWx3)

----------------------------------------------------------
STAGE 1: Split Radius (Spectral Interpolation)
----------------------------------------------------------
1: split_out ← apply_split_radius(I, config)
   where apply_split_radius:
   a: FOR each channel c in [B, G, R]:
   b:     F ← FFT2(I[:, :, c])
   c:     F_shifted ← fftshift(F)
   d:     mask ← create_notch_mask(shape, T_x=9, T_y=7,
                                    peak_r=3, cross_r=1)
   e:     F_filtered ← interpolate_notches(F_shifted, mask)
   f:     split_out[:, :, c] ← Real(IFFT2(ifftshift(F_filtered)))
   g: RETURN split_out

----------------------------------------------------------
STAGE 2: Spatial-Adaptive NLM Blending
----------------------------------------------------------
2: narrow_mask ← compute_narrow_region_mask(I, config)
   where compute_narrow_region_mask:
   a: labels ← KMeans(I.reshape(-1, 3), k=5)
   b: narrow_mask ← zeros(H, W)
   c: FOR each cluster_id in [0, ..., 4]:
   d:     cluster_mask ← (labels == cluster_id)
   e:     IF sum(cluster_mask) < 100: CONTINUE
   f:     dist ← cv2.distanceTransform(cluster_mask, L2)
   g:     contrib ← clip(1 - dist/8.0, 0, 1) · cluster_mask
   h:     narrow_mask ← max(narrow_mask, contrib)
   i: narrow_mask ← GaussianBlur(narrow_mask, sigma=2.0)
   j: RETURN narrow_mask

3: strong_nlm ← cv2.fastNlMeansDenoisingColored(
                    I, h=15, hColor=15,
                    template=7, search=21)

4: w ← narrow_mask[..., newaxis]  # broadcast for 3 channels
5: denoised ← (1 - w) · split_out + w · strong_nlm
6: denoised ← clip(denoised, 0, 255).astype(uint8)

----------------------------------------------------------
STAGE 3: Contour Enhancement
----------------------------------------------------------
7: lab_orig ← BGR_to_LAB(I)
8: lab_denoised ← BGR_to_LAB(denoised)
9: L_orig ← lab_orig[:, :, 0]
10: L_denoised ← lab_denoised[:, :, 0]

11: contour_mask ← cv2.adaptiveThreshold(
                       L_orig, 255,
                       ADAPTIVE_MEAN_C, THRESH_BINARY_INV,
                       block_size=15, C=6.0) / 255.0

12: gx ← cv2.Scharr(L_orig, dx=1, dy=0)
13: gy ← cv2.Scharr(L_orig, dx=0, dy=1)
14: grad_mag ← sqrt(gx² + gy²)
15: grad_norm ← grad_mag / (max(grad_mag) + epsilon)

16: edge_signal ← contour_mask · grad_norm
17: L_new ← clip(L_denoised + 10.0 · edge_signal, 0, 255)

18: lab_result ← lab_denoised
19: lab_result[:, :, 0] ← L_new
20: I_out ← LAB_to_BGR(lab_result)

21: RETURN I_out
```

### 6.2 Helper: Notch Interpolation

```
FUNCTION: interpolate_notches(F_shifted, peak_positions, radius)
INPUT:  F_shifted (complex 2D FFT, DC-centered)
        peak_positions (list of (cx, cy) tuples)
        radius (suppress radius)
OUTPUT: F_interpolated (complex 2D)

1: F_out ← F_shifted.copy()
2: FOR each (px, py) in peak_positions:
3:     Y, X ← meshgrid(range(H), range(W))
4:     in_notch ← (X - px)² + (Y - py)² ≤ radius²
5:     IF count(in_notch) > 0:
6:         # 주변 (annulus) 평균으로 대체
7:         annulus ← radius² < (X-px)² + (Y-py)² ≤ (radius+1)²
8:         neighbor_mean ← mean(F_out[annulus])
9:         F_out[in_notch] ← neighbor_mean
10: RETURN F_out
```

### 6.3 Preset 정의

```
PRESET: V3
1: config ← WeaveRemovalConfig()
2: config.split_radius ← True
3: config.peak_radius ← 3
4: config.cross_peak_radius ← 1
5: config.use_nlm_adaptive ← True
6: config.nlm_h_base ← 10.0
7: config.nlm_h_max ← 15.0
8: config.nlm_h_color_base ← 10.0
9: config.nlm_h_color_max ← 15.0
10: config.nlm_narrow_threshold ← 8.0
11: config.nlm_n_clusters ← 5
12: config.nlm_min_cluster_area ← 100
13: config.nlm_blur_sigma ← 2.0
14: config.contour_boost ← 10.0
15: config.contour_block_size ← 15
16: config.contour_thresh_c ← 6.0
17: RETURN config
```

---

## 7. 파라미터 레퍼런스

### 7.1 Stage 1 (Split Radius)

| 파라미터 | 기본값 | 의미 | 튜닝 가이드 |
|---------|--------|------|------------|
| `T_x` | 9 | 가로 격자 주기 (px) | 직물 측정값. 자동 추정 가능 |
| `T_y` | 7 | 세로 격자 주기 (px) | 직물 측정값 |
| `peak_radius` | 3 | Notch 반경 (px) | 크게: 격자 더 제거 / 디테일 손실 |
| `cross_peak_radius` | 1 | 교차 harmonic 반경 | 보통 1 고정 |
| `n_harmonics` | 5 | 처리할 harmonic 차수 | 큰 격자엔 더 많이 |

### 7.2 Stage 2 (Spatial-Adaptive NLM)

| 파라미터 | 기본값 | 의미 | 튜닝 가이드 |
|---------|--------|------|------------|
| `h_max` | 15.0 | Strong NLM 강도 (luminance) | 크게: 더 강한 평활화 |
| `h_color_max` | 15.0 | Strong NLM 강도 (chrominance) | 보통 h_max 와 동일 |
| `narrow_threshold` | 8.0 | 좁은 영역 거리 임계값 (px) | 크게: 더 많은 영역이 narrow |
| `n_clusters` | 5 | K-means 클러스터 수 | 다색 이미지엔 더 많이 |
| `min_cluster_area` | 100 | 최소 클러스터 픽셀 수 | 노이즈 클러스터 제외 |
| `blur_sigma` | 2.0 | Mask Gaussian blur | 크게: 더 부드러운 전환 |
| `template_window` | 7 | NLM 패치 크기 | OpenCV 기본 |
| `search_window` | 21 | NLM 검색 영역 | OpenCV 기본 |

### 7.3 Stage 3 (Contour Enhancement)

| 파라미터 | 기본값 | 의미 | 튜닝 가이드 |
|---------|--------|------|------------|
| `contour_boost` | 10.0 | L 채널 강화 강도 | 크게: 더 선명, 격자 재주입 위험 |
| `contour_block_size` | 15 | Adaptive threshold 블록 | 크게: 더 큰 윤곽선 |
| `contour_thresh_c` | 6.0 | Adaptive threshold 상수 | 크게: 더 적은 윤곽선 |

---

## 8. Computational Complexity

이미지 크기 N × N 기준:

| Stage | 연산 | 복잡도 | 비고 |
|-------|------|--------|------|
| Stage 1 | 3 × FFT2D + 보간 | O(3 · N² log N) | 채널별 FFT |
| Stage 2.1 | K-means (k=5, 20 iter) | O(20 · 5 · N²) | 색 분할 |
| Stage 2.2 | Distance transform × 5 | O(5 · N²) | 클러스터별 |
| Stage 2.3 | NLM (template=7, search=21) | O(N² · 7² · 21²) | 가장 무거움 |
| Stage 2.4 | Blending | O(N²) | 픽셀 단위 |
| Stage 3 | LAB 변환 + Scharr | O(N²) | 가벼움 |

**병목**: Stage 2.3 (NLM). 일반적으로 전체의 60-80% 시간 점유.

**최적화 방안**:
- NLM 을 narrow region 에만 적용 (현재는 전체 → masked 결과만 사용)
- GPU 가속 (cv2.cuda.fastNlMeansDenoisingColored)
- Downsample → NLM → upsample (정확도 약간 손실)

---

## 9. 구현 매핑

### 9.1 소스 파일 ↔ 알고리즘

| Pseudo-code 단계 | 소스 파일 | 함수/클래스 |
|------------------|----------|------------|
| Stage 1 전체 | `src/kp3d/modules/weave_removal/spectral.py` | `process_image_patchwise()` |
| Stage 2.1 (K-means + distance) | `src/kp3d/modules/weave_removal/nlm_adaptive.py` | `compute_narrow_region_mask()` |
| Stage 2.2-4 (NLM + blend) | `src/kp3d/modules/weave_removal/nlm_adaptive.py` | `spatial_adaptive_nlm()` |
| Stage 3 (Contour) | `src/kp3d/modules/weave_removal/contour.py` | `enhance_contours()` |
| 전체 파이프라인 통합 | `src/kp3d/modules/weave_removal/base.py` | `WeaveRemovalModule.process_bgr()` |
| V3 Preset 정의 | `src/kp3d/modules/weave_removal/base.py` | `WeaveRemovalPreset.V3.apply_preset()` |

### 9.2 사용 예제

```python
from kp3d.modules.weave_removal import (
    WeaveRemovalModule,
    WeaveRemovalConfig,
    WeaveRemovalPreset,
)

# 방법 1: Preset 사용 (권장)
module = WeaveRemovalModule(
    config=WeaveRemovalConfig(preset=WeaveRemovalPreset.V3)
)
result, confidence = module.process_bgr(image_bgr)

# 방법 2: 직접 config 설정
config = WeaveRemovalConfig(
    split_radius=True,
    use_nlm_adaptive=True,
    nlm_h_max=15.0,
    nlm_narrow_threshold=8.0,
    contour_boost=10.0,
)
module = WeaveRemovalModule(config=config)
result, confidence = module.process_bgr(image_bgr)

# 개별 stage 직접 호출
from kp3d.modules.weave_removal import (
    spatial_adaptive_nlm,
    SpatialAdaptiveNLMConfig,
    compute_narrow_region_mask,
)

# Stage 2 만 단독 실행 (테스트/디버깅용)
nlm_cfg = SpatialAdaptiveNLMConfig(h_max=15.0)
narrow_mask = compute_narrow_region_mask(image_bgr, nlm_cfg)
denoised = spatial_adaptive_nlm(
    image_bgr=image_bgr,
    base_processed_bgr=split_radius_output,
    config=nlm_cfg,
)
```

### 9.3 단위 테스트

`tests/test_weave_removal_v3.py` 의 20개 테스트:

| 테스트 클래스 | 검증 항목 |
|-------------|----------|
| `TestV3PresetConfig` | V3 preset 의 필드 값 |
| `TestLegacyPresetPreservation` | QUALITY/CLEAN preset 불변 |
| `TestNarrowMaskComputation` | narrow_mask 출력 검증 |
| `TestSpatialAdaptiveNLM` | NLM 함수 동작 |
| `TestWeaveRemovalModuleV3` | V3 end-to-end |
| `TestWeaveRemovalModuleLegacy` | Legacy preset 회귀 |
| `TestConfigParameterExposure` | 파라미터 노출 검증 |

---

## 부록 A: 메트릭 정의

### A.1 GridE (Grid Energy)

격자 주파수 대역의 평균 에너지:

```
GridE = mean(|FFT(I)|² in grid_band) / mean(|FFT(I)|²)
```

`grid_band` = 격자 harmonic 주파수 근방 (±2px in frequency domain)

### A.2 EdgePres (Edge Preservation)

원본 대비 결과의 Scharr gradient 보존율:

```
EdgePres = corr(Scharr(I_clean), Scharr(I_result))
```

### A.3 NarrowGrid

좁은 영역 (distance < 10px) 에서만 측정한 GridE.

### A.4 EdgeGrid

객체 엣지 (Canny edge dilation 5px) 근처에서 측정한 GridE.

---

## 부록 B: 알고리즘 도출 과정

V3 는 9차에 걸친 실험 라운드의 결과:

| 라운드 | 핵심 발견 | V3 에 반영된 것 |
|--------|---------|-------------|
| #1-2 | NL-Means 단독은 Split Radius 보다 약함 | Split Radius 를 Stage 1 로 채택 |
| #3 | OEE (Object-Edge-Only) 효과적 | (V3 에선 미채택, 기존 Contour 사용) |
| #4 | 격자가 채색되어 있음 (a* 39%, b* 76%) | NLM 의 hColor 활성화 |
| #6 | 격자 재주입 문제 | Contour boost 적정값 (10.0) |
| #7 | 좁은 영역/엣지 격자 잔존 | Stage 2 도입 (R 변형) |
| #9 | V2 (OEE+Contour) 는 격자 재주입 | V3 (Contour only) 선택 |

상세는 `work_reports/2026-05/WORK_REPORT_260522_restoration_v3_nlm_contour.md` 참조.

---

**작성**: Claude (Opus 4)
**라이선스**: 프로젝트 라이선스 준수
