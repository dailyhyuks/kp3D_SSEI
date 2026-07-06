"""decompose() 전체 오케스트레이션 테스트."""
import os

import numpy as np
import pytest

from kp3d.modules.decomposition.decompose import (
    DecompositionResult,
    decompose,
    recompose_result,
)


def _weave_painting() -> np.ndarray:
    """직조 텍스처 + 색 영역 + 먹선이 있는 합성 한국화 (BGR uint8)."""
    xx, yy = np.meshgrid(np.arange(256), np.arange(256))
    base = 190 + 8 * np.sin(2 * np.pi * xx / 7) + 8 * np.sin(2 * np.pi * yy / 7)
    img = np.stack([base * 0.95, base, base * 1.02], axis=2)
    img[80:140, 40:216, :] = (120, 160, 200)   # 색 영역
    img[140:144, 40:216, :] = (45, 45, 55)     # 먹선 (폭 4)
    img[80:140, 40:43, :] = (45, 45, 55)       # 세로 먹선 (폭 3)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_decompose_returns_consistent_result():
    img = _weave_painting()
    result = decompose(img)
    assert isinstance(result, DecompositionResult)
    assert result.line_alpha.shape == img.shape[:2]
    assert result.color_layer.shape == img.shape
    assert result.noise_sigma >= 0.0
    # 직조 주기 7px 검출 (±1)
    assert abs(result.weave.period_x - 7.0) <= 1.0


def test_decompose_finds_ink_lines():
    """먹선 중심부가 선 마스크에 포함되어야 한다."""
    result = decompose(_weave_painting())
    assert float(result.line_mask[141:143, 80:180].mean()) > 0.8


def test_color_layer_has_no_line():
    """C 레이어의 먹선 위치는 주변 색으로 채워져 있어야 한다 (어둡지 않음)."""
    result = decompose(_weave_painting())
    line_region = result.color_layer[141, 80:180, :].astype(np.float64)
    assert float(line_region.mean()) > 100.0  # 먹선 원색(~48)보다 훨씬 밝음


def test_invariant_recompose_exact_outside_lines():
    """불변식: 선 영역 외 재합성 == 원본 (설계 섹션 4.4 계약 0->1)."""
    img = _weave_painting()
    result = decompose(img)
    rec = recompose_result(img, result)
    zero = result.line_alpha == 0.0
    assert np.array_equal(rec[zero], img[zero])


_REAL_IMAGE = os.path.join("data", "ablation_study", "images", "1_0004.png")


@pytest.mark.skipif(not os.path.exists(_REAL_IMAGE), reason="실 데이터 없음")
def test_smoke_on_real_painting():
    """실제 한국화에서 예외 없이 완주하고 불변식을 지켜야 한다."""
    import cv2

    img = cv2.imread(_REAL_IMAGE)
    assert img is not None
    result = decompose(img)
    rec = recompose_result(img, result)
    zero = result.line_alpha == 0.0
    assert np.array_equal(rec[zero], img[zero])
    # 선이 하나라도 검출되어야 한다 (구륵법 그림 전제)
    assert result.skeleton.any()
