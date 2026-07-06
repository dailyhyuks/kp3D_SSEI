"""Stage 1 v2 전 구간 통합 테스트 (실제 v1 경로 포함)."""
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("torch")  # v1 경로가 torch를 요구

from kp3d.modules.weave_removal_v2 import (
    estimate_lattice,
    self_competition_gate,
    weave_band_energy,
)

_REAL_IMAGE = Path("data/ablation_study/images/1_0004.png")


def _weave_painting_with_lines(h: int = 256, w: int = 256) -> np.ndarray:
    """256x256: 직조 주기 8/12px로 충분한 격자 기저 추정."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    base = 100.0 + 60.0 * np.exp(
        -((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (2 * (h / 4) ** 2)
    )
    weave = 12.0 * np.cos(2 * np.pi * yy / 8.0) + 12.0 * np.cos(2 * np.pi * xx / 12.0)
    gray = np.clip(base + weave, 0, 255).astype(np.uint8)
    img = np.stack([gray, gray, gray], axis=-1)
    cv2.line(img, (32, 32), (224, 32), (20, 20, 20), 3)
    cv2.circle(img, (128, 128), 60, (30, 30, 30), 3)
    return img


def test_gate_end_to_end_on_synthetic_weave():
    img = _weave_painting_with_lines()
    result = self_competition_gate(img)
    assert result.winner in ("v2", "v1")
    assert result.restored.shape == img.shape
    assert result.restored.dtype == np.uint8
    assert result.noise_sigma >= 0.0
    gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lattice = estimate_lattice(gray0)
    # 격자 기저가 검출되면 직조 에너지가 감소해야 함
    if lattice.basis.shape[0] > 0:
        e0 = weave_band_energy(gray0, lattice)
        gray1 = cv2.cvtColor(result.restored, cv2.COLOR_BGR2GRAY).astype(np.float32)
        e1 = weave_band_energy(gray1, lattice)
        assert e1 < e0  # 어느 경로가 이기든 직조 에너지는 감소해야 함


@pytest.mark.skipif(not _REAL_IMAGE.exists(), reason="실이미지 데이터 없음")
def test_gate_smoke_on_real_image():
    img = cv2.imread(str(_REAL_IMAGE), cv2.IMREAD_COLOR)
    assert img is not None
    result = self_competition_gate(img)
    assert result.restored.shape == img.shape
    assert result.restored.dtype == np.uint8
    gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lattice = estimate_lattice(gray0)
    if lattice.basis.shape[0] > 0:
        e0 = weave_band_energy(gray0, lattice)
        gray1 = cv2.cvtColor(result.restored, cv2.COLOR_BGR2GRAY).astype(np.float32)
        e1 = weave_band_energy(gray1, lattice)
        assert e1 <= e0  # 직조 에너지가 늘어나면 안 됨
