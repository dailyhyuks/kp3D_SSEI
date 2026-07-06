"""removal.py 테스트: 패치 유도, 대역 에너지 보정, 직물 제거 코어."""
import cv2
import numpy as np
import pytest

from kp3d.modules.weave_removal_v2 import (
    LatticeResult,
    derive_patch_size,
    estimate_lattice,
    remove_weave,
    weave_band_energy,
)


def _lattice_from_basis(basis: np.ndarray) -> LatticeResult:
    b = np.asarray(basis, dtype=np.float64)
    freq = np.linalg.inv(b).T if b.shape[0] == 2 else b / np.linalg.norm(b[0]) ** 2
    return LatticeResult(basis=b, freq_basis=freq, strength=1.0)


def _weave_painting(h: int = 256, w: int = 256, amp: float = 12.0) -> np.ndarray:
    """비주기 베이스(가우시안 블롭) + 축 정렬 직조(주기 8/12) 3채널 이미지."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    base = 100.0 + 60.0 * np.exp(
        -((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (2 * (h / 4) ** 2)
    )
    weave = amp * np.cos(2 * np.pi * yy / 8.0) + amp * np.cos(2 * np.pi * xx / 12.0)
    gray = np.clip(base + weave, 0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def test_derive_patch_size_is_eight_periods_power_of_two():
    lat8 = _lattice_from_basis([[8.0, 0.0], [0.0, 8.0]])
    assert derive_patch_size(lat8, (512, 512)) == 64
    lat12 = _lattice_from_basis([[12.0, 0.0], [0.0, 12.0]])
    assert derive_patch_size(lat12, (512, 512)) == 128  # 96 -> 128
    # 상한: min(H,W)=100 -> 2^floor(log2(100)) = 64
    lat40 = _lattice_from_basis([[40.0, 0.0], [0.0, 40.0]])
    assert derive_patch_size(lat40, (100, 100)) == 64


def test_weave_band_energy_matches_sinusoid_rms():
    h = w = 256
    amp = 10.0
    yy = np.mgrid[0:h, 0:w][0].astype(np.float64)
    gray = (128.0 + amp * np.cos(2 * np.pi * yy / 8.0)).astype(np.float32)
    lattice = estimate_lattice(gray)
    assert lattice.basis.shape[0] >= 1
    energy = weave_band_energy(gray, lattice)
    expected = amp / np.sqrt(2.0)
    assert abs(energy - expected) < 0.15 * expected


def test_remove_weave_reduces_band_energy():
    img = _weave_painting()
    gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lattice = estimate_lattice(gray0)
    e0 = weave_band_energy(gray0, lattice)
    result = remove_weave(img)
    assert result.iterations >= 1
    gray1 = cv2.cvtColor(result.cleaned, cv2.COLOR_BGR2GRAY).astype(np.float32)
    e1 = weave_band_energy(gray1, lattice)
    assert e1 < 0.5 * e0
    assert result.cleaned.shape == img.shape
    assert result.cleaned.dtype == np.uint8


def test_remove_weave_preserves_content():
    h = w = 256
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    base = 100.0 + 60.0 * np.exp(
        -((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (2 * (h / 4) ** 2)
    )
    img = _weave_painting()
    result = remove_weave(img)
    err_before = np.mean(np.abs(img[:, :, 0].astype(np.float64) - base))
    err_after = np.mean(np.abs(result.cleaned[:, :, 0].astype(np.float64) - base))
    assert err_after < err_before


def test_remove_weave_no_lattice_is_identity():
    img = np.full((128, 128, 3), 128, dtype=np.uint8)
    result = remove_weave(img)
    assert result.iterations == 0
    assert np.array_equal(result.cleaned, img)
