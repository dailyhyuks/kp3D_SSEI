"""gate.py 테스트: 자가 경쟁 게이트 (v1/v2 경로는 monkeypatch로 대체).

이미지는 128x128로 고정한다: 프록시 배율이 max(0.25, 64/128)=0.5가 되어
직조 주기 8/12px가 프록시에서 4/6px로 보존된다. (256px면 배율 0.25에서
주기 8px가 나이퀴스트 2px에 걸려 INTER_AREA 축소로 소멸 -> 항상 동률.)
"""
import cv2
import numpy as np

import kp3d.modules.weave_removal_v2.gate as gate_mod
from kp3d.modules.weave_removal_v2 import self_competition_gate
from kp3d.modules.weave_removal_v2.gate import _proxy_scale


def _painting_pair(h: int = 128, w: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """(직조 포함 이미지, 직조 없는 정답 이미지) 쌍. 선·베이스는 동일."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    base = 110.0 + 50.0 * np.exp(
        -((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (2 * (h / 4) ** 2)
    )
    weave = 12.0 * np.cos(2 * np.pi * yy / 8.0) + 12.0 * np.cos(2 * np.pi * xx / 12.0)

    def _finish(field: np.ndarray) -> np.ndarray:
        gray = np.clip(field, 0, 255).astype(np.uint8)
        img = np.stack([gray, gray, gray], axis=-1)
        # Light brushstrokes (±3 from base value ~110) that preserve weave pattern
        cv2.line(img, (8, 8), (120, 8), (113, 113, 113), 1)
        cv2.line(img, (8, 8), (8, 120), (113, 113, 113), 1)
        cv2.circle(img, (64, 64), 12, (107, 107, 107), 1)
        return img

    return _finish(base + weave), _finish(base)


def _fake_path(clean_full: np.ndarray):
    """어떤 크기 입력이 와도 정답(clean)을 그 크기로 반환하는 페이크 경로."""

    def run(image: np.ndarray) -> np.ndarray:
        if image.shape[:2] == clean_full.shape[:2]:
            return clean_full.copy()
        return cv2.resize(
            clean_full, (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    return run


def test_proxy_scale_derivation():
    assert _proxy_scale((1024, 1024), 64) == 0.25          # 1/4 프록시
    assert _proxy_scale((128, 200), 64) == 0.5             # 64/128 클램프
    assert _proxy_scale((48, 48), 64) == 1.0               # min(1.0, ...) 상한


def test_v2_wins_when_v1_is_identity(monkeypatch):
    img, clean = _painting_pair()
    monkeypatch.setattr(gate_mod, "_run_v2", _fake_path(clean))
    monkeypatch.setattr(gate_mod, "_run_v1", lambda image: image.copy())
    result = self_competition_gate(img)
    assert result.winner == "v2"
    assert result.quality_v2 > result.quality_v1
    assert result.restored.shape == img.shape
    assert np.array_equal(result.restored, clean)  # 승자 경로가 전체 해상도에 적용됨


def test_v1_wins_when_v2_is_identity(monkeypatch):
    img, clean = _painting_pair()
    monkeypatch.setattr(gate_mod, "_run_v2", lambda image: image.copy())
    monkeypatch.setattr(gate_mod, "_run_v1", _fake_path(clean))
    result = self_competition_gate(img)
    assert result.winner == "v1"
    assert result.quality_v1 > result.quality_v2
    assert np.array_equal(result.restored, clean)


def test_tie_falls_back_to_v1(monkeypatch):
    img, _clean = _painting_pair()
    monkeypatch.setattr(gate_mod, "_run_v2", lambda image: image.copy())
    monkeypatch.setattr(gate_mod, "_run_v1", lambda image: image.copy())
    result = self_competition_gate(img)
    assert result.winner == "v1"                            # 동률 -> v1 안전망


def test_no_weave_image_falls_back_to_v1(monkeypatch):
    img = np.full((128, 128, 3), 128, dtype=np.uint8)
    monkeypatch.setattr(gate_mod, "_run_v2", lambda image: image.copy())
    monkeypatch.setattr(gate_mod, "_run_v1", lambda image: image.copy())
    result = self_competition_gate(img)
    assert result.winner == "v1"                            # E_before=0 -> 동률 -> v1
    assert result.noise_sigma >= 0.0
