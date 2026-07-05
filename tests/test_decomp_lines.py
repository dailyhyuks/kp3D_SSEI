"""Scale-space 선 검출 테스트."""
import numpy as np
import pytest

from kp3d.modules.decomposition.lines import detect_lines


def _dark_line_image(width: int = 3) -> np.ndarray:
    img = np.full((128, 128), 200.0)
    img[60 : 60 + width, 10:118] = 60.0
    return img


def test_detects_dark_line():
    """폭 3px 먹선을 90% 이상 커버해야 한다."""
    _, mask = detect_lines(_dark_line_image(), min_width=1.0, max_width=8.0)
    assert float(mask[61, 20:100].mean()) > 0.9


def test_low_false_positive_on_background():
    """선에서 먼 배경의 오검출률이 5% 미만이어야 한다."""
    _, mask = detect_lines(_dark_line_image(), min_width=1.0, max_width=8.0)
    assert float(mask[[10, 110], :].mean()) < 0.05


def test_detects_both_thin_and_thick_lines():
    """폭 2px와 6px 선이 공존해도 둘 다 검출해야 한다 (scale-space 목적)."""
    img = np.full((128, 128), 200.0)
    img[30:32, 10:118] = 60.0   # 폭 2
    img[80:86, 10:118] = 60.0   # 폭 6
    _, mask = detect_lines(img, min_width=1.0, max_width=8.0)
    assert float(mask[31, 20:100].mean()) > 0.9
    assert float(mask[83, 20:100].mean()) > 0.9


def test_detect_lines_rejects_invalid_input():
    """비2D 입력과 잘못된 폭 범위는 ValueError를 발생시켜야 한다."""
    img = np.full((64, 64), 255.0, dtype=np.float32)
    with pytest.raises(ValueError):
        detect_lines(np.zeros(16, dtype=np.float32), 1.0, 4.0)
    with pytest.raises(ValueError):
        detect_lines(img, 0.0, 4.0)
    with pytest.raises(ValueError):
        detect_lines(img, 5.0, 4.0)
