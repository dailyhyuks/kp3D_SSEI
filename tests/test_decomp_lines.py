"""Scale-space 선 검출 테스트."""
import numpy as np
import pytest

from kp3d.modules.decomposition.lines import detect_lines, measure_line_widths


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


def test_width_measurement_on_known_line():
    """폭 5px 직선의 스켈레톤 폭 측정값이 5±1이어야 한다."""
    mask = np.zeros((128, 128), dtype=bool)
    mask[60:65, 10:118] = True
    skeleton, width_map = measure_line_widths(mask)
    widths = width_map[skeleton]
    assert widths.size > 0
    assert 4.0 <= float(np.median(widths)) <= 6.0


def test_small_components_removed():
    """대각선 0.5% 미만 길이의 점 잡음은 스켈레톤에서 제거되어야 한다."""
    mask = np.zeros((200, 200), dtype=bool)
    mask[100, 100] = True          # 1px 잡음
    mask[50:53, 20:180] = True     # 실제 선
    skeleton, _ = measure_line_widths(mask)
    assert not skeleton[100, 100]
    assert skeleton[51, 60:140].any()
