"""레이어 분리/재합성 테스트: L over C ≈ I 불변식."""
import numpy as np

from kp3d.modules.decomposition.lines import detect_lines, measure_line_widths
from kp3d.modules.decomposition.split import recompose, split_layers


def _painting_like() -> np.ndarray:
    """어두운 선이 있는 합성 컬러 이미지 (BGR uint8)."""
    img = np.full((128, 128, 3), (180, 200, 210), dtype=np.uint8)
    img[40:60, :, :] = (90, 140, 200)      # 색 영역
    img[60:63, 10:118, :] = (40, 40, 50)   # 먹선 (폭 3)
    return img


def _run_split(img):
    gray = img.astype(np.float64).mean(axis=2)
    response, mask = detect_lines(gray, min_width=1.0, max_width=8.0)
    skeleton, width_map = measure_line_widths(mask)
    return split_layers(img, response, mask, skeleton, width_map)


def test_color_layer_untouched_outside_inpaint_mask():
    """inpaint 마스크 밖의 C는 원본과 완전 일치해야 한다."""
    img = _painting_like()
    line_alpha, color_layer, inpaint_mask = _run_split(img)
    outside = ~inpaint_mask
    assert np.array_equal(color_layer[outside], img[outside])


def test_alpha_zero_outside_line_region():
    """선 영역 밖의 alpha는 0이어야 한다."""
    img = _painting_like()
    line_alpha, _, inpaint_mask = _run_split(img)
    assert float(line_alpha[~inpaint_mask].max()) == 0.0


def test_alpha_positive_on_line():
    """선 위의 alpha는 유의미하게 커야 한다."""
    img = _painting_like()
    line_alpha, _, _ = _run_split(img)
    assert float(line_alpha[61, 20:100].mean()) > 0.5


def test_recompose_exact_outside_line():
    """불변식: alpha==0 픽셀에서 재합성 결과는 원본과 완전 일치."""
    img = _painting_like()
    line_alpha, color_layer, _ = _run_split(img)
    rec = recompose(img, line_alpha, color_layer)
    zero = line_alpha == 0.0
    assert np.array_equal(rec[zero], img[zero])


def test_recompose_small_global_residual():
    """전체 평균 재합성 잔차가 3 미만이어야 한다 (8bit 기준)."""
    img = _painting_like()
    line_alpha, color_layer, _ = _run_split(img)
    rec = recompose(img, line_alpha, color_layer)
    residual = np.abs(rec.astype(np.float64) - img.astype(np.float64)).mean()
    assert residual < 3.0
