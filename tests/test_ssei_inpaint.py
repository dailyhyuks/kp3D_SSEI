"""inpaint.py 테스트: Phase C 통합, G2 조인트 기계 검증, 가림 밖 불변."""
import numpy as np
from scipy.ndimage import binary_dilation

from kp3d.modules.decomposition import recompose
from kp3d.modules.ssei_v2.inpaint import InpaintingResult, inpaint


def _scene(h=64, w=96):
    """수평 먹선(y=32, 폭 3) + 녹색 배경 + 중앙 가림 상자."""
    color = np.zeros((h, w, 3), dtype=np.uint8)
    color[:] = (40, 160, 40)
    line_alpha = np.zeros((h, w), dtype=np.float32)
    skeleton = np.zeros((h, w), dtype=bool)
    width_map = np.zeros((h, w), dtype=np.float32)
    y = 32
    skeleton[y, 8:88] = True
    width_map[y, 8:88] = 3.0
    line_alpha[y - 1:y + 2, 8:88] = 1.0
    image = color.copy()
    image[line_alpha > 0] = (20, 20, 20)
    occ = np.zeros((h, w), dtype=bool)
    occ[24:40, 40:56] = True
    # 가림 내부 지식 소거 + 가림 픽셀은 마젠타(가림 물체)로 오염
    line_alpha[occ] = 0.0
    skeleton[occ] = False
    width_map[occ] = 0.0
    image[occ] = (255, 0, 255)
    color[occ] = (255, 0, 255)
    return image, color, line_alpha, skeleton, width_map, occ


def _run():
    image, color, la, sk, wm, occ = _scene()
    res = inpaint(image, color, la, sk, wm, occ, noise_sigma=1.0)
    return image, color, la, sk, wm, occ, res


def test_inpaint_connects_line_and_fills_color():
    image, color, la, sk, wm, occ, res = _run()
    assert isinstance(res, InpaintingResult)
    assert res.line.connections                  # 획이 실제로 연결됨
    assert res.line.line_alpha[32, 48] > 0.5     # 가림 중앙에 선 복원
    px = res.inpainted[26, 48]                   # 선에서 떨어진 가림 내부
    assert int(px[1]) > int(px[0]) and int(px[1]) > 100  # 녹색 배경 회복 (마젠타 아님)
    assert res.color.by_construction_violations == 0
    assert res.by_construction_violations == 0


def test_g2_joint_machine_verification():
    *_, res = _run()
    # 조인트 접선·곡률 — biclothoid/quintic 모두 경계에서 해석적으로 정확
    assert res.line.connections
    assert res.g2_tangent_max < 1e-6
    assert res.g2_curvature_max < 1e-6


def test_outside_occlusion_invariant():
    image, color, la, sk, wm, occ, res = _run()
    baseline = recompose(image, la, color)
    # 스탬프가 가림 밖 endpoint 주변(창=선폭×2, 정규화 규칙)까지 닿을 수 있어 링 제외
    ring = binary_dilation(occ, iterations=6)
    assert np.array_equal(res.inpainted[~ring], baseline[~ring])


def test_public_api_reexport():
    import kp3d.modules.ssei_v2 as m

    for name in ("Endpoint", "connect_g2", "match_endpoints", "complete_lines",
                 "build_piece_pools", "derive_patch_size", "patchmatch",
                 "fill_color", "inpaint", "InpaintingResult"):
        assert hasattr(m, name)
