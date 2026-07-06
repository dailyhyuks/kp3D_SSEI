"""SSEI 2.0 통합 — 프로토콜 B (스펙 §5.1): 완전 작품 합성 → 지식 소거 → 복원 정량."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from scipy.ndimage import binary_dilation

from kp3d.modules.decomposition import decompose
from kp3d.modules.ssei_v2 import inpaint


def _artwork(h=128, w=128):
    """합성 '완전' 작품: 2색 배경 + 완만한 사인 곡선 먹선 (반지름 2 디스크)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :w // 2] = (60, 170, 60)
    img[:, w // 2:] = (60, 60, 170)
    xs = np.arange(10, w - 10)
    ys = h / 2.0 + 12.0 * np.sin(2.0 * np.pi * (xs - 10) / (w - 20))
    for x, y in zip(xs, ys):
        cv2.circle(img, (int(x), int(round(y))), 2, (20, 20, 20), -1)
    return img


def _psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0.0 else 10.0 * np.log10(255.0 ** 2 / mse)


def test_protocol_b_psnr_and_skeleton():
    gt = _artwork()
    dec = decompose(gt)
    occ = np.zeros(gt.shape[:2], dtype=bool)
    occ[52:76, 52:76] = True
    # 지식 소거 (가림 내부 정보 삭제) + 가림 물체(마젠타) 오염
    la = dec.line_alpha.copy(); la[occ] = 0.0
    sk = dec.skeleton.copy(); sk[occ] = False
    wm = dec.width_map.copy(); wm[occ] = 0.0
    col = dec.color_layer.copy(); col[occ] = (255, 0, 255)
    img_in = gt.copy(); img_in[occ] = (255, 0, 255)

    res = inpaint(img_in, col, la, sk, wm, occ, dec.noise_sigma)

    # ① PSNR(가림 내부): 가시 평균색 채움 베이스라인 초과
    base = gt.copy()
    base[occ] = np.rint(gt[~occ].reshape(-1, 3).mean(axis=0)).astype(np.uint8)
    assert _psnr(res.inpainted[occ], gt[occ]) > _psnr(base[occ], gt[occ])

    # ② 스켈레톤 재현율(가림 내부, 1px 팽창 허용): GT 획 자취 회복
    gt_sk = dec.skeleton & occ  # 소거 전 분해가 본 가림 내부 획 자취
    if np.any(gt_sk):
        cover = binary_dilation(res.line.skeleton, iterations=1)
        recall = (float(np.count_nonzero(gt_sk & cover))
                  / float(np.count_nonzero(gt_sk)))
        assert recall > 0.7

    # ③ 기계 검증 지표
    assert res.by_construction_violations == 0
    assert res.g2_tangent_max < 1e-6
    assert res.g2_curvature_max < 1e-6
