"""patchmatch.py 테스트: 패치 크기 유도(상관 길이), ANN 정확 사본 발견."""
import cv2
import numpy as np

from kp3d.modules.ssei_v2.patchmatch import derive_patch_size, patchmatch


def test_derive_patch_size_bounds_and_order():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((64, 64))
    smooth = cv2.GaussianBlur(noise, (0, 0), 4.0)
    valid = np.ones((64, 64), dtype=bool)
    p_n = derive_patch_size(noise, valid)
    p_s = derive_patch_size(smooth, valid)
    for p in (p_n, p_s):
        assert p % 2 == 1 and 3 <= p <= 64 // 4
    # 상관 길이가 길수록 패치가 크다
    assert p_s > p_n


def test_derive_patch_size_constant_image_min():
    g = np.full((40, 40), 7.0)
    assert derive_patch_size(g, np.ones((40, 40), dtype=bool)) == 3


def test_patchmatch_finds_exact_copies():
    rng = np.random.default_rng(2)
    base = rng.random((8, 8, 3))
    img = np.tile(base, (5, 5, 1))  # 주기 8 텍스처 — 정확 사본 다수
    tm = np.zeros((40, 40), dtype=bool)
    tm[10:20, 5:12] = True
    pm = np.zeros((40, 40), dtype=bool)
    pm[3:37, 23:37] = True
    nnf, dists = patchmatch(img, tm, pm, patch_size=5, noise_sigma=1e-6)
    ys, xs = np.nonzero(tm)
    # 대응은 항상 pool 안
    assert pm[nnf[ys, xs, 0], nnf[ys, xs, 1]].all()
    # 정확 사본이 존재하므로 전파+랜덤 탐색이 SSD 0을 찾는다
    assert float(np.median(dists[tm])) < 1e-9
