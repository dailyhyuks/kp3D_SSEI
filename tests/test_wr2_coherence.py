"""패치 위상 결맞음 테스트."""
import numpy as np

from kp3d.modules.weave_removal_v2.coherence import phase_coherence


def test_global_sinusoid_is_coherent():
    size, s, n = 256, 64, 40
    rng = np.random.default_rng(7)
    xx = np.arange(size, dtype=np.float64)[None, :]
    img = 128.0 + 10.0 * np.cos(2.0 * np.pi * xx / 8.0) * np.ones((size, 1))
    offs = rng.integers(0, size - s, size=(n, 2)).astype(np.float64)
    patches = np.stack(
        [img[int(y):int(y) + s, int(x):int(x) + s] for y, x in offs]
    )
    pf = np.fft.fft2(patches)
    r, w = phase_coherence(pf, offs, np.array([0.0, 1.0 / 8.0]), s)
    assert r > 0.9
    assert w.shape == (n,)
    assert np.all((w >= 0.0) & (w <= 1.0))
    assert w.mean() > 0.9


def test_random_phase_patches_are_incoherent():
    s, n = 64, 40
    rng = np.random.default_rng(11)
    xx = np.arange(s, dtype=np.float64)[None, :]
    patches = np.stack([
        128.0 + 10.0 * np.cos(2.0 * np.pi * xx / 8.0
                              + rng.uniform(0.0, 2.0 * np.pi)) * np.ones((s, 1))
        for _ in range(n)
    ])
    offs = np.zeros((n, 2), dtype=np.float64)
    pf = np.fft.fft2(patches)
    r, w = phase_coherence(pf, offs, np.array([0.0, 1.0 / 8.0]), s)
    assert r < 0.5
    assert np.all((w >= 0.0) & (w <= 1.0))


def test_coherent_beats_incoherent():
    """서수 비교: 결맞음 점수는 전역 신호 > 무작위 위상 (P-adapt 검증)."""
    s, n = 64, 30
    rng = np.random.default_rng(3)
    xx = np.arange(s, dtype=np.float64)[None, :]
    coh = np.stack([128.0 + 10.0 * np.cos(2.0 * np.pi * xx / 8.0)
                    * np.ones((s, 1)) for _ in range(n)])
    inc = np.stack([128.0 + 10.0 * np.cos(2.0 * np.pi * xx / 8.0
                                          + rng.uniform(0.0, 2.0 * np.pi))
                    * np.ones((s, 1)) for _ in range(n)])
    offs = np.zeros((n, 2), dtype=np.float64)
    f = np.array([0.0, 1.0 / 8.0])
    r_coh, _ = phase_coherence(np.fft.fft2(coh), offs, f, s)
    r_inc, _ = phase_coherence(np.fft.fft2(inc), offs, f, s)
    assert r_coh > r_inc


def test_zero_magnitude_bin_returns_zero():
    s, n = 32, 5
    pf = np.zeros((n, s, s), dtype=np.complex128)
    offs = np.zeros((n, 2), dtype=np.float64)
    r, w = phase_coherence(pf, offs, np.array([0.0, 1.0 / 8.0]), s)
    assert r == 0.0
    assert np.all(w == 0.0)
