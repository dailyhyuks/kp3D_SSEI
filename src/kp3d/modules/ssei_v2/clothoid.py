"""G2 연결 곡선 — biclothoid(주 해) + quintic Hermite(안전망) (스펙 §3.2 ③).

기하 관례: 좌표 (y,x), 접선 t=(dy,dx), θ=atan2(dy,dx), 좌법선 N=(t_x,−t_y),
r''=κN. connect_g2의 접선·곡률은 진행 방향(p0→p1) 기준이다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares

# 위치 허용 오차 반 픽셀 — 이산 격자 표본화 한계 (수학 유도)
_HALF_PIXEL = 0.5
# 곡선 표본 밀도 2 샘플/px — 나이퀴스트 (수학 유도)
_SAMPLES_PER_PX = 2.0
# 이산 격자 최소 유의 현길이 1px — 이산 하한 (수학 유도)
_MIN_CHORD = 1.0


@dataclass
class ConnectionCurve:
    """G2 연결 곡선 표본열."""

    points: np.ndarray      # (N,2) float64 (y,x)
    tangents: np.ndarray    # (N,2) float64 단위 접선 (진행 방향)
    curvatures: np.ndarray  # (N,) float64 부호 곡률 [1/px]
    arc_length: float       # 총 호장 [px]
    bending_energy: float   # ∫ κ² ds
    is_clothoid: bool       # biclothoid 수렴 여부 (False = quintic 안전망)


def _left_normal(t: np.ndarray) -> np.ndarray:
    """좌법선 N=(t_x, −t_y) — 모듈 기하 관례."""
    return np.array([t[1], -t[0]], dtype=np.float64)


def _wrap_angle(a: float) -> float:
    """각도를 (−π, π]로 정규화 — 순수 수학."""
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _n_samples(length: float) -> int:
    """호장 length의 나이퀴스트 표본 수 — 최소 4 (3차 형상 식별 하한, 수학 유도)."""
    return max(int(np.ceil(length * _SAMPLES_PER_PX)) + 1, 4)


def _energy(kappa: np.ndarray, s: np.ndarray) -> float:
    """굽힘 에너지 ∫ κ² ds (사다리꼴 적분)."""
    return float(cumulative_trapezoid(kappa ** 2, s, initial=0.0)[-1])


def _quintic(p0, t0, k0, p1, t1, k1) -> ConnectionCurve:
    """quintic Hermite G2 폐형 — v=d(현길이) 매개, 가속 a=0, r''(u)=d²κN."""
    d = float(np.linalg.norm(p1 - p0))
    n = _n_samples(d)
    u = np.linspace(0.0, 1.0, n)
    u2, u3, u4, u5 = u ** 2, u ** 3, u ** 4, u ** 5
    # quintic Hermite 기저 (G2 보간의 폐형 — 순수 수학)
    H = [1 - 10 * u3 + 15 * u4 - 6 * u5,
         u - 6 * u3 + 8 * u4 - 3 * u5,
         0.5 * u2 - 1.5 * u3 + 1.5 * u4 - 0.5 * u5,
         0.5 * u3 - u4 + 0.5 * u5,
         -4 * u3 + 7 * u4 - 3 * u5,
         10 * u3 - 15 * u4 + 6 * u5]
    dH = [-30 * u2 + 60 * u3 - 30 * u4,
          1 - 18 * u2 + 32 * u3 - 15 * u4,
          u - 4.5 * u2 + 6 * u3 - 2.5 * u4,
          1.5 * u2 - 4 * u3 + 2.5 * u4,
          -12 * u2 + 28 * u3 - 15 * u4,
          30 * u2 - 60 * u3 + 30 * u4]
    ddH = [-60 * u + 180 * u2 - 120 * u3,
           -36 * u + 96 * u2 - 60 * u3,
           1 - 9 * u + 18 * u2 - 10 * u3,
           3 * u - 12 * u2 + 10 * u3,
           -24 * u + 84 * u2 - 60 * u3,
           60 * u - 180 * u2 + 120 * u3]
    ctrl = [p0, d * t0, d * d * k0 * _left_normal(t0),
            d * d * k1 * _left_normal(t1), d * t1, p1]
    pts = sum(np.outer(h, c) for h, c in zip(H, ctrl))
    d1 = sum(np.outer(h, c) for h, c in zip(dH, ctrl))
    d2 = sum(np.outer(h, c) for h, c in zip(ddH, ctrl))
    speed = np.hypot(d1[:, 0], d1[:, 1])
    speed = np.where(speed > 0.0, speed, 1.0)
    tangents = d1 / speed[:, None]
    # κ = (dx·ddy − dy·ddx)/|r'|³ — N=(t_x,−t_y) 관례 (Task 1과 동일)
    kappa = (d1[:, 1] * d2[:, 0] - d1[:, 0] * d2[:, 1]) / speed ** 3
    s = cumulative_trapezoid(speed, u, initial=0.0)
    return ConnectionCurve(points=pts, tangents=tangents, curvatures=kappa,
                           arc_length=float(s[-1]),
                           bending_energy=_energy(kappa, s), is_clothoid=False)


def _biclothoid_geometry(theta0: float, k0: float, km: float, k1: float,
                         L1: float, L2: float, p0: np.ndarray):
    """구간 선형 κ 곡선(biclothoid)을 수치 적분 → (pts, tangents, kappa, s, theta)."""
    L = L1 + L2
    n = _n_samples(L)
    s = np.linspace(0.0, L, n)
    kappa = np.where(s <= L1,
                     k0 + (km - k0) * s / L1,
                     km + (k1 - km) * (s - L1) / L2)
    theta = theta0 + cumulative_trapezoid(kappa, s, initial=0.0)
    dy, dx = np.sin(theta), np.cos(theta)
    y = p0[0] + cumulative_trapezoid(dy, s, initial=0.0)
    x = p0[1] + cumulative_trapezoid(dx, s, initial=0.0)
    pts = np.stack([y, x], axis=1)
    tangents = np.stack([dy, dx], axis=1)
    return pts, tangents, kappa, s, theta


def connect_g2(p0, t0, k0, p1, t1, k1) -> ConnectionCurve:
    """G2 경계 조건(위치·접선·곡률)을 만족하는 연결 곡선.

    biclothoid(미지수: log L1, log L2, κm)를 least_squares로 풀고,
    잔차가 반 픽셀 허용 오차를 넘으면 quintic Hermite 폐형으로 강등한다.

    Returns:
        ConnectionCurve — is_clothoid=True면 biclothoid, False면 안전망.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    t0 = np.asarray(t0, dtype=np.float64) / float(np.linalg.norm(t0))
    t1 = np.asarray(t1, dtype=np.float64) / float(np.linalg.norm(t1))
    k0, k1 = float(k0), float(k1)
    d = float(np.linalg.norm(p1 - p0))
    if d < _MIN_CHORD:
        # 표본화 한계 미만 간격 — 2점 직선 퇴화 (이산 하한)
        return ConnectionCurve(points=np.stack([p0, p1]),
                               tangents=np.tile(t0, (2, 1)),
                               curvatures=np.zeros(2), arc_length=d,
                               bending_energy=0.0, is_clothoid=False)
    quintic = _quintic(p0, t0, k0, p1, t1, k1)
    theta0 = float(np.arctan2(t0[0], t0[1]))
    theta1 = float(np.arctan2(t1[0], t1[1]))
    L_q = max(quintic.arc_length, d)
    k_mid = float(quintic.curvatures[len(quintic.curvatures) // 2])

    def residual(u: np.ndarray) -> np.ndarray:
        L1, L2 = float(np.exp(u[0])), float(np.exp(u[1]))
        pts, _, _, _, theta = _biclothoid_geometry(
            theta0, k0, float(u[2]), k1, L1, L2, p0)
        # 각도 잔차에 호장을 곱해 위치 잔차와 단위(px)를 일치 (순수 수학)
        return np.array([pts[-1, 0] - p1[0], pts[-1, 1] - p1[1],
                         _wrap_angle(float(theta[-1]) - theta1) * (L1 + L2)])

    # 초기해: quintic의 호장 절반씩 + 중간 곡률 (해석해에서 유도)
    u0 = np.array([np.log(L_q / 2.0), np.log(L_q / 2.0), k_mid])
    try:
        sol = least_squares(residual, u0)
    except Exception:
        return quintic
    L1, L2 = float(np.exp(sol.x[0])), float(np.exp(sol.x[1]))
    km = float(sol.x[2])
    pts, tangents, kappa, s, theta = _biclothoid_geometry(
        theta0, k0, km, k1, L1, L2, p0)
    pos_err = float(np.linalg.norm(pts[-1] - p1))
    ang_err = abs(_wrap_angle(float(theta[-1]) - theta1))
    # 수락: 위치 ≤ 반 픽셀, 각도 ≤ 반 픽셀/호장 — 표본화 한계 (수학 유도)
    if pos_err > _HALF_PIXEL or ang_err > _HALF_PIXEL / (L1 + L2):
        return quintic
    return ConnectionCurve(points=pts, tangents=tangents, curvatures=kappa,
                           arc_length=float(s[-1]),
                           bending_energy=_energy(kappa, s), is_clothoid=True)
