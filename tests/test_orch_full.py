"""오케스트레이션 통합 — 2객체 합성 장면: 그래프 방향, 아모달 완성 정량(프로토콜 B 동형).

fixture 는 decompose 의 도메인(얇은 어두운 먹 획 + 등휘도 채색 wash + 미세 직조)을
따라야 한다. 수치 검증으로 확정된 제약 3가지:
① 직조(주기 4px) + 약한 노이즈 필수 — 없으면 weave 주기가 구조 크기(60px)로
   오검출되어 RGF sigma_s 가 커지고 3px 먹선이 blob 으로 뭉개진다.
② wash 들은 grayscale 등휘도(채널 평균 동일)여야 한다 — 아니면 wash 경계가
   선으로 분류된다.
③ 가려지는 먹 획의 가시 stub 은 국소 선폭(smear 후 ~20px)보다 길어야 한다 —
   짧으면 stub 양끝이 모두 endpoint 로 검출돼(E=4) 매칭이 퇴화 동률로 전원
   종결된다. 또한 획에 모서리가 닿으면 medial-axis 대각 가지가 끝점 접선을
   오염시키므로 모서리 없는 일자 획을 쓴다.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from scipy.ndimage import binary_dilation

from kp3d.modules.decomposition import decompose
from kp3d.modules.orchestration import orchestrate


def _scene(h=160, w=160, with_disk=True, seed=0):
    """등휘도 wash 배경/사각 + 수평 먹 획(y=90) + 전경 원판(먹 윤곽) + 직조/노이즈."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (60, 170, 60)                                    # 배경 wash
    cv2.rectangle(img, (10, 60), (150, 120), (60, 60, 170), -1)  # 후방 사각 wash
    cv2.line(img, (15, 90), (145, 90), (20, 20, 20), 3)          # 사각 내부 수평 먹 획
    if with_disk:
        cv2.circle(img, (80, 90), 24, (170, 60, 60), -1)         # 전경 원판 wash
        cv2.circle(img, (80, 90), 24, (20, 20, 20), 3)           # 원판 먹 윤곽
    out = img.astype(np.float64)
    yy, xx = np.mgrid[0:h, 0:w]
    weave = 6.0 * (np.sin(2 * np.pi * xx / 4.0) + np.sin(2 * np.pi * yy / 4.0))
    noise = np.random.default_rng(seed).normal(0.0, 2.0, (h, w))
    out += (weave + noise)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _shapes():
    ang = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    circle = [[80.0 + 27.0 * np.cos(a), 90.0 + 27.0 * np.sin(a)] for a in ang]
    return [
        {"label": "object_2_1", "points": circle,
         "shape_type": "polygon", "layer_order": 1},
        {"label": "object_1",
         "points": [[10, 60], [150, 60], [150, 120], [10, 120]],
         "shape_type": "polygon", "layer_order": 2},
    ]


def _psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0.0 else 10.0 * np.log10(255.0 ** 2 / mse)


@pytest.mark.timeout(600)
def test_orchestrate_graph_direction_and_amodal_quality():
    img = _scene()
    gt = _scene(with_disk=False)  # 원판이 없는 장면 = object_1 의 GT

    res = orchestrate(img, _shapes(), restore=False)

    assert res.winner == "none"
    assert res.failures == {}
    # 그래프: 전경 원판(object_2)이 후방 사각(object_1)을 가린다 — 이 간선뿐
    assert [(e.occluder, e.occludee) for e in res.edges] == [("object_2", "object_1")]
    assert "object_2" not in res.completions  # 최전방 객체는 가려지지 않는다
    assert set(res.completions) == {"object_1"}

    comp = res.completions["object_1"]
    occ = res.edges[0].region
    # annotations 는 depth 오름차순 — [object_2, object_1]
    vis_rect = res.visibles[1]

    # ① PSNR(가림 후보 내부): 가시 평균색 채움 베이스라인 초과
    mean_col = np.rint(img[vis_rect].reshape(-1, 3).mean(axis=0)).astype(np.uint8)
    base = gt.copy(); base[occ] = mean_col
    assert _psnr(comp.result.inpainted[occ], gt[occ]) > _psnr(base[occ], gt[occ])

    # ② 스켈레톤 재현율(1px 팽창 허용): 수평 먹 획 자취 회복
    gt_sk = decompose(gt).skeleton & occ
    assert np.any(gt_sk)  # 획 중앙이 원판에 가려져 있어야 실험이 성립
    cover = binary_dilation(comp.result.line.skeleton, iterations=1)
    recall = (float(np.count_nonzero(gt_sk & cover))
              / float(np.count_nonzero(gt_sk)))
    assert recall > 0.7

    # ③ 기계 검증 지표
    assert comp.result.by_construction_violations == 0
    assert comp.result.g2_tangent_max < 1e-6
    assert comp.result.g2_curvature_max < 1e-6
