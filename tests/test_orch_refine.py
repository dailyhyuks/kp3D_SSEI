"""refine.py 테스트: 주입 정련기 호출 규약(set_image 1회), 라벨·depth 보존."""
import numpy as np

from kp3d.modules.orchestration.annotations import ObjectAnnotation
from kp3d.modules.orchestration.refine import refine_annotations


class _StubRefiner:
    """호출 기록 스텁 — SAM 없이 계약만 검증한다."""

    def __init__(self):
        self.calls = []

    def refine_mask(self, image, rough_mask, label, *, set_image):
        self.calls.append((label, bool(set_image)))
        out = np.asarray(rough_mask, dtype=bool).copy()
        out[0, 0] = True  # 정련이 마스크를 실제로 바꾸는지 관측용
        return out


def test_refine_annotations_contract():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    m1 = np.zeros((20, 20), dtype=bool); m1[2:8, 2:8] = True
    m2 = np.zeros((20, 20), dtype=bool); m2[10:18, 10:18] = True
    annos = [ObjectAnnotation(label="object_2", mask=m1, depth=0),
             ObjectAnnotation(label="object_1", mask=m2, depth=2)]
    ref = _StubRefiner()

    out = refine_annotations(img, annos, ref)

    # 첫 호출만 set_image=True — SAM 이미지 임베딩 1회 재사용 계약
    assert ref.calls == [("object_2", True), ("object_1", False)]
    assert [a.label for a in out] == ["object_2", "object_1"]
    assert [a.depth for a in out] == [0, 2]
    assert out[0].mask[0, 0] and out[1].mask[0, 0]  # 정련 결과가 반영됐다
    assert not annos[0].mask[0, 0]  # 입력 어노테이션은 불변
