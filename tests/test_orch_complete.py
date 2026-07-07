"""complete.py 테스트: RGBA 산출 규약, 입력 불변, 기계 검증."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from kp3d.modules.decomposition import decompose
from kp3d.modules.orchestration.complete import complete_object


def _scene(h=96, w=96):
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    img[30:70, :, :] = (80, 140, 60)                    # 객체 채색 띠
    cv2.line(img, (5, 50), (90, 50), (25, 25, 25), 3)   # 객체 내부 먹선
    return img


def test_complete_object_rgba_and_no_mutation():
    img = _scene()
    dec = decompose(img)
    occ = np.zeros(img.shape[:2], dtype=bool)
    occ[38:62, 40:56] = True
    band = np.zeros_like(occ)
    band[30:70, :] = True
    visible = band & ~occ
    la0 = dec.line_alpha.copy()
    sk0 = dec.skeleton.copy()
    wm0 = dec.width_map.copy()

    comp = complete_object("object_1", img, dec, visible, occ)

    assert comp.label == "object_1"
    assert (comp.amodal_mask == (visible | occ)).all()
    assert (comp.rgba[..., 3][comp.amodal_mask] == 255).all()
    assert (comp.rgba[..., 3][~comp.amodal_mask] == 0).all()
    assert (comp.rgba[..., :3][comp.amodal_mask]
            == comp.result.inpainted[comp.amodal_mask]).all()
    assert comp.result.by_construction_violations == 0
    # 입력 분해 산출은 불변 (여러 객체가 같은 dec 를 공유한다)
    assert (dec.line_alpha == la0).all()
    assert (dec.skeleton == sk0).all()
    assert (dec.width_map == wm0).all()
