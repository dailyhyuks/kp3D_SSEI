"""SSEI 2.0 데모: 작품 → Stage 0 분해 → 지식 소거 → inpaint → 산출 저장.

실행: python scripts/demo_ssei_v2.py [입력 이미지 경로]
입력 생략 시 통합 테스트와 같은 구성의 합성 작품(256×256)을 사용한다.
산출: outputs/ssei_v2_demo/ 아래 PNG.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kp3d.modules.decomposition import decompose  # noqa: E402
from kp3d.modules.ssei_v2 import inpaint  # noqa: E402


def synthetic_artwork(h: int = 256, w: int = 256) -> np.ndarray:
    """2색 배경 + 사인 곡선 먹선 합성 작품."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :w // 2] = (60, 170, 60)
    img[:, w // 2:] = (60, 60, 170)
    xs = np.arange(10, w - 10)
    ys = h / 2.0 + 0.1 * h * np.sin(2.0 * np.pi * (xs - 10) / (w - 20))
    for x, y in zip(xs, ys):
        cv2.circle(img, (int(x), int(round(y))), 2, (20, 20, 20), -1)
    return img


def main() -> None:
    out_dir = Path("outputs/ssei_v2_demo")
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is None:
            raise SystemExit(f"이미지를 읽을 수 없음: {sys.argv[1]}")
    else:
        img = synthetic_artwork()
    h, w = img.shape[:2]
    dec = decompose(img)
    occ = np.zeros((h, w), dtype=bool)
    occ[h * 2 // 5:h * 3 // 5, w * 2 // 5:w * 3 // 5] = True  # 중앙 상자 가림
    la = dec.line_alpha.copy(); la[occ] = 0.0
    sk = dec.skeleton.copy(); sk[occ] = False
    wm = dec.width_map.copy(); wm[occ] = 0.0
    col = dec.color_layer.copy(); col[occ] = (255, 0, 255)
    img_in = img.copy(); img_in[occ] = (255, 0, 255)

    res = inpaint(img_in, col, la, sk, wm, occ, dec.noise_sigma)

    cv2.imwrite(str(out_dir / "input_occluded.png"), img_in)
    cv2.imwrite(str(out_dir / "line_alpha.png"),
                np.rint(res.line.line_alpha * 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / "color_filled.png"), res.color.filled)
    cv2.imwrite(str(out_dir / "inpainted.png"), res.inpainted)
    print(f"연결 {len(res.line.connections)}건, 종결 {len(res.line.terminations)}건, "
          f"조각 {len(res.color.pieces)}개, 패치 {res.color.patch_size}px, "
          f"레벨 {res.color.levels}")
    print(f"G2 접선 불연속 최대 {res.g2_tangent_max:.2e} rad, "
          f"곡률 불연속 최대 {res.g2_curvature_max:.2e} 1/px")
    print(f"by-construction 위반 {res.by_construction_violations}건")
    print(f"저장: {out_dir}")


if __name__ == "__main__":
    main()
