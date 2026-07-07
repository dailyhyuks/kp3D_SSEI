"""SSEI 2.0 실험: labelme 어노테이션 기반 가림 제거 복원.

실행: python scripts/exp_ssei_annotation.py <stem> [occluder_label]
예: python scripts/exp_ssei_annotation.py 1_0004 object_1

data_original_painting/target_data/<stem>.png + <stem>.json 을 읽어
occluder_label(기본 object_1) 폴리곤을 가림 마스크로 삼는다:
Stage 0 분해 → 가림 내부 지식 소거 → inpaint → outputs/ssei_v2_real/<stem>/ 저장.
실제 가림(GT 없음) — 정성 평가 산출물.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kp3d.modules.decomposition import decompose  # noqa: E402
from kp3d.modules.ssei_v2 import inpaint  # noqa: E402

DATA_DIR = Path("data_original_painting/target_data")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("사용법: python scripts/exp_ssei_annotation.py <stem> [occluder_label]")
    stem = sys.argv[1]
    occluder = sys.argv[2] if len(sys.argv) > 2 else "object_1"

    img = cv2.imread(str(DATA_DIR / f"{stem}.png"))
    if img is None:
        raise SystemExit(f"이미지를 읽을 수 없음: {DATA_DIR / f'{stem}.png'}")
    with open(DATA_DIR / f"{stem}.json", encoding="utf-8") as f:
        anno = json.load(f)
    h, w = img.shape[:2]

    occ8 = np.zeros((h, w), dtype=np.uint8)
    n_poly = 0
    for s in anno["shapes"]:
        if s["label"] == occluder:
            pts = np.asarray(s["points"], dtype=np.float64).round().astype(np.int32)
            cv2.fillPoly(occ8, [pts], 1)
            n_poly += 1
    if n_poly == 0:
        labels = sorted({s["label"] for s in anno["shapes"]})
        raise SystemExit(f"라벨 '{occluder}' 없음. 가능한 라벨: {labels}")
    occ = occ8.astype(bool)

    dec = decompose(img)
    la = dec.line_alpha.copy(); la[occ] = 0.0
    sk = dec.skeleton.copy(); sk[occ] = False
    wm = dec.width_map.copy(); wm[occ] = 0.0
    col = dec.color_layer.copy(); col[occ] = (255, 0, 255)
    img_in = img.copy(); img_in[occ] = (255, 0, 255)

    res = inpaint(img_in, col, la, sk, wm, occ, dec.noise_sigma)

    out_dir = Path("outputs/ssei_v2_real") / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "original.png"), img)
    cv2.imwrite(str(out_dir / "input_occluded.png"), img_in)
    cv2.imwrite(str(out_dir / "occlusion_mask.png"), occ8 * 255)
    cv2.imwrite(str(out_dir / "line_alpha.png"),
                np.rint(res.line.line_alpha * 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / "color_filled.png"), res.color.filled)
    cv2.imwrite(str(out_dir / "inpainted.png"), res.inpainted)

    frac = float(occ.mean()) * 100.0
    print(f"{stem}: 가림({occluder}) 폴리곤 {n_poly}개, 면적 {frac:.1f}%")
    print(f"연결 {len(res.line.connections)}건, 종결 {len(res.line.terminations)}건, "
          f"조각 {len(res.color.pieces)}개, 패치 {res.color.patch_size}px, "
          f"레벨 {res.color.levels}")
    print(f"G2 접선 불연속 최대 {res.g2_tangent_max:.2e} rad, "
          f"곡률 불연속 최대 {res.g2_curvature_max:.2e} 1/px")
    print(f"by-construction 위반 {res.by_construction_violations}건")
    print(f"저장: {out_dir}")


if __name__ == "__main__":
    main()
