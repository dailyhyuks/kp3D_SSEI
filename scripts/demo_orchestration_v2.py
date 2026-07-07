"""Stage 2 오케스트레이션 데모: 실제 작품 + labelme 어노테이션 → 객체별 RGBA.

실행: python scripts/demo_orchestration_v2.py [stem] [--sam]
stem 기본 1_0004 (data_original_painting/target_data/<stem>.png + .json).
--sam: ~/.cache/sam/sam_vit_h.pth 가 있으면 SAM 정련 훅을 주입 (없으면 경고 후 생략).
산출: outputs/orchestration_v2/<stem>/ 아래 restored.png + <label>_rgba.png.
"""
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kp3d.modules.orchestration import orchestrate  # noqa: E402

DATA_DIR = Path("data_original_painting/target_data")


def _try_build_refiner():
    """SAM 체크포인트가 있을 때만 v1 정련기를 구성한다 (guarded import)."""
    ckpt = Path.home() / ".cache" / "sam" / "sam_vit_h.pth"
    if not ckpt.exists():
        print(f"경고: SAM 체크포인트 없음({ckpt}) — 정련 생략")
        return None
    try:
        from segment_anything import SamPredictor, sam_model_registry
        from kp3d.modules.occlusion.sam_mask_refiner import SAMMaskRefiner
    except ImportError as exc:
        print(f"경고: SAM 의존성 없음({exc}) — 정련 생략")
        return None
    sam = sam_model_registry["vit_h"](checkpoint=str(ckpt))
    return SAMMaskRefiner(sam_predictor=SamPredictor(sam))


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--sam"]
    stem = args[0] if args else "1_0004"
    use_sam = "--sam" in sys.argv[1:]

    img = cv2.imread(str(DATA_DIR / f"{stem}.png"))
    if img is None:
        raise SystemExit(f"이미지를 읽을 수 없음: {DATA_DIR / f'{stem}.png'}")
    with open(DATA_DIR / f"{stem}.json", encoding="utf-8") as f:
        shapes = json.load(f)["shapes"]

    refiner = _try_build_refiner() if use_sam else None
    res = orchestrate(img, shapes, refiner=refiner)

    out_dir = Path("outputs/orchestration_v2") / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "restored.png"), res.restored)
    for label, comp in res.completions.items():
        cv2.imwrite(str(out_dir / f"{label}_rgba.png"), comp.rgba)

    print(f"{stem}: 게이트 승자 {res.winner}, 객체 {len(res.annotations)}개, "
          f"간선 {len(res.edges)}건")
    for e in res.edges:
        print(f"  {e.occluder} → {e.occludee} (면적 {int(e.region.sum())}px)")
    for label, comp in res.completions.items():
        r = comp.result
        print(f"  {label}: 연결 {len(r.line.connections)}건, "
              f"G2 {r.g2_tangent_max:.2e}/{r.g2_curvature_max:.2e}, "
              f"위반 {r.by_construction_violations}건")
    for label, msg in res.failures.items():
        print(f"  {label}: 실패 — {msg}")
    print(f"저장: {out_dir}")


if __name__ == "__main__":
    main()
