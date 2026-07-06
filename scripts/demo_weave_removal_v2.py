"""Stage 1 v2 데모: 게이트 실행 결과와 중간 산출물을 저장.

사용법: python scripts/demo_weave_removal_v2.py [이미지 경로]
기본 이미지: data/ablation_study/images/1_0004.png
출력: output/weave_removal_v2_demo/
"""
import sys
from pathlib import Path

import cv2
import numpy as np

from kp3d.modules.weave_removal_v2 import (
    estimate_lattice,
    restore,
    self_competition_gate,
    weave_band_energy,
)

_DEFAULT = "data/ablation_study/images/1_0004.png"


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else _DEFAULT)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"이미지를 읽을 수 없습니다: {path}")
        return 1
    out_dir = Path("output/weave_removal_v2_demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lattice = estimate_lattice(gray)
    e0 = weave_band_energy(gray, lattice)

    r = restore(img)
    gate = self_competition_gate(img)
    gray_gate = cv2.cvtColor(gate.restored, cv2.COLOR_BGR2GRAY).astype(np.float32)
    e1 = weave_band_energy(gray_gate, lattice)

    cv2.imwrite(str(out_dir / "input.png"), img)
    cv2.imwrite(str(out_dir / "v2_restored.png"), r.restored)
    cv2.imwrite(str(out_dir / "v2_color_cleaned.png"), r.color_cleaned)
    cv2.imwrite(str(out_dir / "v2_line_alpha.png"),
                (r.line_alpha * 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / "gate_restored.png"), gate.restored)

    print(f"격자 기저 수 K={lattice.basis.shape[0]}, strength={lattice.strength:.3f}")
    print(f"직조 대역 에너지: {e0:.3f} -> {e1:.3f}")
    print(f"v2 반복 횟수: {r.weave.iterations}, sigma_n={r.noise_sigma:.3f}")
    print(f"게이트 승자: {gate.winner} (Q_v2={gate.quality_v2:.4f}, "
          f"Q_v1={gate.quality_v1:.4f})")
    print(f"저장 위치: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
