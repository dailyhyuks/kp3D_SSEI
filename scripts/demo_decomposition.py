"""Stage 0 분해 결과 시각화 데모.

사용법: python scripts/demo_decomposition.py <이미지 경로> [출력 디렉터리]
출력: line_layer.png (L: 흰 배경 위 선), color_layer.png (C),
      alpha.png, recomposed.png
"""
import os
import sys

import cv2
import numpy as np

from kp3d.modules.decomposition import decompose, recompose_result


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    image_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        "output", "decomposition_demo"
    )
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        print(f"이미지를 읽을 수 없음: {image_path}")
        sys.exit(1)

    result = decompose(img)
    rec = recompose_result(img, result)

    alpha = result.line_alpha[..., None]
    white = np.full_like(img, 255)
    line_vis = (alpha * img + (1 - alpha) * white).astype(np.uint8)

    cv2.imwrite(os.path.join(out_dir, "line_layer.png"), line_vis)
    cv2.imwrite(os.path.join(out_dir, "color_layer.png"), result.color_layer)
    cv2.imwrite(
        os.path.join(out_dir, "alpha.png"),
        (result.line_alpha * 255).astype(np.uint8),
    )
    cv2.imwrite(os.path.join(out_dir, "recomposed.png"), rec)

    residual = np.abs(
        rec.astype(np.float64) - img.astype(np.float64)
    ).mean()
    print(f"직조 주기: x={result.weave.period_x:.1f}px "
          f"(강도 {result.weave.strength_x:.2f}), "
          f"y={result.weave.period_y:.1f}px "
          f"(강도 {result.weave.strength_y:.2f})")
    print(f"노이즈 σ_n: {result.noise_sigma:.2f}")
    print(f"평균 재합성 잔차: {residual:.3f}")
    print(f"출력: {out_dir}")


if __name__ == "__main__":
    main()
