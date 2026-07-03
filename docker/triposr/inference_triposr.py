#!/usr/bin/env python
"""TripoSR inference script for Docker container.

Usage:
    python inference_triposr.py --input /input/image.png --output /output/mesh.obj
"""

import argparse
import sys
from pathlib import Path

# NumPy 2.0 compatibility shim: trimesh 4.x calls ndarray.ptp() which was
# removed in NumPy 2.0. Restore it as a thin wrapper around np.ptp.
import numpy as _np
if not hasattr(_np.ndarray, 'ptp'):
    _np.ndarray.ptp = lambda self, *a, **kw: _np.ptp(self, *a, **kw)


def load_model(device="cuda"):
    """Load TripoSR model."""
    import torch
    print("Loading TripoSR model...")

    from tsr.system import TSR

    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt"
    )
    model.to(device)
    model.renderer.set_chunk_size(8192)

    print(f"TripoSR loaded on {device}")
    return model


def remove_background(image_path: str):
    """Remove background using rembg."""
    from PIL import Image
    from rembg import remove

    img = Image.open(image_path)
    return remove(img)


def _strip_background_shell(mesh, light_thr: int = 175, gray_tol: int = 25,
                             keep_largest: bool = True):
    """Remove background-shell vertices and keep largest component.

    TripoSR tends to extrude the input's white/gray background into a thin shell
    surrounding the actual object. We detect these vertices as:
      - bright (luma >= light_thr) AND
      - desaturated (channel spread <= gray_tol)
    Then we drop any faces touching them and keep the largest component.

    Manually constructs new Trimesh to avoid trimesh.submesh()'s use of
    ndarray.ptp() which was removed in NumPy 2.0.
    """
    import numpy as np
    import trimesh

    if not hasattr(mesh, 'visual') or not hasattr(mesh.visual, 'vertex_colors'):
        print("[strip] no vertex_colors, skipping shell strip")
        return mesh
    vc = np.asarray(mesh.visual.vertex_colors)
    if vc.ndim != 2 or vc.shape[1] < 3:
        print(f"[strip] unexpected vertex_colors shape {vc.shape}, skipping")
        return mesh
    rgb = vc[:, :3].astype(np.int32)

    # Diagnostics: color distribution
    luma = rgb.mean(axis=1)
    spread = rgb.max(axis=1) - rgb.min(axis=1)
    pct = np.percentile(luma, [10, 50, 75, 90, 95, 99])
    print(f"[strip] luma pct (10/50/75/90/95/99): "
          f"{pct[0]:.0f}/{pct[1]:.0f}/{pct[2]:.0f}/{pct[3]:.0f}/{pct[4]:.0f}/{pct[5]:.0f}; "
          f"mean spread={spread.mean():.1f}")

    is_shell = (luma >= light_thr) & (spread <= gray_tol)
    print(f"[strip] shell candidates (luma>={light_thr}, spread<={gray_tol}): "
          f"{is_shell.sum()}/{len(rgb)} ({100.0*is_shell.sum()/len(rgb):.1f}%)")

    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)
    VC = np.asarray(mesh.visual.vertex_colors)

    if is_shell.any():
        keep_v = ~is_shell
        face_mask = keep_v[F].all(axis=1)
        if not face_mask.any():
            print("[strip] all faces touch shell verts, giving up")
            return mesh
        # Manually drop faces, then construct new mesh (avoids submesh's ptp bug)
        F = F[face_mask]

    # Pick largest connected component using scipy (avoids trimesh.split's
    # use of ndarray.ptp() which was removed in NumPy 2.0)
    if keep_largest and len(F) > 0:
        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import connected_components
            n_v = len(V)
            e0 = F[:, [0, 1]]
            e1 = F[:, [1, 2]]
            e2 = F[:, [2, 0]]
            edges = np.vstack([e0, e1, e2])
            rows = np.concatenate([edges[:, 0], edges[:, 1]])
            cols = np.concatenate([edges[:, 1], edges[:, 0]])
            data = np.ones(len(rows), dtype=np.int8)
            adj = csr_matrix((data, (rows, cols)), shape=(n_v, n_v))
            n_comp, labels = connected_components(adj, directed=False)
            if n_comp > 1:
                counts = np.bincount(labels)
                largest = int(np.argmax(counts))
                keep_v = labels == largest
                face_mask = keep_v[F].all(axis=1)
                F = F[face_mask]
                print(f"[strip] kept largest of {n_comp} comps "
                      f"({counts[largest]}/{n_v} verts)")
        except Exception as e:
            print(f"[strip] CC split failed: {e}")

    # Rebuild mesh; remove unreferenced vertices to compact
    new_mesh = trimesh.Trimesh(vertices=V, faces=F, vertex_colors=VC,
                                process=False)
    try:
        new_mesh.remove_unreferenced_vertices()
    except Exception as e:
        print(f"[strip] remove_unreferenced failed: {e}")

    print(f"[strip] final V={len(new_mesh.vertices)}, F={len(new_mesh.faces)}")
    return new_mesh


def process_image(model, input_path: str, output_path: str, device="cuda"):
    """Process single image."""
    import torch
    import trimesh
    from PIL import Image

    print(f"Processing: {input_path}")

    # Load input. If RGBA with non-trivial alpha mask, skip rembg (the upstream
    # pipeline already produced a clean per-object mask). Otherwise run rembg.
    src = Image.open(input_path)
    if src.mode == 'RGBA':
        alpha = src.split()[3]
        a_arr = alpha.getextrema()  # (min, max)
        if a_arr[0] < a_arr[1]:
            print("Input is RGBA with mask, skipping rembg.")
            image = src
        else:
            image = remove_background(input_path)
    else:
        image = remove_background(input_path)

    image = image.resize((512, 512), Image.Resampling.LANCZOS)

    # Ensure RGB (TripoSR expects 3-channel input). Composite onto white using
    # alpha as mask to avoid feeding any non-mask color into TripoSR.
    if image.mode == 'RGBA':
        bg = Image.new('RGB', image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        image = bg
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    # Run TripoSR (higher threshold suppresses background halo)
    print("Running TripoSR...")
    with torch.no_grad():
        scene_codes = model([image], device=device)
        meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=256, threshold=35.0)

    mesh = meshes[0]

    # Post-process: drop near-white "background shell" vertices and keep only
    # the largest connected component of the remaining foreground geometry.
    import numpy as np
    mesh = _strip_background_shell(mesh)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path)
    print(f"Saved: {output_path}")

    # Save preview renders
    output_dir = Path(output_path).parent
    save_multiview_renders(mesh, output_dir, Path(output_path).stem)

    return output_path


def save_multiview_renders(mesh, output_dir, name, num_views=6):
    """Save multi-view renders of mesh."""
    import numpy as np
    import cv2

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)

        # Normalize
        center = vertices.mean(axis=0)
        vertices = vertices - center
        scale = np.abs(vertices).max()
        if scale > 0:
            vertices = vertices / scale

        # Get colors
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
            colors = np.array(mesh.visual.vertex_colors)[:, :3] / 255.0
        else:
            colors = np.ones((len(vertices), 3)) * 0.7

        fig = plt.figure(figsize=(18, 3))

        for i in range(num_views):
            ax = fig.add_subplot(1, num_views, i + 1, projection='3d')
            angle = i * (360 / num_views)

            poly3d = [[vertices[f[j]] for j in range(3)] for f in faces]
            face_colors = [colors[f].mean(axis=0) for f in faces]

            collection = Poly3DCollection(poly3d, alpha=0.9)
            collection.set_facecolor(face_colors)
            collection.set_edgecolor('none')
            ax.add_collection3d(collection)

            ax.view_init(elev=20, azim=angle)
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.set_axis_off()

        plt.tight_layout()
        plt.savefig(output_dir / f"{name}_multiview.png", dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved multiview: {output_dir / f'{name}_multiview.png'}")

    except Exception as e:
        print(f"Multiview render failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="TripoSR 3D reconstruction")
    parser.add_argument("--input", "-i", type=str, help="Input image")
    parser.add_argument("--output", "-o", type=str, help="Output mesh (.obj)")
    parser.add_argument("--input-dir", type=str, help="Batch input directory")
    parser.add_argument("--output-dir", type=str, help="Batch output directory")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    if not args.input and not args.input_dir:
        parser.print_help()
        sys.exit(1)

    model = load_model(args.device)

    if args.input:
        output = args.output or args.input.replace(".png", ".obj")
        process_image(model, args.input, output, args.device)

    elif args.input_dir:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir or args.input_dir)

        for img in input_dir.glob("*.png"):
            try:
                process_image(model, str(img), str(output_dir / f"{img.stem}.obj"), args.device)
            except Exception as e:
                print(f"Failed {img}: {e}")


if __name__ == "__main__":
    main()
