#!/usr/bin/env python
"""Wonder3D++ inference wrapper for Docker container.

Drop-in replacement for the TripoSR inference script. Same CLI:
    python inference_wonder3d.py --input /input/img.png --output /output/mesh.obj

Internally runs Wonder3D++'s multi-stage pipeline:
  1) Cross-domain MV diffusion (color + normal, 6 views)
  2) Coarse mesh reconstruction
  3) MV enhancement (ControlNet) + iterative mesh refinement (default 2 iters)

Final mesh is exported as .obj with vertex colors.
"""
import argparse
import os
import sys
import shutil
import tempfile
import traceback
from pathlib import Path

# NumPy 2.0 compatibility shim for some downstream libraries.
import numpy as _np
if not hasattr(_np.ndarray, 'ptp'):
    try:
        _np.ndarray.ptp = lambda self, *a, **kw: _np.ptp(self, *a, **kw)
    except Exception:
        pass  # built-in, may not be patchable, downstream tries np.ptp directly


WONDER3D_ROOT = Path('/app/wonder3d')
DEFAULT_CKPT_DIR = WONDER3D_ROOT / 'ckpts'


def ensure_checkpoints():
    """Download Wonder3D_plus checkpoints into HF cache and symlink to ./ckpts.

    The upstream code hard-codes './ckpts' relative paths in configs, so we
    symlink the downloaded snapshot directory there.
    """
    from huggingface_hub import snapshot_download
    if DEFAULT_CKPT_DIR.exists():
        # Already present (either previous run mounted it or baked into image)
        print(f"Checkpoints present at {DEFAULT_CKPT_DIR}")
        return DEFAULT_CKPT_DIR
    print("Downloading Wonder3D_plus checkpoints (one-time, ~9.6 GB)...")
    snap_dir = snapshot_download(
        repo_id='flamehaze1115/Wonder3D_plus',
        cache_dir=os.environ.get('HF_HOME', '/cache/huggingface'),
    )
    # Symlink so relative './ckpts' resolves
    DEFAULT_CKPT_DIR.parent.mkdir(parents=True, exist_ok=True)
    if DEFAULT_CKPT_DIR.is_symlink() or DEFAULT_CKPT_DIR.exists():
        DEFAULT_CKPT_DIR.unlink()
    DEFAULT_CKPT_DIR.symlink_to(snap_dir, target_is_directory=True)
    print(f"Checkpoints linked: {DEFAULT_CKPT_DIR} -> {snap_dir}")
    return DEFAULT_CKPT_DIR


def prepare_input_image(input_path: str):
    """Open input. If already RGBA with non-trivial mask, keep alpha.
    Otherwise run rembg to produce one. Wonder3D++ expects the alpha channel
    as foreground mask.
    """
    from PIL import Image
    from rembg import remove
    src = Image.open(input_path)
    if src.mode == 'RGBA':
        alpha_min, alpha_max = src.split()[3].getextrema()
        if alpha_min < alpha_max:
            print("Input is RGBA with mask, skipping rembg.")
            return src
    print("Running rembg to extract foreground mask...")
    return remove(src.convert('RGB') if src.mode != 'RGB' else src)


def load_pipelines(device='cuda'):
    """Load Wonder3D++ MV diffusion + ControlNet enhancement pipelines."""
    import torch
    from omegaconf import OmegaConf

    # Wonder3D++ imports require cwd at /app/wonder3d for relative configs.
    os.chdir(str(WONDER3D_ROOT))
    sys.path.insert(0, str(WONDER3D_ROOT))
    sys.path.insert(0, str(WONDER3D_ROOT / 'MVMeshRecon'))
    sys.path.insert(0, str(WONDER3D_ROOT / 'MVMeshRecon' / 'utils'))

    from run_mv_prediction import load_wonder3d_pipeline
    from run_mv_enhancement import load_controlnet_pipeline

    config_mv = OmegaConf.load(str(WONDER3D_ROOT / 'configs' / 'mvdiffusion-joint.yaml'))
    config_cn = OmegaConf.load(str(WONDER3D_ROOT / 'configs' / 'controlnet.yaml'))

    dev = torch.device(device)
    print("Loading mv diffusion pipeline...")
    mv_pipe = load_wonder3d_pipeline(config_mv).to(dev)
    print("Loading mv enhancement pipeline...")
    en_pipe = load_controlnet_pipeline(config_cn).to(dev)
    print(f"Pipelines on {device}")
    return mv_pipe, en_pipe, config_mv, config_cn


def process_image_pipeline(mv_pipe, en_pipe, config_mv, config_cn,
                           input_image_path: str, output_obj_path: str,
                           camera_type: str = 'ortho', crop_size: int = 192,
                           num_refine: int = 2, seed: int = 42,
                           device: str = 'cuda'):
    """Run the full Wonder3D++ pipeline on one image, export .obj."""
    import torch
    import trimesh
    from PIL import Image

    from run_mv_prediction import pred_multiview_joint
    from run_mv_enhancement import pred_enhancement_joint
    from MVMeshRecon.Coarse_recon import coarse_recon
    from MVMeshRecon.Iterative_refine import iterative_refine
    from MVMeshRecon.utils.refine_lr_to_sr import sr_front

    # Wonder3D's run.py utility helpers (reimplemented inline for clarity)
    def add_margin(pil_img, color=0, size=256):
        w, h = pil_img.size
        result = Image.new(pil_img.mode, (size, size), color)
        result.paste(pil_img, ((size - w) // 2, (size - h) // 2))
        return result

    def views_6to4(imgs):
        return [imgs[i] for i in range(6) if i not in (1, 5)]

    def process_input(image_input, image_size=2048, crop_size_px=1536):
        import numpy as np
        if np.asarray(image_input).shape[-1] == 3:
            # No alpha; run rembg
            from rembg import remove
            image_input = remove(image_input)
        if crop_size_px != -1:
            alpha_np = np.asarray(image_input)[:, :, 3]
            coords = np.stack(np.nonzero(alpha_np), 1)[:, (1, 0)]
            min_x, min_y = np.min(coords, 0)
            max_x, max_y = np.max(coords, 0)
            ref = image_input.crop((min_x, min_y, max_x, max_y))
            w, h = ref.size
            if w < 400 or h < 400:
                ref = sr_front(ref)
            h2, w2 = ref.height, ref.width
            scale = crop_size_px / max(h2, w2)
            ref = ref.resize((int(scale * w2), int(scale * h2)))
            return add_margin(ref, size=image_size)
        return add_margin(image_input, size=max(image_input.height, image_input.width)).resize((image_size, image_size))

    # Work in a temp dir so we don't pollute /output. Final mesh is copied at end.
    out_obj = Path(output_obj_path)
    out_obj.parent.mkdir(parents=True, exist_ok=True)
    stem = out_obj.stem

    with tempfile.TemporaryDirectory(prefix='w3d_') as td:
        work = Path(td)
        # Wonder3D writes outputs to {output_path}/{stem}/...
        sub = work / stem
        sub.mkdir(parents=True, exist_ok=True)

        # Stage 1: preprocess input
        print(f"Stage 1: preprocess {input_image_path}")
        raw = prepare_input_image(input_image_path)
        img = process_input(raw, image_size=2048, crop_size_px=crop_size * 8)
        front_path = sub / 'front_img.png'
        img.save(front_path)

        # Stage 2: MV diffusion (color + normal, 6 views)
        print("Stage 2: MV diffusion (color + normal, 6 views)")
        normals_pred, images_pred = pred_multiview_joint(
            img, mv_pipe, seed=seed, crop_size=crop_size,
            camera_type=camera_type, cfg=config_mv,
            case_name=str(front_path), output_path=str(work))

        # Stage 3: Coarse mesh from MV normals
        print("Stage 3: coarse mesh reconstruction")
        rgb_rd, normal_rd, vertices, faces = coarse_recon(
            front_image=img, rgbs=images_pred, normals=normals_pred,
            camera_type=camera_type, scence_name=stem,
            crop_size=crop_size, output_path=str(work))

        # Stage 4: iterative refinement (default 2 iters)
        mv_n_4 = views_6to4(normals_pred)
        mv_i_4 = views_6to4(images_pred)
        rendered_normals_pop = normal_rd
        rendered_imgs_pop = rgb_rd
        v_cur, f_cur = vertices, faces

        for ri in range(num_refine):
            print(f"Stage 4.{ri+1}: enhancement + iterative refine")
            normals_pred2, images_pred2 = pred_enhancement_joint(
                mv_image=mv_i_4, mv_normlas=mv_n_4,
                renderd_mv_image=rendered_imgs_pop,
                renderd_mv_normal=rendered_normals_pop,
                front_image=img, pipeline=en_pipe, seed=seed,
                crop_size=crop_size, camera_type=camera_type,
                cfg=config_cn, case_name=str(front_path),
                refine_idx=ri, output_path=str(work))

            rgb_rd2, normal_rd2, v_cur, f_cur = iterative_refine(
                vertex_init=v_cur, face_init=f_cur,
                front_image=img, rgbs=images_pred2, normals=normals_pred2,
                camera_type=camera_type, scence_name=stem,
                crop_size=crop_size, output_path=str(work),
                refine_idx=ri, do_sr=(ri == num_refine - 1))
            rendered_normals_pop = normal_rd2
            rendered_imgs_pop = rgb_rd2
            torch.cuda.empty_cache()

        # Wonder3D++ saves final textured mesh as model.glb in 3d_model/.
        glb_path = sub / '3d_model' / 'model.glb'
        if not glb_path.exists():
            # Fallback: look for any glb under sub
            cands = list(sub.rglob('*.glb'))
            if cands:
                glb_path = cands[-1]
        print(f"Loading final mesh: {glb_path}")
        if glb_path.exists():
            scene = trimesh.load(str(glb_path))
            if hasattr(scene, 'geometry') and len(scene.geometry) > 0:
                mesh = trimesh.util.concatenate(list(scene.geometry.values()))
            else:
                mesh = scene
        else:
            # Fallback: build trimesh from returned vertices/faces
            print("model.glb missing, building mesh from refined vertices/faces")
            import numpy as np
            v = v_cur.detach().cpu().numpy() if hasattr(v_cur, 'detach') else _np.asarray(v_cur)
            f = f_cur.detach().cpu().numpy() if hasattr(f_cur, 'detach') else _np.asarray(f_cur)
            mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)

        mesh.export(str(out_obj))
        print(f"Saved: {out_obj} (V={len(mesh.vertices)}, F={len(mesh.faces)})")
    return str(out_obj)


def main():
    parser = argparse.ArgumentParser(description="Wonder3D++ 3D reconstruction")
    parser.add_argument('--input', '-i', type=str, help='Input image')
    parser.add_argument('--output', '-o', type=str, help='Output mesh (.obj)')
    parser.add_argument('--input-dir', type=str, help='Batch input directory')
    parser.add_argument('--output-dir', type=str, help='Batch output directory')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--camera-type', default='ortho', choices=['ortho', 'persp'])
    parser.add_argument('--crop-size', type=int, default=192)
    parser.add_argument('--num-refine', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if not args.input and not args.input_dir:
        parser.print_help()
        sys.exit(1)

    # Ensure ckpts + pipelines
    ensure_checkpoints()
    mv_pipe, en_pipe, cfg_mv, cfg_cn = load_pipelines(device=args.device)

    if args.input:
        out = args.output or args.input.replace('.png', '.obj')
        try:
            process_image_pipeline(mv_pipe, en_pipe, cfg_mv, cfg_cn,
                                    args.input, out,
                                    camera_type=args.camera_type,
                                    crop_size=args.crop_size,
                                    num_refine=args.num_refine,
                                    seed=args.seed, device=args.device)
        except Exception as e:
            print(f"FAILED for {args.input}: {e}")
            traceback.print_exc()
            sys.exit(2)
    elif args.input_dir:
        in_dir = Path(args.input_dir)
        out_dir = Path(args.output_dir or args.input_dir)
        for img in sorted(in_dir.glob('*.png')):
            out = out_dir / f"{img.stem}.obj"
            try:
                process_image_pipeline(mv_pipe, en_pipe, cfg_mv, cfg_cn,
                                        str(img), str(out),
                                        camera_type=args.camera_type,
                                        crop_size=args.crop_size,
                                        num_refine=args.num_refine,
                                        seed=args.seed, device=args.device)
            except Exception as e:
                print(f"FAILED for {img}: {e}")
                traceback.print_exc()


if __name__ == '__main__':
    main()
