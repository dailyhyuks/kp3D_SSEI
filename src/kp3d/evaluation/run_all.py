"""CLI entry point for evaluation experiments.

Usage:
    python -m kp3d.evaluation.run_all --config eval_config.yaml
    python -m kp3d.evaluation.run_all enhancement
    python -m kp3d.evaluation.run_all inpainting
    python -m kp3d.evaluation.run_all ablation
    python -m kp3d.evaluation.run_all report --results-dir evaluation_results/
"""

import argparse
import sys
from pathlib import Path
from typing import Optional


def run_enhancement(config_path: Optional[str] = None, dry_run: bool = False):
    """Run Enhancement (Stage 1) evaluation."""
    from kp3d.evaluation.config import load_config
    from kp3d.evaluation.experiments.restoration_eval import RestorationExperiment
    from kp3d.evaluation.reporting.json_export import save_results
    from kp3d.evaluation.reporting.latex_table import generate_enhancement_table

    config = load_config(config_path)
    if dry_run:
        config.dry_run = True

    print("=" * 60)
    print("Enhancement (Weave Removal) Evaluation")
    print("=" * 60)
    print(f"  Data dir: {config.data_dir}")
    print(f"  Baselines: {config.enhancement_baselines}")
    print(f"  Our preset: {config.weave_removal_preset}")
    print(f"  Dry run: {config.dry_run}")
    print()

    experiment = RestorationExperiment(config)
    results = experiment.run()

    if config.dry_run:
        print(f"\n[DRY RUN] Would process {len(results)} image-method pairs")
        return

    aggregated = experiment.aggregate()

    # Save results
    output_dir = Path(config.output_dir)
    save_results(aggregated, str(output_dir / "enhancement_results.json"))
    print(f"\nResults saved to: {output_dir / 'enhancement_results.json'}")

    # Generate LaTeX table
    latex = generate_enhancement_table(aggregated)
    latex_path = output_dir / "enhancement_table.tex"
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    latex_path.write_text(latex, encoding="utf-8")
    print(f"LaTeX table saved to: {latex_path}")
    print("\n" + latex)


def run_inpainting(config_path: Optional[str] = None, dry_run: bool = False):
    """Run Inpainting (Stage 4) evaluation."""
    from kp3d.evaluation.config import load_config
    from kp3d.evaluation.experiments.inpainting_eval import InpaintingExperiment
    from kp3d.evaluation.reporting.json_export import save_results
    from kp3d.evaluation.reporting.latex_table import generate_inpainting_table

    config = load_config(config_path)
    if dry_run:
        config.dry_run = True

    print("=" * 60)
    print("Inpainting (SSEI V25) Evaluation")
    print("=" * 60)
    print(f"  Data dir: {config.data_dir}")
    print(f"  Baselines: {config.inpainting_baselines}")
    print(f"  Occlusion types: {config.occlusion_types}")
    print(f"  Dry run: {config.dry_run}")
    print()

    experiment = InpaintingExperiment(config)
    results = experiment.run()

    if config.dry_run:
        print(f"\n[DRY RUN] Would process {len(results)} combinations")
        return

    aggregated = experiment.aggregate()

    # Save results
    output_dir = Path(config.output_dir)
    save_results(aggregated, str(output_dir / "inpainting_results.json"))
    print(f"\nResults saved to: {output_dir / 'inpainting_results.json'}")

    # Generate LaTeX table
    latex = generate_inpainting_table(aggregated)
    latex_path = output_dir / "inpainting_table.tex"
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    latex_path.write_text(latex, encoding="utf-8")
    print(f"LaTeX table saved to: {latex_path}")
    print("\n" + latex)


def run_ablation(config_path: Optional[str] = None, dry_run: bool = False):
    """Run E2E Pipeline Ablation Study."""
    from kp3d.evaluation.config import load_config
    from kp3d.evaluation.experiments.e2e_ablation import AblationExperiment
    from kp3d.evaluation.reporting.json_export import save_results
    from kp3d.evaluation.reporting.latex_table import generate_latex_table

    config = load_config(config_path)
    if dry_run:
        config.dry_run = True

    print("=" * 60)
    print("E2E Pipeline Ablation Study")
    print("=" * 60)
    print(f"  Data dir: {config.data_dir}")
    print(f"  Dry run: {config.dry_run}")
    print()

    experiment = AblationExperiment(config)
    results = experiment.run()

    if config.dry_run:
        print(f"\n[DRY RUN] Would run ablation study")
        return

    aggregated = experiment.aggregate()

    # Save results
    output_dir = Path(config.output_dir)
    save_results(aggregated, str(output_dir / "ablation_results.json"))
    print(f"\nResults saved to: {output_dir / 'ablation_results.json'}")

    # Generate LaTeX table
    latex = generate_latex_table(
        aggregated,
        metrics=["psnr_vs_full", "ssim_vs_full", "grid_energy", "edge_preservation"],
        caption="E2E Pipeline Ablation Study.",
        label="tab:ablation",
        metric_formats={
            "psnr_vs_full": "{:.2f}",
            "ssim_vs_full": "{:.4f}",
            "grid_energy": "{:.4f}",
            "edge_preservation": "{:.3f}",
        },
        metric_directions={
            "psnr_vs_full": "higher",
            "ssim_vs_full": "higher",
            "grid_energy": "lower",
            "edge_preservation": "higher",
        },
    )
    latex_path = output_dir / "ablation_table.tex"
    latex_path.write_text(latex, encoding="utf-8")
    print(f"LaTeX table saved to: {latex_path}")
    print("\n" + latex)


def run_report(results_dir: str):
    """Generate all reports from existing JSON results."""
    from kp3d.evaluation.reporting.json_export import load_results
    from kp3d.evaluation.reporting.latex_table import (
        generate_enhancement_table,
        generate_inpainting_table,
        generate_latex_table,
    )

    results_path = Path(results_dir)

    print("=" * 60)
    print("Report Generation")
    print("=" * 60)

    # Enhancement table
    enh_path = results_path / "enhancement_results.json"
    if enh_path.exists():
        data = load_results(str(enh_path))
        latex = generate_enhancement_table(data)
        print("\n--- Enhancement Table ---")
        print(latex)

    # Inpainting table
    inp_path = results_path / "inpainting_results.json"
    if inp_path.exists():
        data = load_results(str(inp_path))
        latex = generate_inpainting_table(data)
        print("\n--- Inpainting Table ---")
        print(latex)

    # Ablation table
    abl_path = results_path / "ablation_results.json"
    if abl_path.exists():
        data = load_results(str(abl_path))
        latex = generate_latex_table(
            data,
            metrics=["psnr_vs_full", "ssim_vs_full", "grid_energy", "edge_preservation"],
            caption="E2E Pipeline Ablation Study.",
            label="tab:ablation",
        )
        print("\n--- Ablation Table ---")
        print(latex)


def main():
    parser = argparse.ArgumentParser(
        description="Korean Painting 3D - Evaluation Framework",
        prog="python -m kp3d.evaluation.run_all",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["enhancement", "inpainting", "ablation", "report", "all"],
        default="all",
        help="Evaluation to run (default: all)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup without full computation",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="evaluation_results",
        help="Results directory for report command",
    )

    args = parser.parse_args()

    if args.command == "enhancement":
        run_enhancement(args.config, args.dry_run)
    elif args.command == "inpainting":
        run_inpainting(args.config, args.dry_run)
    elif args.command == "ablation":
        run_ablation(args.config, args.dry_run)
    elif args.command == "report":
        run_report(args.results_dir)
    elif args.command == "all":
        run_enhancement(args.config, args.dry_run)
        print("\n")
        run_inpainting(args.config, args.dry_run)
        print("\n")
        run_ablation(args.config, args.dry_run)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
