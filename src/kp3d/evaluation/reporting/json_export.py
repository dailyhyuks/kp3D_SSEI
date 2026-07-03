"""JSON result export and import utilities."""

import json
from pathlib import Path
from typing import Any, Dict


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save evaluation results to JSON file.

    Args:
        results: Results dictionary to save.
        output_path: Output JSON file path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to native Python
    clean = _convert_numpy(results)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)


def load_results(json_path: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file.

    Args:
        json_path: Path to JSON results file.

    Returns:
        Results dictionary.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _convert_numpy(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
