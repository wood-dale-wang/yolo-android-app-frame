#!/usr/bin/env python3
"""Export Ultralytics YOLO model to TensorFlow Lite and optionally copy to Android assets.

Usage examples:
  uv run python tools/export_yolo_tflite.py --model yolov8n.pt --imgsz 640 --quant fp16 --copy-to-assets
  uv run python tools/export_yolo_tflite.py --model yolov8n.pt --imgsz 640 --quant int8 --nms
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO to TFLite for Android")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model name/path")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument(
        "--quant",
        choices=["fp32", "fp16", "int8"],
        default="fp16",
        help="Quantization mode",
    )
    parser.add_argument(
        "--nms",
        action="store_true",
        help="Export model with NMS head when supported",
    )
    parser.add_argument(
        "--output-dir",
        default="export_out",
        help="Directory for export outputs",
    )
    parser.add_argument(
        "--copy-to-assets",
        action="store_true",
        help="Copy exported model and labels to Android assets",
    )
    parser.add_argument(
        "--assets-dir",
        default="app/src/main/assets",
        help="Android assets directory",
    )
    parser.add_argument(
        "--android-model-name",
        default="yolov8n.tflite",
        help="Destination model filename in assets",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_model_path(model_arg: str, models_dir: Path) -> Path:
    ensure_dir(models_dir)
    requested = Path(model_arg).expanduser()

    if requested.is_absolute() and requested.exists():
        resolved = requested.resolve()
        print(f"[OK] Using explicit model path: {resolved}")
        return resolved

    if requested.exists():
        resolved = requested.resolve()
        print(f"[OK] Using local model path: {resolved}")
        return resolved

    cached = models_dir / requested.name
    if cached.exists():
        resolved = cached.resolve()
        print(f"[OK] Using cached model from models: {resolved}")
        return resolved

    if requested.parent != Path("."):
        raise FileNotFoundError(
            f"Model path not found: {requested}. For auto-download, pass only the model name (for example yolov8n.pt)."
        )

    print(f"[INFO] Model not found in models, downloading: {requested.name}")
    downloaded_model = YOLO(requested.name)

    candidates: list[Path] = []
    ckpt_path = getattr(downloaded_model, "ckpt_path", None)
    if ckpt_path:
        ckpt_candidate = Path(str(ckpt_path)).expanduser()
        if ckpt_candidate.exists():
            candidates.append(ckpt_candidate.resolve())

    local_candidate = Path.cwd() / requested.name
    if local_candidate.exists():
        candidates.append(local_candidate.resolve())

    if cached.exists():
        resolved = cached.resolve()
        print(f"[OK] Downloaded model already in models: {resolved}")
        return resolved

    for source in candidates:
        if source.exists():
            ensure_dir(cached.parent)
            shutil.copy2(source, cached)
            resolved = cached.resolve()
            print(f"[OK] Downloaded model saved to models: {resolved}")
            return resolved

    raise FileNotFoundError(
        f"Model download finished but no model file was found for {requested.name}."
    )


def choose_tflite_file(candidates: Iterable[Path], quant: str) -> Path:
    files = [p for p in candidates if p.suffix == ".tflite"]
    if not files:
        raise FileNotFoundError("No .tflite file found in export result")

    # Prefer quantization-specific filenames when available.
    preferred = {
        "fp32": ["float32", "full_integer"],
        "fp16": ["float16", "fp16"],
        "int8": ["int8", "integer"],
    }[quant]

    lower_map = {p: p.name.lower() for p in files}
    for keyword in preferred:
        for path, lower_name in lower_map.items():
            if keyword in lower_name:
                return path

    return files[0]


def write_labels(path: Path, names: dict | list) -> None:
    if isinstance(names, dict):
        ordered = [names[idx] for idx in sorted(names.keys())]
    else:
        ordered = list(names)
    path.write_text("\n".join(str(item) for item in ordered) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = get_project_root()
    models_dir = project_root / "models"

    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)

    model_path = resolve_model_path(args.model, models_dir)
    model = YOLO(str(model_path))
    export_kwargs = {
        "format": "tflite",
        "imgsz": args.imgsz,
        "half": args.quant == "fp16",
        "int8": args.quant == "int8",
        "nms": args.nms,
        "project": str(output_dir),
        "name": "ultralytics_tflite",
    }

    result = model.export(**export_kwargs)
    result_path = Path(result)

    if result_path.is_file() and result_path.suffix == ".tflite":
        tflite_path = result_path
        export_root = result_path.parent
    else:
        export_root = result_path if result_path.is_dir() else output_dir
        tflite_path = choose_tflite_file(export_root.rglob("*.tflite"), args.quant)

    labels_out = export_root / "labels.txt"
    write_labels(labels_out, model.names)

    print(f"[OK] Export root: {export_root}")
    print(f"[OK] TFLite model: {tflite_path}")
    print(f"[OK] Labels file: {labels_out}")

    if args.copy_to_assets:
        assets_dir = Path(args.assets_dir).resolve()
        ensure_dir(assets_dir)

        dest_model = assets_dir / args.android_model_name
        dest_labels = assets_dir / "labels.txt"

        shutil.copy2(tflite_path, dest_model)
        shutil.copy2(labels_out, dest_labels)

        print(f"[OK] Copied model to: {dest_model}")
        print(f"[OK] Copied labels to: {dest_labels}")


if __name__ == "__main__":
    main()
