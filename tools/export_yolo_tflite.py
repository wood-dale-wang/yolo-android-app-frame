#!/usr/bin/env python3
"""Export Ultralytics YOLO model to ncnn and optionally copy to Android assets.

Usage examples:
    uv run python tools/export_yolo_tflite.py --model yolo26s.pt --imgsz 640 --copy-to-assets
    uv run python tools/export_yolo_tflite.py --model yolov8n.pt --imgsz 640 --half
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO to ncnn for Android")
    parser.add_argument("--model", default="yolo26s.pt", help="YOLO model name/path")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export with FP16 where backend supports it",
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
        "--android-param-name",
        default="yolo26s.ncnn.param",
        help="Destination .param filename in assets",
    )
    parser.add_argument(
        "--android-bin-name",
        default="yolo26s.ncnn.bin",
        help="Destination .bin filename in assets",
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


def choose_ncnn_files(candidates: Iterable[Path]) -> tuple[Path, Path]:
    candidate_list = list(candidates)
    params = [p for p in candidate_list if p.suffix.lower() == ".param"]
    bins = [p for p in candidate_list if p.suffix.lower() == ".bin"]
    if not params:
        raise FileNotFoundError("No .param file found in ncnn export result")
    if not bins:
        raise FileNotFoundError("No .bin file found in ncnn export result")

    def pick(paths: list[Path], preferred_name: str) -> Path:
        for path in paths:
            if path.name == preferred_name:
                return path
        return paths[0]

    return pick(params, "model.ncnn.param"), pick(bins, "model.ncnn.bin")


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
        "format": "ncnn",
        "imgsz": args.imgsz,
        "half": args.half,
        "project": str(output_dir),
        "name": "ultralytics_ncnn",
    }

    result = model.export(**export_kwargs)
    result_path = Path(result)

    export_root = result_path if result_path.is_dir() else output_dir
    param_path, bin_path = choose_ncnn_files(export_root.rglob("*"))

    labels_out = export_root / "labels.txt"
    write_labels(labels_out, model.names)

    print(f"[OK] Export root: {export_root}")
    print(f"[OK] NCNN param: {param_path}")
    print(f"[OK] NCNN bin: {bin_path}")
    print(f"[OK] Labels file: {labels_out}")

    if args.copy_to_assets:
        assets_dir = Path(args.assets_dir).resolve()
        ensure_dir(assets_dir)

        dest_param = assets_dir / args.android_param_name
        dest_bin = assets_dir / args.android_bin_name
        dest_labels = assets_dir / "labels.txt"

        shutil.copy2(param_path, dest_param)
        shutil.copy2(bin_path, dest_bin)
        shutil.copy2(labels_out, dest_labels)

        print(f"[OK] Copied param to: {dest_param}")
        print(f"[OK] Copied bin to: {dest_bin}")
        print(f"[OK] Copied labels to: {dest_labels}")


if __name__ == "__main__":
    main()
