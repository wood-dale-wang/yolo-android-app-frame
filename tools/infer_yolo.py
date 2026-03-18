#!/usr/bin/env python3
"""Run single-image YOLO inference and save annotated output image.

Usage examples:
  uv run python tools/infer_yolo.py --model yolov8n.pt --source demo.jpg
  uv run python tools/infer_yolo.py --model models/yolov8n.pt --source demo.jpg --output inference_out --conf 0.3
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO inference on a single image")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model name/path")
    parser.add_argument("--source", required=True, help="Path to a single input image")
    parser.add_argument(
        "--output",
        default="inference_out",
        help="Directory for annotated output image",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
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


def resolve_source_image(source_arg: str) -> Path:
    source = Path(source_arg).expanduser()
    if not source.is_absolute():
        source = (Path.cwd() / source).resolve()

    if not source.exists():
        raise FileNotFoundError(f"Input image not found: {source}")
    if not source.is_file():
        raise ValueError(f"Input path must be a single image file: {source}")

    return source


def main() -> None:
    args = parse_args()
    project_root = get_project_root()
    models_dir = project_root / "models"

    model_path = resolve_model_path(args.model, models_dir)
    source_image = resolve_source_image(args.source)

    output_dir = Path(args.output).expanduser()
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    ensure_dir(output_dir)

    model = YOLO(str(model_path))
    results = model.predict(source=str(source_image), conf=args.conf, verbose=False)
    if not results:
        raise RuntimeError("No prediction result was returned")

    result = results[0]
    out_name = f"{source_image.stem}_pred{source_image.suffix or '.jpg'}"
    out_path = output_dir / out_name
    result.save(filename=str(out_path))

    print(f"[OK] Input image: {source_image}")
    print(f"[OK] Model used: {model_path}")
    print(f"[OK] Output image: {out_path}")

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        print("[OK] No detections")
        return

    names = result.names
    print(f"[OK] Detections: {len(boxes)}")
    for idx in range(len(boxes)):
        box = boxes[idx]
        display_idx = idx + 1
        cls_id = int(box.cls.squeeze().item())
        conf = float(box.conf.squeeze().item())
        coords = box.xyxy.squeeze().tolist()
        if not isinstance(coords, list):
            coords = [float(coords)]
        label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)

        if len(coords) >= 4:
            x1, y1, x2, y2 = coords[:4]
            print(
                f"[DET {display_idx}] class={label} conf={conf:.4f} "
                f"box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})"
            )
        else:
            print(f"[DET {display_idx}] class={label} conf={conf:.4f} box={coords}")


if __name__ == "__main__":
    main()
