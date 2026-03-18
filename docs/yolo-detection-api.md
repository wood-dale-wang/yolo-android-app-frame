# YOLO Detection Internal API (Android, Kotlin)

## Overview
This document describes the internal API for the on-device object detection pipeline:
- Image input from local file or camera snapshot.
- TensorFlow Lite YOLO inference.
- Output of detected object list and annotated image.

Current default model contract:
- Asset model file: `yolov8n.tflite`
- Labels file: `labels.txt`
- Input image source: gallery (`GetContent`) or camera (`TakePicturePreview`)

## Module Boundaries

### 1) Detector Layer
File: `app/src/main/kotlin/org/example/ml/YoloDetector.kt`

#### Constructor
`YoloDetector(context, modelAssetPath = "yolov8n.tflite", labelsAssetPath = "labels.txt")`

Responsibilities:
- Load TFLite model and labels from assets.
- Read model input/output tensor shape.
- Run preprocessing, inference, parsing, and NMS.

#### Method
`detect(bitmap, confidenceThreshold = 0.3f, iouThreshold = 0.45f, maxResults = 200): List<DetectionResult>`

Input:
- `bitmap`: source image for inference.
- `confidenceThreshold`: class score filter threshold.
- `iouThreshold`: IoU threshold used by NMS.
- `maxResults`: max number of objects returned after NMS.

Output:
- List of `DetectionResult` sorted by score descending.

Coordinate parsing notes:
- Detector supports two common YOLO TFLite box output styles:
- Normalized box (`cx, cy, w, h` in range 0~1)
- Pixel box (`cx, cy, w, h` in model input space)
- Runtime parsing auto-detects coordinate style and maps to original bitmap pixels.
- `DetectionResult.boundingBox` always uses pixel coordinates in source image space.

Exception behavior:
- Throws runtime exceptions for model load or inference failures.
- Callers should catch exceptions and map to user-facing error state.

### 2) Result Model
File: `app/src/main/kotlin/org/example/ml/DetectionResult.kt`

`DetectionResult` fields:
- `classId: Int`
- `label: String`
- `score: Float`
- `boundingBox: RectF` (pixel coordinates relative to original bitmap)

### 3) Annotation Layer
File: `app/src/main/kotlin/org/example/ml/ImageAnnotator.kt`

#### Method
`annotate(source, detections): Bitmap`

Behavior:
- Draws rectangle + label text on a mutable bitmap copy.
- Returns new annotated bitmap; does not mutate caller-owned immutable source.

### 4) ViewModel Layer
File: `app/src/main/kotlin/org/example/ui/MainViewModel.kt`

`AppUiState`:
- `sourceImage: Bitmap?`
- `annotatedImage: Bitmap?`
- `detections: List<DetectionResult>`
- `isDetecting: Boolean`
- `errorMessage: String?`

Methods:
- `detectFromBitmap(bitmap)`: trigger full detection pipeline.
- `clearError()`: clear consumable error state.

Threading model:
- Detection is executed in `Dispatchers.Default`.
- State updates are performed in ViewModel coroutine scope.

### 5) UI Entry Layer
File: `app/src/main/kotlin/org/example/App.kt`

Input channel APIs used:
- Local file: `ActivityResultContracts.GetContent()`
- Camera snapshot: `ActivityResultContracts.TakePicturePreview()`

Output rendering:
- Annotated image preview via Compose `Image`.
- Detected object list rendered in a fixed-height, scrollable container.
- Each list item displays label, confidence, and bounding box coordinates.

## End-to-End Call Sequence
1. User chooses image source.
2. UI obtains `Bitmap` from URI or camera preview callback.
3. UI calls `MainViewModel.detectFromBitmap(bitmap)`.
4. ViewModel normalizes bitmap format and invokes `YoloDetector.detect(...)`.
5. Detector returns `List<DetectionResult>`.
6. ViewModel calls `ImageAnnotator.annotate(...)` and updates `AppUiState`.
7. Compose recomposes: object list and annotated image are displayed.

## UI Result Item Format
Each detection item in UI includes:
- Class label: `label`
- Confidence: `score`
- Box coordinates (pixel): `[left, top, right, bottom]`

Example:
- `person | 92%`
- `坐标: [121, 84, 288, 412]`

## Error Handling Conventions
- Model missing/corrupted: detector throws; ViewModel maps to `errorMessage`.
- Empty detection: valid result, not an error (`detections.isEmpty()`).
- Label file missing: detector falls back to `class_<id>` naming.

## Troubleshooting
- Symptom: boxes cluster near top-left corner.
- Cause: coordinate output type mismatch (normalized vs pixel) during post-processing.
- Fix: detector now auto-detects coordinate style and applies proper mapping.

- Symptom: detection list appears incomplete.
- Cause: list viewport clipping or result cap too low.
- Fix: UI list is scrollable and default `maxResults` increased to 200.

## Performance Notes
- Default threads: 4 (`Interpreter.Options.setNumThreads(4)`).
- Avoid repeated model creation; detector is held in ViewModel lifecycle.
- For slower devices, prefer fp16 model for better latency.

## Script Contract (Model Export)
Related script: `tools/export_yolo_tflite.py`

Recommended execution:
```bash
uv run python tools/export_yolo_tflite.py --model yolov8n.pt --imgsz 640 --quant fp16 --copy-to-assets
```

Expected side effects:
- Exported `.tflite` file and `labels.txt` generated under `export_out`.
- Model copied to `app/src/main/assets/yolov8n.tflite`.
- Labels copied to `app/src/main/assets/labels.txt`.
