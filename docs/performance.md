**English** · [繁體中文](zh-TW/performance.md)

# Performance

The target is a small ARM device, so the pipeline is built to do as little as possible. All numbers below were
measured on an 8-core x86 test server (OpenCV DNN on the CPU). Re-measure on your own hardware.

## Four power savers

| Saver | What it does | Parameters |
|---|---|---|
| **Motion gate** | Shrinks the frame to 64×48 grey and compares it with the previous one. Below the threshold and in a stable state, detection is skipped and the last result reused. A full detection still runs every `force_detect_seconds`, and always in MAYBE_AWAY / AWAY. | `motion_gate`, `motion_threshold`, `force_detect_seconds` |
| **Embedding throttle** | Reuses a face's embedding unless the face is new, moved / resized by more than `embed_move_thresh`, or `embed_interval` has passed. | `embed_throttle`, `embed_interval`, `embed_move_thresh` |
| **Adaptive fps** | The server suggests a frame rate: `fps_low` when stable, `fps_high` when faces appear, participants vanish, tracks just changed, or in MAYBE_AWAY / AWAY. | `adaptive_fps`, `fps_low`, `fps_high` |
| **ROI crop** | Crops the ROI plus `crop_margin`, scales it to at most `max_width`, detects, maps boxes back. | `roi_crop`, `crop_margin`, `max_width` |

### Benchmark

`uv run python scripts/benchmark.py` simulates 60 s at 1 fps: a person standing still (0–15 s), moving (15–30 s),
an empty ROI (30–45 s), the person back (45–60 s). Frames are synthesized and JPEG-coded in memory; no image file
is written. Best of 3 runs.

| Setting | ms / frame avg | p95 | Total CPU | vs. all off |
|---|---|---|---|---|
| All savers off | 21.2 | 27.2 | 6.27 s | — |
| ROI crop only | 16.4 | 23.0 | 4.43 s | −29% |
| Motion gate only | 10.1 | 25.9 | 2.94 s | −53% |
| Embedding throttle only | 14.4 | 25.5 | 4.50 s | −28% |
| All on, fixed fps | 7.9 | 22.1 | 2.06 s | −67% |
| All on + adaptive fps | 6.9 | 18.2 | 1.81 s | **−71%** |
| Pose, savers off | 31.1 | 37.9 | 7.93 s | +27% |
| All on + pose | 13.7 | 31.9 | 2.97 s | −53% |
| All on + pose + distance | 14.0 | 32.4 | 3.09 s | −51% |

Every setting still detected AWAY correctly. The p95 barely drops: frames that need a full detection cost the same,
the savings come from frames that need nothing.

## Where the time goes

Per-stage latency, 2 fps, a moving face at 640×480, all savers on (p50 ms):

| Stage | legacy + rules | kalman + hmm + score |
|---|---|---|
| JPEG decode | 0.7 | 0.6 |
| Motion gate | 0.2 | 0.2 |
| Face detection (YuNet, cropped) | 2.4 | 2.4 |
| **Embedding (SFace)** | **9.9** | **10.0** |
| Pose (MoveNet, when on) | 6.3 | 6.1 |
| Clothing histogram | — | 0.1 |
| Tracker + presence + intent | < 0.1 | 0.3 |
| **Total** (no pose / pose) | 13.7 / 19.9 | 14.0 / 20.2 |

- The face embedding is the biggest cost; then pose; then detection.
- The smarter layers (Kalman, HMM, intent) together cost less than 0.5 ms.
- 1280×720 input adds about 1 ms (only decoding gets slower, thanks to the ROI crop).
- Empty or static scenes finish most frames within 1 ms.
- WebSocket end-to-end p50: about 17–18 ms without pose, about 25 ms with pose.

`max_width` matters for small faces: at 320 px, faces under about 80 px tall were missed, so the default is 640
(detection ≈ 2.4 → 5 ms). Faces should be at least 40–50 px wide after downscaling.

## Embedding backends and NPU

`embed_backend` selects how SFace is computed. The model is the same; only the runtime changes.

| Backend | Model | Runtime | Use |
|---|---|---|---|
| `fp32` (default) | opencv_zoo SFace ONNX | OpenCV DNN, CPU | Reference, most accurate |
| `int8` | opencv_zoo official INT8 ONNX | OpenCV DNN, CPU | 4× smaller; often faster on ARM CPUs |
| `npu` | TFLite exported by this project (`int8` or `fp16`) | LiteRT, optional external delegate | Android NPU / GPU; XNNPACK on a server |

Whole pipeline with an embedding on every frame (p50 / p95 ms, 640×480):

| Backend | Embed | Total |
|---|---|---|
| fp32, OpenCV DNN | 10.5 / 12.7 | 14.7 / 17.2 |
| int8, OpenCV DNN | 12.2 / 14.6 | 16.6 / 19.4 |
| npu fp16, LiteRT XNNPACK | 3.6 / 4.1 | 7.7 / 8.4 |

Accuracy against fp32 (`scripts/eval_embed_backends.py`: 31 CC0 portraits, 6 augmentations each, 651 pairs,
threshold 0.40):

| Backend | Mean cosine vs fp32 | Same decision as fp32 |
|---|---|---|
| int8 ONNX | 0.972 | 99.85% |
| npu int8 TFLite | 0.991 | 100% |
| npu fp16 TFLite | 1.000 | 100% |

So `cos_match_int8` defaults to the same 0.40. This compares quantized models with fp32; it is not a recognition
accuracy benchmark.

### Getting the TFLite models

They are not in git (10 and 19 MB). Export them in a separate environment:

```bash
uv venv -p 3.12 /tmp/.venv-sface-export
VIRTUAL_ENV=/tmp/.venv-sface-export uv pip install onnx2tf tensorflow tf-keras onnx onnxruntime \
    onnx-graphsurgeon onnxsim sng4onnx psutil ai-edge-litert opencv-python-headless
/tmp/.venv-sface-export/bin/python scripts/export_sface_tflite.py
```

The script works on a copy of the ONNX file, rewrites PReLU so it becomes a single `PRELU` op (NPU-friendly),
writes an fp16 and a full-INT8 model (calibrated on CC0 portraits in memory) and checks both against fp32.

### Using a delegate

- On the server, set `PRESENCE_NPU_DELEGATE=/path/to/delegate.so`. For security it can only be set through the
  environment, never over the WebSocket: loading a shared library runs native code.
- If the delegate fails to load, LiteRT falls back to the CPU. The resource panel shows the backend actually in
  use and why.
- Without a delegate, `npu_precision = int8` automatically uses the fp16 model, which is about ten times faster on
  an x86 CPU.
- On Android, use the same `.tflite` in the app with the NNAPI or a vendor delegate. Input: int8
  `[1, 112, 112, 3]`, RGB − 128, aligned with the same 5-point similarity transform. Measure with TensorFlow's
  `benchmark_model` (`--use_nnapi`, `--external_delegate_path`, `--enable_op_profiling`) to see which ops really
  run on the NPU. None of this has been validated on a device yet.

## Audio cost

The PureSound pipeline costs about 5 ms per 20 ms chunk on one core (real-time factor ≈ 0.25), far more than vision
at 1–2 fps. Measure on the target device first. See [Audio](audio.md#performance).

## Ideas for an embedded port

- Run detection and embedding on the NPU; keep the gate, tracker and state machine on the CPU.
- The tracker and state machine are small and dependency-free; they can be ported (for example to Rust or Kotlin)
  with this Python version and its scenario tests as the reference.
- Use the camera's Y plane directly for the motion gate and skip JPEG encode / decode.
- With no face and no sound, drop to 0.2 fps and wake on voice activity.
