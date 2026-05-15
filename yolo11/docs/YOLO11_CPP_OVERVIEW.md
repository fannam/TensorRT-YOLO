# YOLO11 C++ Overview

This document explains the C++ portion of `yolo11/cpp/*` and how it connects to `shared/cpp/bytetrack`.

## 1. Module Map

- `yolo11/cpp/detect`
  Static-image detection binary. It has one primary output tensor from the detection head.
- `yolo11/cpp/pose`
  Uses the same overall structure as `detect`, but each candidate also carries 17 keypoints.
- `yolo11/cpp/segment`
  Uses two output tensors: a detection head and a proto head for mask reconstruction.
- `yolo11/cpp/track`
  There is no dedicated tracking head. This module combines the `detect` detector with `BYTETracker`.
- `shared/cpp/bytetrack`
  Contains the tracking state machine, Kalman filter, IoU cost computation, and assignment solver.

## 2. End-to-End Data Flow

### Detect

1. `main.cpp` resolves the ONNX and `.plan` paths.
2. `YoloDetector` in `src/infer.cpp`:
   - loads `.plan` if it already exists
   - otherwise builds the engine from ONNX and serializes it
3. `src/preprocess.cu`
   - letterbox
   - bilinear resize
   - BGR -> RGB
   - HWC -> CHW
   - normalize to `[0, 1]`
4. TensorRT enqueues work on the same CUDA stream.
5. `src/postprocess.cu`
   - transpose `[C, N] -> [N, C]`
   - decode bbox/class
   - run GPU NMS
6. `scale_bbox()` maps boxes from the padded 640x640 space back to the original image.

### Pose

The flow is the same as `detect`, but the decode step also preserves `17 * 3 = 51` keypoint values.

After copying back to the host:

- `scale_bbox()` rescales the box to the original image
- `scale_kpt_coords()` rescales each keypoint to the original image
- `draw_image()` can render boxes, keypoints, and the COCO skeleton

### Segment

The flow differs in two places:

1. TensorRT has two outputs:
   - `proto`: typically `[1, 32, 160, 160]`
   - `detect`: typically `[1, 116, 8400]`
2. After decode + NMS, `process_mask()`:
   - gathers the 32-element coefficient vector for each detection
   - multiplies those coefficients by the proto tensor to produce masks at `160x160`
   - crops by bbox
   - removes letterbox padding
   - resizes each mask to the original image size

## 3. Tensor Shape Map

### Detect

- Input: `[1, 3, 640, 640]`
- Raw output: `[1, 84, 8400]`
- After transpose: `[8400, 84]`
- After decode: `[count, boxes...]`, with 7 floats per box

`84 = 4 bbox + 80 classes`

### Pose

- Input: `[1, 3, 640, 640]`
- Raw output: `[1, 56, 8400]`
- After transpose: `[8400, 56]`
- After decode: each box has `7 + 51 = 58` floats

`56 = 4 bbox + 1 class + 51 keypoint values`

### Segment

- Input: `[1, 3, 640, 640]`
- Proto output: `[1, 32, 160, 160]`
- Detect output: `[1, 116, 8400]`
- After detect transpose: `[8400, 116]`
- After decode: each box has `7 + 32 = 39` floats

`116 = 4 bbox + 80 classes + 32 mask coefficients`

## 4. Memory Map

### Long-Lived Resources

Each `YoloDetector` keeps:

- `runtime`
- `engine`
- `context`
- `cudaStream_t stream`
- `vBufferD`: TensorRT bindings
- `transposeDevice`
- `decodeDevice`
- `outputData` on the host

### Detect/Pose

- `vBufferD[input]`: float32 input tensor
- `vBufferD[output]`: raw output tensor
- `transposeDevice`: scratch buffer for `[C, N] -> [N, C]`
- `decodeDevice`: flat buffer after decode/NMS

### Segment

Additional resources:

- `vBufferD[proto]`: proto tensor from the mask head
- temporary buffers inside `process_mask()`:
  - `maskCoefDevice`
  - `maskDevice`
  - `bboxDevice`
  - `cutMaskDevice`
  - `scaledMaskDevice`

These temporary buffers are allocated inside `process_mask()` and freed immediately after the masks are copied back to the host.

## 5. What Changes Between TRT8 and TRT10

The repo keeps two compatibility branches:

- TRT10:
  - enumerate tensors with `getNbIOTensors()`
  - set bindings by tensor name
  - enqueue with `enqueueV3()`
- TRT8/9:
  - enumerate bindings with `getNbBindings()`
  - set shapes by binding index
  - enqueue with `enqueueV2()`

This is heavily commented in the code because shape and binding lookup are common sources of confusion when upgrading TensorRT versions.

## 6. FP16 and INT8

- `bFP16Mode`
  Enables the FP16 builder flag. It does not change the runtime flow.
- `bINT8Mode`
  Enables the calibration path:
  - read images from `../calibrator`
  - preprocess on the CPU
  - write or read `int8.cache`

INT8 only affects engine build time. It does not change the normal inference loop once the `.plan` file already exists.

## 7. Detect vs Pose vs Segment

- `detect`
  The lightest path, requiring only bbox/class output.
- `pose`
  Extends detect by attaching 51 keypoint values to each candidate.
- `segment`
  Has the heaviest postprocess path because it reconstructs masks from `proto + coefficients`.

If you want to understand the common pattern across all three tasks, read `detect` first, then `pose`, then `segment`.

## 8. What the Tracker Consumes From the Detector

`yolo11/cpp/track/main.cpp` converts detect output into:

- `cv::Rect_<float> rect` in `tlwh` format
- `label`
- `prob`

`BYTETracker.update()` does not use feature embeddings. It only needs:

- bbox
- score
- prior state
- Kalman prediction
- IoU-based cost matrix

## 9. Recommended Reading Order

Suggested order for newcomers:

1. `yolo11/cpp/detect/main.cpp`
2. `yolo11/cpp/detect/include/infer.h`
3. `yolo11/cpp/detect/src/infer.cpp`
4. `yolo11/cpp/detect/src/preprocess.cu`
5. `yolo11/cpp/detect/src/postprocess.cu`
6. `yolo11/cpp/pose/src/infer.cpp`
7. `yolo11/cpp/segment/src/infer.cpp`
8. `yolo11/cpp/track/main.cpp`
9. `shared/cpp/bytetrack/include/STrack.h`
10. `shared/cpp/bytetrack/src/BYTETracker.cpp`

## 10. Note on `yolo26`

`yolo26` should currently be treated as a scaffold with a similar layout.

- Do not use it as the source of truth for detailed C++ runtime behavior.
- When you need the real implementation details, prefer `yolo11`.
