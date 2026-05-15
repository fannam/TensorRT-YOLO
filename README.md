# TensorRT-YOLO

> **Based on original work by [emptysoal](https://github.com/emptysoal/TensorRT-YOLO11).**
> This repo now uses a model-first layout so multiple YOLO families can live side by side under one tree.
> Current implementation is complete for **YOLO11** and scaffolded for **YOLO26**.

## Introduction

- Deploy `YOLO11` detect, pose, segment, and tracking tasks using TensorRT;
- Reserve the same layout for future model families such as `YOLO26`;
- Supports **TensorRT 8** and **TensorRT 10** (parallel support);
- Supports `Jetson` series (tested on **Jetson AGX Orin**) and `Linux x86_64`;
- No CUDA-supported OpenCV required — all tensor operations for pre/post-processing are implemented via CUDA programming;
- Model conversion: `.pt` -> `.onnx` -> `.plan(.engine)`;
- Both `Python` and `cpp` APIs implemented;
- Object-oriented design — easy to integrate into other projects;
- `C++` version compiles to a shared library for use as an interface in other projects;

## Effect

|            input image            |                detect                 |
| :-------------------------------: | :-----------------------------------: |
|      ![004](assets/005.jpeg)      | ![004_detect](assets/005_detect.jpeg) |
|             **pose**              |              **segment**              |
| ![004_pose](assets/005_pose.jpeg) |    ![004_seg](assets/005_seg.jpeg)    |

- ByteTrack

![result](./assets/result.gif)

## Inference speed

|        | detect | pose  | segment |
| :----: | :----: | :---: | :-----: |
|  C++   |  3 ms  | 4 ms  |  6 ms   |
| python | 10 ms  | 13 ms |  45 ms  |

- Inference time includes pre-processing, model inference, and post-processing
- Benchmarked on `x86_64 Linux`, `Ubuntu`, GPU: `GeForce RTX 2080 Ti`

## Environment

### Requirements

- `TensorRT 8.0+` or `TensorRT 10.0+`
- `OpenCV 3.4.0+`

### Linux x86_64 — Docker (recommended)

For TensorRT 8:
```bash
docker pull nvcr.io/nvidia/tensorrt:22.04-py3
```

| CUDA   | cuDNN    | TensorRT | Python |
| ------ | -------- | -------- | ------ |
| 11.6.2 | 8.4.0.27 | 8.2.4.2  | 3.8.10 |

For TensorRT 10:
```bash
docker pull nvcr.io/nvidia/tensorrt:24.05-py3
```

Then install OpenCV manually inside the container.

### Jetson AGX Orin

- Flash `JetPack 6.x` system image
- Default environment:

| CUDA | cuDNN | TensorRT | OpenCV |
| ---- | ----- | -------- | ------ |
| 11.4 | 8.6   | 8.5.2    | 4.5.4  |

## Run

The repo is organized by model family first, then by runtime:

```text
TensorRT-YOLO/
├── yolo11/
│   ├── cpp/
│   ├── python/
│   ├── assets/
│   └── docs/
├── yolo26/
│   ├── cpp/
│   ├── python/
│   ├── assets/
│   └── docs/
├── shared/
│   ├── cpp/
│   └── python/
└── assets/
```

- [YOLO11 docs](yolo11/docs/README.md)
- [YOLO11 cpp detect](yolo11/cpp/detect)
- [YOLO11 cpp pose](yolo11/cpp/pose)
- [YOLO11 cpp segment](yolo11/cpp/segment)
- [YOLO11 cpp track](yolo11/cpp/track)
- [YOLO11 python detect](yolo11/python/detect)
- [YOLO11 python pose](yolo11/python/pose)
- [YOLO11 python segment](yolo11/python/segment)
- [YOLO11 python track](yolo11/python/track)
- [YOLO26 scaffold](yolo26/docs/README.md)

## Credits

Original project: [emptysoal/TensorRT-YOLO11](https://github.com/emptysoal/TensorRT-YOLO11)
