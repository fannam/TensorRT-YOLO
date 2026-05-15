# TensorRT-YOLO

## Introduction

- Deploy detect, pose, segment, and tracking for `YOLO11`;
- Keep a model-first layout so `YOLO11` and future families such as `YOLO26` can coexist cleanly;

- Support `Jetson` series, also `Linux x86_64`;
- This project does not need to compile and install `CUDA-supported OpenCV`, all tensor operations related to pre and post processing are implemented by cuda programming;
- Mode convert: `.pth` -> `.onnx` -> `.plan(.engine)`;

- I use `Python` and `cpp` APIs to do the implementation;
- All of them adopt object-oriented, which is easy to combine with other projects;
- The `C++` version will also be compiled as a dynamic link library, which is easy to call as an interface in other projects; 

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

- The inference time here includes pre-processing, model inference, and post-processing
- The inference time here base on `x86_64 Linux ` ，`Ubuntu`，GPU is `GeForce RTX 2080 Ti`

## Environment

1. Base requirements：

- `TensorRT 8.0+`
- `OpenCV 3.4.0+`

**If the basic requirements are met, you can directly go to each directory and run each task**

**Environment construction can refer to the following:**

2. If  `Linux x86_64`, `docker` is recommended 

```bash
docker pull nvcr.io/nvidia/tensorrt:22.04-py3
```

- This docker image contains：

| CUDA   | cuDNN    | TensorRT | python |
| ------ | -------- | -------- | ------ |
| 11.6.2 | 8.4.0.27 | 8.2.4.2  | 3.8.10 |

- Then install opencv yourself 

3. If `Jetson`, such as `Jetson Nano`

- The burned system image is `Jetpack 4.6.1`，original environment is as follows：

| CUDA | cuDNN | TensorRT | OpenCV |
| ---- | ----- | -------- | ------ |
| 10.2 | 8.2   | 8.2.1    | 4.1.1  |

## Run

The repository is now organized by model family first and runtime second.

- [YOLO11 docs](yolo11/docs/README-en.md)
- [YOLO11 cpp detect](yolo11/cpp/detect)
- [YOLO11 cpp pose](yolo11/cpp/pose)
- [YOLO11 cpp segment](yolo11/cpp/segment)
- [YOLO11 cpp track](yolo11/cpp/track)
- [YOLO11 python detect](yolo11/python/detect)
- [YOLO11 python pose](yolo11/python/pose)
- [YOLO11 python segment](yolo11/python/segment)
- [YOLO11 python track](yolo11/python/track)
- [YOLO26 scaffold](yolo26/docs/README-en.md)
