# YOLO26 cpp Segment

1. Put the exported ONNX model under `onnx_model/`, for example `yolo26m-seg.onnx`.
2. Build and run:

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
./segment ../images yolo26m-seg
```

If `yolo26m-seg.plan` does not exist, TensorRT will build it from ONNX first.
