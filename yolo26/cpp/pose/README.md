# YOLO26 cpp Pose

1. Put the exported ONNX model under `onnx_model/`, for example `yolo26m-pose.onnx`.
2. Build and run:

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
./pose ../images yolo26m-pose
```

The implementation supports both raw pose outputs like `1x56x8400` and end-to-end top-k outputs like `1x300x56` or `1x300x57`.
