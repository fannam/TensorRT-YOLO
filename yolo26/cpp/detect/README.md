# YOLO26 cpp Detect

1. Put the exported ONNX model under `onnx_model/`, for example `yolo26m.onnx`.
2. Build and run:

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
./detect ../images yolo26m
```

The implementation supports both raw detector outputs like `1x84x8400` and end-to-end top-k outputs like `1x300x6`.
