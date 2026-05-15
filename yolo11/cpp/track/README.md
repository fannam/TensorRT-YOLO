# YOLO11 cpp ByteTrack

![street](../../../assets/result.gif)

## Run

1. Finish detector conversion in [../detect/README-en.md](../detect/README-en.md) so `yolo11/cpp/detect/build/yolo11s.plan` exists.
2. Install the extra dependency:

```bash
apt install libeigen3-dev
```

3. Build and run from this directory:

```bash
mkdir -p build
cd build
cmake ..
make
./trt_yolo11_track_cli ../../assets/street.mp4
```

- The detector engine is resolved automatically from the sibling `detect` task
- The output video is written to `../output/result.mp4`
