# YOLO11 Python ByteTrack

![street](../../../assets/result.gif)

## Run

1. Finish the detector conversion in [../detect/README-en.md](../detect/README-en.md) so `yolo11/python/detect/model.plan` exists.
2. Install the tracking extras in the same Python environment:

```bash
pip install lap
pip install cython_bbox
```

3. Run tracking from this directory:

```bash
python main.py --video ../../assets/street.mp4
```

- The default detector engine is `../detect/model.plan`
- The output video is written to `./output/result.mp4`
