# YOLO11 Docs

`yolo11/` is the fully implemented model family in the current repository.

- `cpp/detect`, `cpp/pose`, `cpp/segment`, `cpp/track`: cpp runtime tasks
- `python/detect`, `python/pose`, `python/segment`, `python/track`: Python runtime tasks
- `assets/`: YOLO11-specific media and sample inputs

Notes:

- Tracking resolves its detector from the same model family instead of the old root-level layout
- ByteTrack now lives under `shared/cpp/bytetrack` and `shared/python/tracker`
