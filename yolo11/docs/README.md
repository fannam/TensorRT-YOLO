# YOLO11 Docs

`yolo11/` là model family đã có implementation đầy đủ trong repo hiện tại.

- `cpp/detect`, `cpp/pose`, `cpp/segment`, `cpp/track`: runtime cpp theo layout mới
- `python/detect`, `python/pose`, `python/segment`, `python/track`: runtime Python theo layout mới
- `assets/`: media và input dùng riêng cho YOLO11

Điểm đáng chú ý:

- Tracking dùng detector từ cùng model family, không còn phụ thuộc path ở root layout cũ
- ByteTrack đã được tách sang `shared/cpp/bytetrack` và `shared/python/tracker`
- Tài liệu giải thích C++ cho `YOLO11 + shared`:
  - `YOLO11_CPP_OVERVIEW.md`: bản đồ module, luồng dữ liệu, tensor shape, memory map
  - `YOLO11_CPP_BYTETRACK.md`: state machine ByteTrack, Kalman, IoU, assignment flow
