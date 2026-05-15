# YOLO11 C++ ByteTrack Appendix

Tài liệu này chỉ tập trung vào `shared/cpp/bytetrack` và cách `yolo11/cpp/track` sử dụng nó.

## 1. Các abstraction chính

- `Object`
  Detection đầu vào cho tracker: `rect + label + prob`
- `STrack`
  Một track có trạng thái và lịch sử
- `TrackState`
  `New`, `Tracked`, `Lost`, `Removed`
- `BYTETracker`
  Điều phối toàn bộ pipeline update theo từng frame
- `KalmanFilter`
  Dự đoán và cập nhật state động học bbox
- `lapjv`
  Solver assignment tối ưu cho cost matrix

## 2. Ba hệ toạ độ bbox cần nhớ

- `tlwh`
  `[top-left x, top-left y, width, height]`
- `tlbr`
  `[x1, y1, x2, y2]`
- `xyah`
  `[center_x, center_y, aspect_ratio, height]`

ByteTrack dùng:

- `tlwh` để giao tiếp gần detector và draw
- `tlbr` để tính IoU
- `xyah` làm measurement/state cho Kalman

## 3. State vector Kalman

State 8 chiều:

`[x, y, a, h, vx, vy, va, vh]`

Trong đó:

- `x, y`
  tâm bbox
- `a`
  aspect ratio = `w / h`
- `h`
  chiều cao bbox
- `vx, vy, va, vh`
  vận tốc tương ứng

Lý do dùng `xyah` thay vì `xywh` là aspect ratio ổn định hơn width tuyệt đối trong nhiều cảnh tracking.

## 4. State machine update của ByteTrack

Trong `BYTETracker::update()` có thể đọc theo 5 bước:

### Bước 1. Tách detection điểm cao và điểm thấp

- `score >= track_thresh`
  vào `detections`
- còn lại
  vào `detections_low`

Ý tưởng cốt lõi của ByteTrack là không bỏ hẳn low-score detections.

### Bước 2. Ghép lần 1 với `tracked + lost`

- gộp `tracked_stracks` và `lost_stracks` thành `strack_pool`
- chạy `STrack::multi_predict()` để Kalman dự đoán vị trí frame hiện tại
- tính `iou_distance()`
- giải assignment bằng `lapjv()`

Kết quả:

- matched track được `update()` hoặc `re_activate()`
- unmatched track đi tiếp sang bước sau

### Bước 3. Ghép lần 2 với low-score detections

Chỉ những track đang `Tracked` nhưng trượt lượt 1 mới tham gia.

Mục đích:

- giữ continuity cho object bị detector chấm điểm thấp trong vài frame
- giảm ID switch

### Bước 4. Xử lý unconfirmed và track mới

- `unconfirmed`
  là track mới xuất hiện quá ít frame
- nếu unconfirmed không ghép lại được
  -> `Removed`
- detection còn dư nhưng đủ `high_thresh`
  -> `activate()` thành track mới

### Bước 5. Dọn pool trạng thái

- lost quá `max_time_lost`
  -> `Removed`
- merge lại `tracked`, `lost`, `removed`
- loại track trùng lặp bằng `remove_duplicate_stracks()`

## 5. Kalman predict/update diễn ra thế nào

### `initiate()`

Tạo track mới:

- measurement từ detector đổi sang `xyah`
- vận tốc khởi tạo bằng 0
- covariance khởi tạo theo kích thước bbox

### `predict()`

Đẩy state sang frame kế tiếp:

- `x += vx`
- `y += vy`
- `a += va`
- `h += vh`

đồng thời cộng motion noise.

### `project()`

Chiếu state 8D về measurement space 4D để so sánh với detection mới.

### `update()`

Dùng measurement detector sửa lại state dự đoán:

- innovation = detection - prediction
- Kalman gain quyết định "tin detector bao nhiêu"

## 6. IoU và assignment flow

`iou_distance()` làm:

1. chuyển track sang `tlbr`
2. tính IoU matrix
3. đổi sang cost matrix bằng `1 - IoU`

`linear_assignment()` làm:

1. gọi `lapjv()`
2. thu về:
   - `matches`
   - `unmatched_a`
   - `unmatched_b`

`cost_limit` chính là ngưỡng để một cặp còn được chấp nhận.

## 7. Ý nghĩa một số trường trong `STrack`

- `is_activated`
  track đã đủ điều kiện xuất ra ngoài
- `tracklet_len`
  số frame liên tiếp track được update kể từ lần activate/re-activate gần nhất
- `frame_id`
  frame cuối cùng track được nhìn thấy hoặc dự đoán tới
- `start_frame`
  frame track được sinh ra
- `track_id`
  ID ổn định dùng để vẽ lên video output

## 8. `track/main.cpp` ghép detector với tracker thế nào

1. chạy `YoloDetector::inference(img)`
2. lọc class theo `trackClasses`
3. đổi bbox `xyxy` sang `tlwh`
4. tạo `Object`
5. gọi `tracker.update(objects)`
6. draw `track_id` và bbox sau tracking

Điểm quan trọng:

- tracker tiêu thụ detection đã scale về ảnh gốc
- tracker không biết gì về TensorRT, CUDA hay head YOLO
- coupling giữa detector và tracker chỉ là `bbox + score + label`
