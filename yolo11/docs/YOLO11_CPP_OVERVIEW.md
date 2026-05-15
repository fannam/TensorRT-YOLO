# YOLO11 C++ Overview

Tài liệu này giải thích phần C++ của `yolo11/cpp/*` và cách nó ghép với `shared/cpp/bytetrack`.

## 1. Bản đồ module

- `yolo11/cpp/detect`
  Binary detect ảnh tĩnh. Một output tensor chính từ head detect.
- `yolo11/cpp/pose`
  Cùng khung detect nhưng output mỗi candidate còn mang thêm 17 keypoint.
- `yolo11/cpp/segment`
  Có hai output tensor: detect head và proto head để tái tạo mask.
- `yolo11/cpp/track`
  Không có head tracking riêng. Module này ghép detector `detect` với `BYTETracker`.
- `shared/cpp/bytetrack`
  Chứa state machine tracking, Kalman filter, IoU cost và solver assignment.

## 2. Luồng dữ liệu end-to-end

### Detect

1. `main.cpp` resolve đường dẫn ONNX và `.plan`.
2. `YoloDetector` trong `src/infer.cpp`:
   - nạp `.plan` nếu có
   - nếu chưa có thì build engine từ ONNX rồi serialize lại
3. `src/preprocess.cu`
   - letterbox
   - bilinear resize
   - BGR -> RGB
   - HWC -> CHW
   - normalize về `[0, 1]`
4. TensorRT enqueue trên cùng CUDA stream.
5. `src/postprocess.cu`
   - transpose `[C, N] -> [N, C]`
   - decode bbox/class
   - NMS trên GPU
6. `scale_bbox()` đưa bbox từ hệ 640x640 có padding về ảnh gốc.

### Pose

Luồng giống detect, nhưng bước decode giữ thêm `17 * 3 = 51` giá trị keypoint.

Sau khi copy về host:

- `scale_bbox()` scale bbox về ảnh gốc
- `scale_kpt_coords()` scale từng keypoint về ảnh gốc
- `draw_image()` có thể vẽ bbox, keypoint và skeleton COCO

### Segment

Luồng khác ở hai điểm:

1. TensorRT có hai output:
   - `proto`: thường là `[1, 32, 160, 160]`
   - `detect`: thường là `[1, 116, 8400]`
2. Sau decode + NMS, `process_mask()`:
   - gom `n` vector coefficient dài 32 của `n` detection
   - nhân với proto để tạo `n` mask ở độ phân giải `160x160`
   - crop theo bbox
   - cắt bỏ vùng padding do letterbox
   - resize về đúng kích thước ảnh gốc

## 3. Tensor shape map

### Detect

- Input: `[1, 3, 640, 640]`
- Raw output: `[1, 84, 8400]`
- Sau transpose: `[8400, 84]`
- Sau decode: `[count, boxes...]`, mỗi box có 7 float

`84 = 4 bbox + 80 class`

### Pose

- Input: `[1, 3, 640, 640]`
- Raw output: `[1, 56, 8400]`
- Sau transpose: `[8400, 56]`
- Sau decode: mỗi box có `7 + 51 = 58` float

`56 = 4 bbox + 1 class + 51 keypoint values`

### Segment

- Input: `[1, 3, 640, 640]`
- Proto output: `[1, 32, 160, 160]`
- Detect output: `[1, 116, 8400]`
- Sau transpose detect: `[8400, 116]`
- Sau decode: mỗi box có `7 + 32 = 39` float

`116 = 4 bbox + 80 class + 32 mask coefficient`

## 4. Memory map

### Resource sống lâu

Mỗi `YoloDetector` giữ:

- `runtime`
- `engine`
- `context`
- `cudaStream_t stream`
- `vBufferD`: các binding TensorRT
- `transposeDevice`
- `decodeDevice`
- `outputData` trên host

### Detect/Pose

- `vBufferD[ input ]`: tensor input float32
- `vBufferD[ output ]`: raw output tensor
- `transposeDevice`: scratch buffer cho `[C, N] -> [N, C]`
- `decodeDevice`: buffer phẳng sau decode/NMS

### Segment

Có thêm:

- `vBufferD[ proto ]`: proto tensor từ head mask
- vùng tạm trong `process_mask()`:
  - `maskCoefDevice`
  - `maskDevice`
  - `bboxDevice`
  - `cutMaskDevice`
  - `scaledMaskDevice`

Các buffer tạm này được cấp phát trong `process_mask()` rồi giải phóng ngay sau khi copy mask về host.

## 5. TRT8 và TRT10 khác nhau ở đâu

Repo đang giữ hai nhánh tương thích:

- TRT10:
  - enumerate tensor bằng `getNbIOTensors()`
  - set binding bằng tên tensor
  - enqueue bằng `enqueueV3()`
- TRT8/9:
  - enumerate binding bằng `getNbBindings()`
  - set shape bằng binding index
  - enqueue bằng `enqueueV2()`

Lý do comment phần này kỹ trong code là vì shape/binding lookup là chỗ người mới rất hay nhầm khi nâng version TensorRT.

## 6. FP16 và INT8

- `bFP16Mode`
  Bật flag FP16 cho builder. Không đổi luồng runtime.
- `bINT8Mode`
  Bật calibration path:
  - đọc ảnh trong `../calibrator`
  - preprocess trên CPU
  - ghi hoặc đọc `int8.cache`

INT8 chỉ tác động ở lúc build engine, không tác động vào vòng inference thường ngày nếu `.plan` đã tồn tại.

## 7. Detect vs Pose vs Segment

- `detect`
  Nhẹ nhất, chỉ cần bbox/class.
- `pose`
  Cùng detect nhưng mỗi candidate thêm 51 số keypoint.
- `segment`
  Tốn hậu xử lý nhất vì phải tái tạo mask từ `proto + coefficient`.

Nếu cần hiểu pattern chung của cả ba task, nên đọc `detect` trước rồi mới đọc `pose` và `segment`.

## 8. Tracker ăn dữ liệu gì từ detector

`yolo11/cpp/track/main.cpp` lấy output detect và đổi sang:

- `cv::Rect_<float> rect` theo `tlwh`
- `label`
- `prob`

`BYTETracker.update()` không dùng feature embedding. Nó chỉ cần:

- bbox
- score
- state trước đó
- Kalman predict
- cost matrix từ IoU

## 9. Đọc repo theo thứ tự nào

Khuyến nghị cho người mới:

1. `yolo11/cpp/detect/main.cpp`
2. `yolo11/cpp/detect/include/infer.h`
3. `yolo11/cpp/detect/src/infer.cpp`
4. `yolo11/cpp/detect/src/preprocess.cu`
5. `yolo11/cpp/detect/src/postprocess.cu`
6. `yolo11/cpp/pose/src/infer.cpp`
7. `yolo11/cpp/segment/src/infer.cpp`
8. `yolo11/cpp/track/main.cpp`
9. `shared/cpp/bytetrack/include/STrack.h`
10. `shared/cpp/bytetrack/src/BYTETracker.cpp`

## 10. Ghi chú về `yolo26`

`yolo26` hiện chỉ nên xem như scaffold/tương đồng layout.

- Đừng dùng nó làm nguồn chân lý để hiểu chi tiết runtime C++.
- Khi cần hiểu implementation thật, ưu tiên `yolo11`.
