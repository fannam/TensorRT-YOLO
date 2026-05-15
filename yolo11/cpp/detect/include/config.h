#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <vector>

// Cấu hình compile-time cho binary detect.
// Các hằng số này đồng thời chi phối shape TensorRT, preprocess CUDA và bước scale
// output về ảnh gốc, nên phải được hiểu như "hợp đồng chung" của cả pipeline.
const int kGpuId = 0;
const int kNumClass = 80;
const int kInputH = 640;
const int kInputW = 640;
const float kNmsThresh = 0.45f;
const float kConfThresh = 0.25f;

// Buffer decode trên GPU/CPU được cấp phát cố định cho tối đa 1000 box hợp lệ.
// Nếu cần nhiều hơn, phải sửa đồng bộ layout decode/NMS và host buffer.
const int kMaxNumOutputBbox = 1000;
// Mỗi box sau decode chiếm 7 float: [x1, y1, x2, y2, conf, class_id, keep_flag].
const int kNumBoxElement = 7;

// Builder flag cho TensorRT. Repo giữ ở dạng hằng số để sample dễ đọc hơn CLI động.
const bool bFP16Mode = false;
const bool bINT8Mode = false;
const std::string cacheFile = "./int8.cache";
// Thư mục ảnh calibration chỉ được dùng khi bINT8Mode=true.
const std::string calibrationDataPath = "../calibrator";

// Bản đồ class COCO để bước vẽ và log có thể chuyển class_id sang tên có nghĩa.
const std::vector<std::string> vClassNames {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

#endif  // CONFIG_H
