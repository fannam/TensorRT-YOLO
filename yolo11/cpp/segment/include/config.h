#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <vector>

// Segment dùng cùng head detect cho bbox/class nhưng bổ sung 32 hệ số mask coefficient
// và một tensor proto riêng để tái tạo mask nhị phân cho từng detection.
const int kGpuId = 0;
const int kNumClass = 80;
const int kInputH = 640;
const int kInputW = 640;
const float kNmsThresh = 0.45f;
const float kConfThresh = 0.25f;
const int kMaxNumOutputBbox = 1000;
// 7 phần tử detect cơ bản + 32 hệ số mask coefficient.
const int kNumBoxElement = 7 + 32;

const bool bFP16Mode = true;
const bool bINT8Mode = false;
const std::string cacheFile = "./int8.cache";
const std::string calibrationDataPath = "../calibrator";

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
