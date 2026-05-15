#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <vector>


const int kGpuId = 0;
const int kNumClass = 80;
const int kInputH = 640;
const int kInputW = 640;
const float kNmsThresh = 0.45f;
const float kConfThresh = 0.25f;
const int kMaxNumOutputBbox = 1000;  // assume the box outputs no more than kMaxNumOutputBbox boxes that conf >= kNmsThresh;
const int kNumBoxElement = 7;  // left, top, right, bottom, confidence, class, keepflag(whether drop when NMS)

// const std::string trtFile = "./yolo26m.plan";
// const std::string testDataDir = "../images";

// INT8 remains compile-time only because the calibration pipeline is not exposed through the CLI.
const bool bINT8Mode = false;
const std::string cacheFile = "./int8.cache";
const std::string calibrationDataPath = "../calibrator";  // Directory containing images used for INT8 calibration

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
