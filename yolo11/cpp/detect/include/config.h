#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <vector>

// Compile-time configuration for the detect binary.
// These constants jointly define TensorRT shape, CUDA preprocess, and the scaling step
// back to the original image, so they should be treated as the shared contract of the pipeline.
const int kGpuId = 0;
const int kNumClass = 80;
const int kInputH = 640;
const int kInputW = 640;
const float kNmsThresh = 0.45f;
const float kConfThresh = 0.25f;

// GPU/CPU decode buffers are fixed-size for up to 1000 valid boxes.
// If you need more, update the decode/NMS layout and host buffer together.
const int kMaxNumOutputBbox = 1000;
// Each decoded box uses 7 floats: [x1, y1, x2, y2, conf, class_id, keep_flag].
const int kNumBoxElement = 7;

// INT8 remains compile-time only because the calibration pipeline is not exposed through the CLI.
const bool bINT8Mode = false;
const std::string cacheFile = "./int8.cache";
// The calibration image directory is only used when bINT8Mode=true.
const std::string calibrationDataPath = "../calibrator";

// COCO class names let drawing and logging map class_id to readable labels.
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
