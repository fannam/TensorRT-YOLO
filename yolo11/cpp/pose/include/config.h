#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <vector>

// Pose reuses most of the detect structure, but its output has only one class ("person")
// and adds 17 keypoints, each represented by 3 values [x, y, conf].
const int kGpuId = 0;
const int kNumClass = 1;
const int kNumKpt = 17;
const int kKptDims = 3;
const int kInputH = 640;
const int kInputW = 640;
const float kNmsThresh = 0.45f;
const float kConfThresh = 0.25f;
const int kMaxNumOutputBbox = 1000;
// 7 base detect elements + 17 * 3 keypoint values.
const int kNumBoxElement = 7 + kNumKpt * kKptDims;

// INT8 remains compile-time only because the calibration pipeline is not exposed through the CLI.
const bool bINT8Mode = false;
const std::string cacheFile = "./int8.cache";
const std::string calibrationDataPath = "../calibrator";

const std::vector<std::string> vClassNames {"person"};

// COCO pose skeleton. Used only during rendering and does not affect inference.
const std::vector<std::vector<int>> skeleton {
    {16, 14},
    {14, 12},
    {17, 15},
    {15, 13},
    {12, 13},
    {6, 12},
    {7, 13},
    {6, 7},
    {6, 8},
    {7, 9},
    {8, 10},
    {9, 11},
    {2, 3},
    {1, 2},
    {1, 3},
    {2, 4},
    {3, 5},
    {4, 6},
    {5, 7}
};

#endif  // CONFIG_H
