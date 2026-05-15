#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <vector>

// Pose kế thừa hầu hết khung detect nhưng output chỉ có 1 class ("person")
// và bổ sung thêm 17 keypoint, mỗi keypoint gồm 3 giá trị [x, y, conf].
const int kGpuId = 0;
const int kNumClass = 1;
const int kNumKpt = 17;
const int kKptDims = 3;
const int kInputH = 640;
const int kInputW = 640;
const float kNmsThresh = 0.45f;
const float kConfThresh = 0.25f;
const int kMaxNumOutputBbox = 1000;
// 7 phần tử detect cơ bản + 17 * 3 giá trị keypoint.
const int kNumBoxElement = 7 + kNumKpt * kKptDims;

const std::string onnxFile = "../onnx_model/yolo11s-pose.onnx";

const bool bFP16Mode = false;
const bool bINT8Mode = false;
const std::string cacheFile = "./int8.cache";
const std::string calibrationDataPath = "../calibrator";

const std::vector<std::string> vClassNames {"person"};

// Skeleton theo chuẩn COCO pose. Chỉ dùng ở bước render, không ảnh hưởng suy luận.
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
