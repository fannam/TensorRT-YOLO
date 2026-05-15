#ifndef TYPES_H
#define TYPES_H

#include <string>
#include "config.h"

// Detection của pose mang cả bbox lẫn keypoint:
// - kpts: dữ liệu thô còn ở hệ toạ độ input 640x640
// - vKpts: dữ liệu đã scale về ảnh gốc để draw/tracking tiếp theo dùng ngay
struct Detection
{
    float bbox[4];
    float conf;
    int classId;
    float kpts[kNumKpt * kKptDims];
    std::vector<std::vector<float>> vKpts;
};

#endif  // TYPES_H
