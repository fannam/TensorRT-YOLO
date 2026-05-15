#ifndef TYPES_H
#define TYPES_H

#include <string>
#include "config.h"

// Pose detection carries both bbox and keypoint data:
// - kpts: raw data still in 640x640 input coordinates
// - vKpts: data already scaled back to the original image for immediate drawing/tracking use
struct Detection
{
    float bbox[4];
    float conf;
    int classId;
    float kpts[kNumKpt * kKptDims];
    std::vector<std::vector<float>> vKpts;
};

#endif  // TYPES_H
