#ifndef TYPES_H
#define TYPES_H

#include <string>

// Minimal detection representation after:
// 1) decoding head output into xyxy boxes,
// 2) running NMS on the GPU,
// 3) scaling from 640x640 letterbox space back to the original image.
struct Detection
{
    float bbox[4];
    float conf;
    int classId;
};

#endif  // TYPES_H
