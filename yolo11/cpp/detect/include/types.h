#ifndef TYPES_H
#define TYPES_H

#include <string>

// Dạng detection tối giản sau khi đã:
// 1) decode head output về bbox xyxy,
// 2) chạy NMS trên GPU,
// 3) scale từ không gian letterbox 640x640 về ảnh gốc.
struct Detection
{
    float bbox[4];
    float conf;
    int classId;
};

#endif  // TYPES_H
