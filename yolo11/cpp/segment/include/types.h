#ifndef TYPES_H
#define TYPES_H

#include <string>
#include <vector>

// Dữ liệu segment sau decode:
// - mask: 32 hệ số coefficient đi kèm mỗi detection
// - maskMatrix: mask đã được ghép với proto, crop theo bbox và resize về ảnh gốc
struct Detection
{
    float bbox[4];
    float conf;
    int classId;
    float mask[32];
    std::vector<float> maskMatrix;
};

#endif  // TYPES_H
