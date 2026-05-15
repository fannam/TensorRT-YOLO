#ifndef TYPES_H
#define TYPES_H

#include <string>
#include <vector>

// Segment data after decode:
// - mask: the 32 coefficients associated with each detection
// - maskMatrix: the mask after proto reconstruction, bbox crop, and resize back to the original image
struct Detection
{
    float bbox[4];
    float conf;
    int classId;
    float mask[32];
    std::vector<float> maskMatrix;
};

#endif  // TYPES_H
