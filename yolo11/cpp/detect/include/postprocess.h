#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "config.h"

// Convert head output layout from [C, N] to [N, C] so each decode thread handles one full candidate.
void transpose(float* src, float* dst, int numBboxes, int numElements, cudaStream_t stream);

// Decode head output into a flat array with layout:
// [count, box0..., box1..., ...], with kNumBoxElement floats per box.
void decode(float* src, float* dst, int numBboxes, int numClasses, float confThresh, int maxObjects, int numBoxElement, cudaStream_t stream);

// Per-class NMS runs on the GPU so the code does not need to copy all 8400 candidates back to the CPU before filtering.
void nms(float* data, float kNmsThresh, int maxObjects, int numBoxElement, cudaStream_t stream);

__inline__ void scale_bbox(cv::Mat& img, float bbox[4]){
    // Precisely invert preprocess letterboxing: remove padding first, then divide by scale.
    float r_w = kInputW / (img.cols * 1.0);
    float r_h = kInputH / (img.rows * 1.0);
    float r = std::min(r_w, r_h);
    float pad_h = (kInputH - r * img.rows) / 2;
    float pad_w = (kInputW - r * img.cols) / 2;

    bbox[0] = (bbox[0] - pad_w) / r;
    bbox[1] = (bbox[1] - pad_h) / r;
    bbox[2] = (bbox[2] - pad_w) / r;
    bbox[3] = (bbox[3] - pad_h) / r;
}

#endif  // POSTPROCESS_H
