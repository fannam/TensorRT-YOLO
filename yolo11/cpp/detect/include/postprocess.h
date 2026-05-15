#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "config.h"

// Đổi layout head output từ [C, N] sang [N, C] để mỗi thread decode xử lý trọn một candidate.
void transpose(float* src, float* dst, int numBboxes, int numElements, cudaStream_t stream);

// Decode head output thành mảng phẳng có layout:
// [count, box0..., box1..., ...], mỗi box có kNumBoxElement float.
void decode(float* src, float* dst, int numBboxes, int numClasses, float confThresh, int maxObjects, int numBoxElement, cudaStream_t stream);

// NMS cùng class chạy trên GPU để tránh copy toàn bộ 8400 candidate về CPU rồi mới lọc.
void nms(float* data, float kNmsThresh, int maxObjects, int numBoxElement, cudaStream_t stream);

__inline__ void scale_bbox(cv::Mat& img, float bbox[4]){
    // Đảo ngược chính xác bước letterbox của preprocess: bỏ padding trước rồi chia theo scale.
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
