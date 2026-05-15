#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "config.h"

// Đổi head output từ [56, 8400] sang [8400, 56] để mỗi candidate nằm liền mạch.
void transpose(float* src, float* dst, int numBboxes, int numElements, cudaStream_t stream);

// Decode bbox + class + toàn bộ vector keypoint của mỗi candidate.
void decode(float* src, float* dst, int numBboxes, int numClasses, int numKpts, float confThresh, int maxObjects, int numBoxElement, cudaStream_t stream);

void nms(float* data, float kNmsThresh, int maxObjects, int numBoxElement, cudaStream_t stream);

__inline__ void scale_bbox(cv::Mat& img, float bbox[4]){
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

__inline__ std::vector<std::vector<float>> scale_kpt_coords(cv::Mat& img, float* pkpt){
    // Keypoint được decode trong không gian letterbox 640x640, nên phải đảo padding/scale
    // giống bbox trước khi vẽ hoặc tiêu thụ tiếp.
    float r_w = kInputW / (img.cols * 1.0);
    float r_h = kInputH / (img.rows * 1.0);
    float r = std::min(r_w, r_h);
    float pad_h = (kInputH - r * img.rows) / 2;
    float pad_w = (kInputW - r * img.cols) / 2;

    std::vector<std::vector<float>> vScaledKpts;
    float x;
    float y;
    float conf;
    float* pSingleKpt;
    for (int i = 0; i < kNumKpt; i++){
        pSingleKpt = pkpt + i * kKptDims;
        x = pSingleKpt[0] - pad_w;
        y = pSingleKpt[1] - pad_h;
        x = x / r;
        y = y / r;
        conf = pSingleKpt[2];
        std::vector<float> scaledKpt {x, y, conf};
        vScaledKpts.push_back(scaledKpt);
    }

    return vScaledKpts;
}

#endif  // POSTPROCESS_H
