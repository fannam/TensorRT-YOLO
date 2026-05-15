#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <cmath>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "config.h"

void transpose(float* src, float* dst, int numBboxes, int numElements, cudaStream_t stream);

// Decode bbox/class while preserving the 32 mask coefficients for each candidate.
void decode(float* src, float* dst, int numBboxes, int numClasses, int numMasks, float confThresh, int maxObjects, int numBoxElement, cudaStream_t stream);

void nms(float* data, float kNmsThresh, int maxObjects, int numBoxElement, cudaStream_t stream);

// Multiply the [n, 32] matrix by the proto tensor [32, H*W] to reconstruct masks as [n, H*W].
void matrix_multiply(float* aMatrix, int aRows, int aCols, float* bMatrix, int bRows, int bCols, float* cMatrix, cudaStream_t stream, bool sigm = false);

void downsample_bbox(float* bboxDevice, int length, float heightRatio, float widthRatio, cudaStream_t stream);

// Remove mask regions outside the corresponding bbox to reduce noise before resizing back to the original image.
void crop_mask(float* masksDevice, int maskNum, int maskHeight, int maskWidth, float* bboxesDevice, cudaStream_t stream);

void cut_mask(
    float* masksDevice, int maskNum, int maskHeight, int maskWidth,
    float* cutMasksDevice, int cutMaskTop, int cutMaskLeft, int cutMaskH, int cutMaskW, cudaStream_t stream
);

void resize(float* masksDevice, int maskNum, int maskHeight, int maskWidth, float* dstMasksDevice, int dstMaskH, int dstMaskW, cudaStream_t stream);

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

#endif  // POSTPROCESS_H
