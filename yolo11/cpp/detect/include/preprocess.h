#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// Write the normalized NCHW float32 input tensor directly onto the GPU:
// letterbox -> bilinear resize -> BGR->RGB -> HWC->CHW -> normalize [0,1].
void preprocess(const cv::Mat& srcImg, float* dstDevData, const int dstHeight, const int dstWidth, cudaStream_t stream);

#endif  // PREPROCESS_H
