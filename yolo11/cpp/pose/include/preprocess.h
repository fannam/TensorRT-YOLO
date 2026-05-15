#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// Pose uses the same preprocess as detect: letterbox -> RGB -> CHW -> normalize.
void preprocess(const cv::Mat& srcImg, float* dstDevData, const int dstHeight, const int dstWidth, cudaStream_t stream);

#endif  // PREPROCESS_H
