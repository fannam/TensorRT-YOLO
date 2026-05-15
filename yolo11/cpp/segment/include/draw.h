#ifndef DRAW_H
#define DRAW_H

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// Overlay the resized binary mask on top of the input BGR image.
void draw_mask(cv::Mat& img, float* mask);

#endif  // DRAW_H
