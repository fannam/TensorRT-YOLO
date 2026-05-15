#include "preprocess.h"
#include "public.h"

// This file implements the entire preprocess on the GPU to avoid host round-trips:
// letterbox + bilinear resize + color conversion + layout conversion + normalization.

__global__ void letterbox(const uchar* srcData, const int srcH, const int srcW, uchar* tgtData,
    const int tgtH, const int tgtW, const int rszH, const int rszW, const int startY, const int startX)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = ix + iy * tgtW;
    int idx3 = idx * 3;

    if ( ix >= tgtW || iy >= tgtH ) return;
    // Pixels outside the resized image region are filled with the constant value 114.
    if ( iy < startY || iy > (startY + rszH - 1) ) {
        tgtData[idx3] = 114;
        tgtData[idx3 + 1] = 114;
        tgtData[idx3 + 2] = 114;
        return;
    }
    if ( ix < startX || ix > (startX + rszW - 1) ){
        tgtData[idx3] = 114;
        tgtData[idx3 + 1] = 114;
        tgtData[idx3 + 2] = 114;
        return;
    }

    float scaleY = (float)rszH / (float)srcH;
    float scaleX = (float)rszW / (float)srcW;

    // Each destination pixel is traced back to source-image coordinates and bilinearly interpolated.
    // The +0.5/-0.5 formula keeps pixel centers better aligned during scaling.
    float beforeX = float(ix - startX + 0.5) / scaleX - 0.5;
    float beforeY = float(iy - startY + 0.5) / scaleY - 0.5;
    int topY = static_cast<int>(beforeY);
    int bottomY = topY + 1;
    int leftX = static_cast<int>(beforeX);
    int rightX = leftX + 1;
    float u = beforeX - leftX;
    float v = beforeY - topY;

    if (topY >= srcH - 1 && leftX >= srcW - 1)
    {
        for (int k = 0; k < 3; k++)
        {
            tgtData[idx3 + k] = (1. - u) * (1. - v) * srcData[(leftX + topY * srcW) * 3 + k];
        }
    }
    else if (topY >= srcH - 1)
    {
        for (int k = 0; k < 3; k++)
        {
            tgtData[idx3 + k]
            = (1. - u) * (1. - v) * srcData[(leftX + topY * srcW) * 3 + k]
            + (u) * (1. - v) * srcData[(rightX + topY * srcW) * 3 + k];
        }
    }
    else if (leftX >= srcW - 1)
    {
        for (int k = 0; k < 3; k++)
        {
            tgtData[idx3 + k]
            = (1. - u) * (1. - v) * srcData[(leftX + topY * srcW) * 3 + k]
            + (1. - u) * (v) * srcData[(leftX + bottomY * srcW) * 3 + k];
        }
    }
    else
    {
        for (int k = 0; k < 3; k++)
        {
            tgtData[idx3 + k]
            = (1. - u) * (1. - v) * srcData[(leftX + topY * srcW) * 3 + k]
            + (u) * (1. - v) * srcData[(rightX + topY * srcW) * 3 + k]
            + (1. - u) * (v) * srcData[(leftX + bottomY * srcW) * 3 + k]
            + u * v * srcData[(rightX + bottomY * srcW) * 3 + k];
        }
    }
}

__global__ void process(const uchar* srcData, float* tgtData, const int h, const int w)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = ix + iy * w;
    int idx3 = idx * 3;

    if (ix < w && iy < h)
    {
        // TensorRT expects NCHW float32 in RGB order, while OpenCV input arrives as HWC BGR uchar.
        tgtData[idx] = (float)srcData[idx3 + 2] / 255.0;
        tgtData[idx + h * w] = (float)srcData[idx3 + 1] / 255.0;
        tgtData[idx + h * w * 2] = (float)srcData[idx3] / 255.0;
    }
}

void preprocess(const cv::Mat& srcImg, float* dstDevData, const int dstHeight, const int dstWidth, cudaStream_t stream)
{
    int srcHeight = srcImg.rows;
    int srcWidth = srcImg.cols;
    int srcElements = srcHeight * srcWidth * 3;
    int dstElements = dstHeight * dstWidth * 3;

    // midDevData holds the intermediate uchar letterboxed image so the process kernel can read it contiguously.
    uchar* midDevData;
    CHECK(cudaMalloc((void**)&midDevData, sizeof(uchar) * dstElements));
    uchar* srcDevData;
    CHECK(cudaMalloc((void**)&srcDevData, sizeof(uchar) * srcElements));
    CHECK(cudaMemcpyAsync(srcDevData, srcImg.data, sizeof(uchar) * srcElements, cudaMemcpyHostToDevice, stream));

    // Compute the letterbox ratio so the original aspect ratio is preserved and padding is inserted on the remaining side.
    int w, h, x, y;
    float r_w = dstWidth / (srcWidth * 1.0);
    float r_h = dstHeight / (srcHeight * 1.0);
    if (r_h > r_w) {
        w = dstWidth;
        h = r_w * srcHeight;
        x = 0;
        y = (dstHeight - h) / 2;
    }
    else {
        w = r_h * srcWidth;
        h = dstHeight;
        x = (dstWidth - w) / 2;
        y = 0;
    }

    dim3 blockSize(32, 32);
    dim3 gridSize((dstWidth + blockSize.x - 1) / blockSize.x, (dstHeight + blockSize.y - 1) / blockSize.y);

    // Kernel 1: resize + padding 114.
    letterbox<<<gridSize, blockSize, 0, stream>>>(srcDevData, srcHeight, srcWidth, midDevData, dstHeight, dstWidth, h, w, y, x);
    // Kernel 2: HWC BGR uchar -> normalized CHW RGB float32.
    process<<<gridSize, blockSize, 0, stream>>>(midDevData, dstDevData, dstHeight, dstWidth);

    CHECK(cudaFree(srcDevData));
    CHECK(cudaFree(midDevData));
}
