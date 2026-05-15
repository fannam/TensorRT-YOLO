#ifndef INFER_H
#define INFER_H

#include <opencv2/opencv.hpp>
#include "public.h"
#include "types.h"
#include "config.h"

using namespace nvinfer1;

// Segment quản lý ba binding chính: input, proto output và detect output.
class YoloDetector
{
public:
    YoloDetector(const std::string trtFile, const std::string onnxFile);
    ~YoloDetector();
    std::vector<Detection> inference(cv::Mat& img);
    double inference_model_only(cv::Mat& img);
    static void draw_image(cv::Mat& img, std::vector<Detection>& inferResult, bool drawBbox=true);

private:
    void get_engine();
    // Ghép mask coefficient với proto để tái tạo mask từng detection ngay trên GPU.
    static void process_mask(
        float* protoDevice, Dims protoOutDims, std::vector<Detection>& vDetections,
        int kInputH, int kInputW, cv::Mat& img, cudaStream_t stream
    );

private:
    Logger              gLogger;
    std::string         trtFile_;
    std::string         onnxFile_;

    ICudaEngine *       engine;
    IRuntime *          runtime;
    IExecutionContext * context;

    cudaStream_t        stream;

    float *             outputData;
    std::vector<void *> vBufferD;
    float *             transposeDevide;
    float *             decodeDevice;

    int                 OUTPUT_CANDIDATES;
    // Proto thường có shape [1, 32, 160, 160] và là "dictionary" để tái tạo mask.
    Dims                protoOutDims;

    int                 inputIndex_;
    int                 protoIndex_;
    int                 outputIndex_;
    std::string         inputName_;
    std::string         protoName_;
    std::string         outputName_;
};

#endif  // INFER_H
