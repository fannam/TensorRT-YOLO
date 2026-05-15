#ifndef INFER_H
#define INFER_H

#include <opencv2/opencv.hpp>
#include "public.h"
#include "types.h"
#include "config.h"

using namespace nvinfer1;

enum class Precision {
    kFP32,
    kFP16
};

inline const char* precision_to_string(Precision precision) {
    return precision == Precision::kFP16 ? "FP16" : "FP32";
}

inline const char* precision_to_cli_name(Precision precision) {
    return precision == Precision::kFP16 ? "fp16" : "fp32";
}


class YoloDetector
{
public:
    YoloDetector(const std::string trtFile, const std::string onnxFile, Precision precision=Precision::kFP32);
    ~YoloDetector();
    std::vector<Detection> inference(cv::Mat& img);
    double inference_model_only(cv::Mat& img);
    static void draw_image(cv::Mat& img, std::vector<Detection>& inferResult, bool drawBbox=true);

private:
    void get_engine();
    static void process_mask(
        float* protoDevice, Dims protoOutDims, std::vector<Detection>& vDetections, 
        int kInputH, int kInputW, cv::Mat& img, cudaStream_t stream
    );

private:
    Logger              gLogger;
    std::string         trtFile_;
    std::string         onnxFile_;
    Precision           precision_;

    ICudaEngine *       engine;
    IRuntime *          runtime;
    IExecutionContext * context;

    cudaStream_t        stream;

    float *             outputData;
    std::vector<void *> vBufferD;
    Dims              protoOutDims;  // proto shape [1 32 160 160]
    Dims              outputOutDims;  // output0 shape [1 300 38]
    int                 outputRows_;
    int                 outputCols_;

    int                 inputIndex_;
    int                 protoIndex_;
    int                 outputIndex_;
    std::string         inputName_;
    std::string         protoName_;
    std::string         outputName_;
};

#endif  // INFER_H
