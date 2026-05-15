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

// Detector cho pose giữ nguyên khung TensorRT của detect, nhưng decode ra bbox + class + keypoint.
class YoloDetector
{
public:
    YoloDetector(const std::string trtFile, const std::string onnxFile, Precision precision=Precision::kFP32);
    ~YoloDetector();
    std::vector<Detection> inference(cv::Mat& img);
    double inference_model_only(cv::Mat& img);
    static void draw_image(cv::Mat& img, std::vector<Detection>& inferResult, bool drawBbox=true, bool kptLine=true);

private:
    void get_engine();

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
    float *             transposeDevice;
    float *             decodeDevice;

    // Với pose, head thường có shape [1, 56, 8400] = 4 bbox + 1 class + 51 keypoint values.
    int                 OUTPUT_CANDIDATES;

    int                 inputIndex_;
    int                 outputIndex_;
    std::string         inputName_;
    std::string         outputName_;
};

#endif  // INFER_H
