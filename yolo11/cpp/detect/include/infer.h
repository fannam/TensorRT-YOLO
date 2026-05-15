#ifndef INFER_H
#define INFER_H

#include <opencv2/opencv.hpp>
#include "public.h"
#include "config.h"
#include "types.h"

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

// YoloDetector owns all resources that live across repeated inference calls:
// TensorRT engine/runtime/context, CUDA stream, device/host buffers, and binding metadata.
class YoloDetector
{
public:
    // trtFile: path to an existing serialized .plan, or the output location after build.
    // onnxFile: source of truth for engine build when no .plan exists.
    YoloDetector(
        const std::string trtFile,
        const std::string onnxFile,
        Precision precision=Precision::kFP32,
        int gpuId=kGpuId,
        float nmsThresh=kNmsThresh,
        float confThresh=kConfThresh,
        int numClass=kNumClass
    );
    ~YoloDetector();

    // Returns detections already scaled to the original image.
    std::vector<Detection> inference(cv::Mat& img);
    // Measures TensorRT enqueue time only via CUDA events.
    double inference_model_only(cv::Mat& img);
    static void draw_image(cv::Mat& img, std::vector<Detection>& inferResult);

private:
    // Loads an engine from .plan if available, otherwise builds from ONNX and serializes it.
    void get_engine();

private:
    Logger              gLogger;
    std::string         trtFile_;
    std::string         onnxFile_;
    Precision           precision_;

    int                 numClass_;
    float               nmsThresh_;
    float               confThresh_;

    ICudaEngine *       engine;
    IRuntime *          runtime;
    IExecutionContext * context;

    cudaStream_t        stream;

    float *             outputData;
    std::vector<void *> vBufferD;
    float *             transposeDevice;
    float *             decodeDevice;

    // Number of head candidates, typically 8400 for 640 input.
    int                 OUTPUT_CANDIDATES;

    // The repo keeps both indices and tensor names because TRT8 uses binding indices while TRT10 prefers tensor names.
    int                 inputIndex_;
    int                 outputIndex_;
    std::string         inputName_;
    std::string         outputName_;
};

#endif  // INFER_H
