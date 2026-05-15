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

// YoloDetector gom toàn bộ tài nguyên sống xuyên suốt nhiều lần inference:
// engine/runtime/context của TensorRT, stream CUDA, buffer device/host và metadata binding.
class YoloDetector
{
public:
    // trtFile: đường dẫn .plan đã serialize hoặc nơi sẽ được ghi sau khi build.
    // onnxFile: nguồn chân lý để build engine khi chưa có .plan.
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

    // Trả về detection đã scale theo ảnh gốc.
    std::vector<Detection> inference(cv::Mat& img);
    // Chỉ đo thời gian enqueue TensorRT bằng CUDA event.
    double inference_model_only(cv::Mat& img);
    static void draw_image(cv::Mat& img, std::vector<Detection>& inferResult);

private:
    // Tải engine từ .plan nếu có, ngược lại build từ ONNX rồi serialize ra đĩa.
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

    // Số candidate head sinh ra, thường là 8400 cho input 640.
    int                 OUTPUT_CANDIDATES;

    // Repo giữ cả index lẫn tên tensor vì TRT8 dùng binding index còn TRT10 ưu tiên tensor name.
    int                 inputIndex_;
    int                 outputIndex_;
    std::string         inputName_;
    std::string         outputName_;
};

#endif  // INFER_H
