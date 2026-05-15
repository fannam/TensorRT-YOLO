#include <algorithm>
#include <numeric>
#include <string>

#include "utils.h"
#include "infer.h"


static std::string basename_without_ext(const std::string& path) {
    size_t slash = path.find_last_of("/\\");
    std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
    size_t dot = name.find_last_of('.');
    return dot == std::string::npos ? name : name.substr(0, dot);
}

static std::string resolve_onnx_path(const std::string& modelArg) {
    if (modelArg.find('/') != std::string::npos || modelArg.find('\\') != std::string::npos) {
        return modelArg;
    }
    if (modelArg.size() >= 5 && modelArg.substr(modelArg.size() - 5) == ".onnx") {
        return "../onnx_model/" + modelArg;
    }
    return "../onnx_model/" + modelArg + ".onnx";
}

static int run_trt_benchmark(
    char* imageDir,
    const std::string& onnxPath,
    const std::string& trtPath,
    const std::string& modelName,
    std::vector<double>& model_times,
    std::vector<double>& full_times
) {
    std::vector<std::string> file_names;
    if (read_files_in_dir(imageDir, file_names) < 0) return -1;
    std::sort(file_names.begin(), file_names.end());

    YoloDetector detector(trtPath, onnxPath);

    cv::Mat dummy(kInputH, kInputW, CV_8UC3, cv::Scalar(114, 114, 114));
    detector.inference_model_only(dummy);
    detector.inference(dummy);

    for (auto& fn : file_names) {
        cv::Mat img = cv::imread(std::string(imageDir) + "/" + fn, cv::IMREAD_COLOR);
        if (img.empty()) continue;

        double model_ms = detector.inference_model_only(img);
        model_times.push_back(model_ms);

        auto t0 = std::chrono::high_resolution_clock::now();
        auto res = detector.inference(img);
        auto t1 = std::chrono::high_resolution_clock::now();

        double full_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        full_times.push_back(full_ms);

        std::cout << "[TRT] " << fn
                  << " model-only: " << model_ms
                  << " ms, full: " << full_ms << " ms\n";

        YoloDetector::draw_image(img, res);
        cv::imwrite(modelName + "_" + fn, img);
    }

    return 0;
}


int main(int argc, char *argv[])
{
    if (argc < 3 || argc > 4) {
        printf("Usage: ./pose [image dir] [onnx file|model name] [plan file optional]\n");
        printf("Example: ./pose ../images yolo26m-pose\n");
        printf("Example: ./pose ../images ../onnx_model/yolo26m-pose.onnx ./yolo26m-pose.plan\n");
        return 1;
    }

    std::string onnxPath = resolve_onnx_path(argv[2]);
    std::string modelName = basename_without_ext(onnxPath);
    std::string trtPath = argc == 4 ? argv[3] : "./" + modelName + ".plan";
    std::vector<double> trt_model_times, trt_full_times;

    std::cout << "ONNX: " << onnxPath << "\n";
    std::cout << "TensorRT plan: " << trtPath << "\n";
    std::cout << "Output prefix: " << modelName << "_\n";

    std::cout << "\n=== TensorRT (" << (bFP16Mode ? "FP16" : "FP32") << ") ===\n";
    run_trt_benchmark(argv[1], onnxPath, trtPath, modelName, trt_model_times, trt_full_times);

    auto avg = [](const std::vector<double>& v) {
        if (v.empty()) return 0.0;
        double s = std::accumulate(v.begin(), v.end(), 0.0);
        return s / v.size();
    };

    std::cout << "\n=== Summary (avg after explicit warm-up) ===\n";
    std::cout << "TRT model-only avg: " << avg(trt_model_times) << " ms\n";
    std::cout << "TRT full pipeline avg: " << avg(trt_full_times) << " ms\n";

    return 0;
}
