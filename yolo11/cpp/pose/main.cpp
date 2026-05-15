#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include "utils.h"
#include "infer.h"

// The pose binary keeps the same benchmark structure as detect,
// but the full pipeline also decodes keypoints and renders the skeleton.

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

struct CliOptions {
    std::string imageDir;
    std::string modelArg;
    std::string planFile;
    Precision precision = Precision::kFP32;
};

static void print_usage() {
    printf("Usage: ./pose [image dir] [onnx file|model name] [plan file optional] [--precision fp16|fp32]\n");
    printf("Example: ./pose ../images yolo11s-pose --precision fp32\n");
    printf("Example: ./pose ../images ../onnx_model/yolo11s-pose.onnx ./yolo11s-pose.plan --precision fp16\n");
}

static bool parse_precision(const std::string& value, Precision& precision) {
    if (value == "fp16") {
        precision = Precision::kFP16;
        return true;
    }
    if (value == "fp32") {
        precision = Precision::kFP32;
        return true;
    }
    return false;
}

static bool parse_cli(int argc, char* argv[], CliOptions& options) {
    std::vector<std::string> positionalArgs;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--precision") {
            if (i + 1 >= argc || !parse_precision(argv[i + 1], options.precision)) {
                return false;
            }
            ++i;
            continue;
        }
        if (arg.rfind("--", 0) == 0) {
            return false;
        }
        positionalArgs.push_back(arg);
    }

    if (positionalArgs.size() < 2 || positionalArgs.size() > 3) {
        return false;
    }

    options.imageDir = positionalArgs[0];
    options.modelArg = positionalArgs[1];
    if (positionalArgs.size() == 3) {
        options.planFile = positionalArgs[2];
    }
    return true;
}

static std::string default_plan_path(const std::string& modelName, Precision precision) {
    return "./" + modelName + "_" + precision_to_cli_name(precision) + ".plan";
}

static int run_trt_benchmark(
    const std::string& imageDir,
    const std::string& onnxPath,
    const std::string& trtPath,
    const std::string& modelName,
    Precision precision,
    std::vector<double>& model_times,
    std::vector<double>& full_times
) {
    std::vector<std::string> file_names;
    if (read_files_in_dir(const_cast<char*>(imageDir.c_str()), file_names) < 0) return -1;
    std::sort(file_names.begin(), file_names.end());

    YoloDetector detector(trtPath, onnxPath, precision);

    // Warm up model-only and full-pipeline runs separately, just like detect, for more stable timings.
    cv::Mat dummy(kInputH, kInputW, CV_8UC3, cv::Scalar(114, 114, 114));
    detector.inference_model_only(dummy);
    detector.inference(dummy);

    for (auto& fn : file_names) {
        cv::Mat img = cv::imread(imageDir + "/" + fn, cv::IMREAD_COLOR);
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
    CliOptions options;
    if (!parse_cli(argc, argv, options)) {
        print_usage();
        return 1;
    }

    std::string onnxPath = resolve_onnx_path(options.modelArg);
    std::string modelName = basename_without_ext(onnxPath);
    std::string trtPath = options.planFile.empty() ? default_plan_path(modelName, options.precision) : options.planFile;
    std::vector<double> trt_model_times, trt_full_times;

    std::cout << "ONNX: " << onnxPath << "\n";
    std::cout << "TensorRT plan: " << trtPath << "\n";
    std::cout << "Output prefix: " << modelName << "_\n";

    std::cout << "\n=== TensorRT (" << precision_to_string(options.precision) << ") ===\n";
    run_trt_benchmark(options.imageDir, onnxPath, trtPath, modelName, options.precision, trt_model_times, trt_full_times);

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
