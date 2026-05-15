#include <array>
#include <numeric>
#include <algorithm>
#include <string>
#include <vector>
#ifdef ENABLE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

#include "utils.h"
#include "infer.h"

// Binary segment đo riêng model-only và full pipeline.
// Với full pipeline, phần mask reconstruction thường chiếm đáng kể thời gian hơn detect thuần.

#ifdef ENABLE_ONNXRUNTIME
static void preprocess_cpu(const cv::Mat& img, float* data, int th, int tw) {
    float r = std::min((float)tw / img.cols, (float)th / img.rows);
    int nw = (int)(img.cols * r);
    int nh = (int)(img.rows * r);
    int pw = (tw - nw) / 2;
    int ph = (th - nh) / 2;

    cv::Mat resized, padded(th, tw, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::resize(img, resized, cv::Size(nw, nh));
    resized.copyTo(padded(cv::Rect(pw, ph, nw, nh)));

    cv::Mat rgb, fp;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(fp, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> ch(3);
    cv::split(fp, ch);
    for (int c = 0; c < 3; c++)
        memcpy(data + c * th * tw, ch[c].data, th * tw * sizeof(float));
}
#endif

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
    printf("Usage: ./segment [image dir] [onnx file|model name] [plan file optional] [--precision fp16|fp32]\n");
    printf("Example: ./segment ../images yolo11s-seg --precision fp32\n");
    printf("Example: ./segment ../images ../onnx_model/yolo11l-seg.onnx ./yolo11l-seg.plan --precision fp16\n");
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

static int run_ort(const std::string& imageDir, const std::string& onnxPath, std::vector<double>& times_out) {
#ifdef ENABLE_ONNXRUNTIME
    std::vector<std::string> file_names;
    if (read_files_in_dir(const_cast<char*>(imageDir.c_str()), file_names) < 0) return -1;
    std::sort(file_names.begin(), file_names.end());

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ort_bench");
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    OrtCUDAProviderOptions cuda_opts{};
    cuda_opts.device_id = kGpuId;
    opts.AppendExecutionProvider_CUDA(cuda_opts);

    Ort::Session session(env, onnxPath.c_str(), opts);

    Ort::AllocatorWithDefaultOptions alloc;
    std::string in_name  = session.GetInputNameAllocated(0, alloc).get();
    std::string out0_name = session.GetOutputNameAllocated(0, alloc).get();
    std::string out1_name = session.GetOutputNameAllocated(1, alloc).get();

    const char* input_names[]  = { in_name.c_str() };
    const char* output_names[] = { out0_name.c_str(), out1_name.c_str() };

    std::vector<float> input_buf(3 * kInputH * kInputW);
    std::array<int64_t, 4> input_shape{1, 3, kInputH, kInputW};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    {
        cv::Mat dummy(kInputH, kInputW, CV_8UC3, cv::Scalar(114, 114, 114));
        preprocess_cpu(dummy, input_buf.data(), kInputH, kInputW);
        auto tensor = Ort::Value::CreateTensor<float>(mem, input_buf.data(), input_buf.size(),
                                                      input_shape.data(), input_shape.size());
        session.Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1, output_names, 2);
    }

    for (auto& fn : file_names) {
        cv::Mat img = cv::imread(imageDir + "/" + fn, cv::IMREAD_COLOR);
        if (img.empty()) continue;

        preprocess_cpu(img, input_buf.data(), kInputH, kInputW);
        auto tensor = Ort::Value::CreateTensor<float>(mem, input_buf.data(), input_buf.size(),
                                                      input_shape.data(), input_shape.size());

        auto t0 = std::chrono::high_resolution_clock::now();
        session.Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1, output_names, 2);
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times_out.push_back(ms);
        std::cout << "[ORT] " << fn << " model-only: " << ms << " ms\n";
    }
    return 0;
#else
    (void)imageDir;
    (void)onnxPath;
    (void)times_out;
    std::cout << "[ORT] skipped; build with -DENABLE_ONNXRUNTIME=ON to enable ONNX Runtime benchmark\n";
    return 0;
#endif
}

int main(int argc, char* argv[]) {
    CliOptions options;
    if (!parse_cli(argc, argv, options)) {
        print_usage();
        return 1;
    }

    std::string onnxPath = resolve_onnx_path(options.modelArg);
    std::string modelName = basename_without_ext(onnxPath);
    std::string trtPath = options.planFile.empty() ? default_plan_path(modelName, options.precision) : options.planFile;
    std::vector<double> trt_model_times, trt_full_times;
#ifdef ENABLE_ONNXRUNTIME
    std::vector<double> ort_times;
#endif

    std::cout << "ONNX: " << onnxPath << "\n";
    std::cout << "TensorRT plan: " << trtPath << "\n";
    std::cout << "Output prefix: " << modelName << "_\n";

    std::cout << "\n=== TensorRT (" << precision_to_string(options.precision) << ") ===\n";
    run_trt_benchmark(options.imageDir, onnxPath, trtPath, modelName, options.precision, trt_model_times, trt_full_times);

#ifdef ENABLE_ONNXRUNTIME
    std::cout << "\n=== ONNX Runtime (CUDA) ===\n";
    run_ort(options.imageDir, onnxPath, ort_times);
#endif

    auto avg = [](const std::vector<double>& v, int skip) {
        if ((int)v.size() <= skip) return 0.0;
        double s = std::accumulate(v.begin() + skip, v.end(), 0.0);
        return s / (v.size() - skip);
    };

    std::cout << "\n=== Summary (avg after explicit warm-up) ===\n";
    std::cout << "TRT model-only avg: " << avg(trt_model_times, 0) << " ms\n";
#ifdef ENABLE_ONNXRUNTIME
    std::cout << "ORT model-only avg: " << avg(ort_times, 0) << " ms\n";
#endif
    std::cout << "TRT full pipeline avg: " << avg(trt_full_times, 0) << " ms\n";
#ifdef ENABLE_ONNXRUNTIME
    std::cout << "Model-only speedup TRT/ORT: "
              << avg(ort_times, 0) / std::max(avg(trt_model_times, 0), 1.0) << "x\n";
#endif

    return 0;
}
