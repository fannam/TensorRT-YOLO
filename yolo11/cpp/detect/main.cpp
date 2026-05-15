#include <algorithm>
#include <array>
#include <numeric>
#include <string>
#ifdef ENABLE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

#include "utils.h"
#include "infer.h"

// Binary mẫu cho bài toán detect:
// CLI -> resolve ONNX/.plan -> warm-up TensorRT -> benchmark model-only và full pipeline
// -> ghi ảnh đã vẽ bbox ra thư mục hiện tại.

#ifdef ENABLE_ONNXRUNTIME
// Preprocess CPU này cố ý mô phỏng đúng preprocess CUDA để benchmark ORT/TensorRT
// chủ yếu khác nhau ở runtime engine chứ không phải ở khâu chuẩn hóa input.
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
    for (int c = 0; c < 3; c++) {
        memcpy(data + c * th * tw, ch[c].data, th * tw * sizeof(float));
    }
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

    // Warm-up tách riêng hai kiểu đo để loại chi phí lần chạy đầu:
    // - model_only: chỉ enqueue TensorRT
    // - full: nhìn từ phía người dùng end-to-end
    cv::Mat dummy(kInputH, kInputW, CV_8UC3, cv::Scalar(114, 114, 114));
    for (int i = 0; i < 10; i++) {
        detector.inference_model_only(dummy);
    }
    for (int i = 0; i < 2; i++) {
        detector.inference(dummy);
    }

    for (auto& fn : file_names) {
        cv::Mat img = cv::imread(std::string(imageDir) + "/" + fn, cv::IMREAD_COLOR);
        if (img.empty()) continue;

        double model_ms = detector.inference_model_only(img);
        model_times.push_back(model_ms);

        // Full pipeline timing bao gồm preprocess, postprocess, copy host và scale bbox.
        auto t0 = std::chrono::high_resolution_clock::now();
        auto res = detector.inference(img);
        auto t1 = std::chrono::high_resolution_clock::now();

        double full_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        full_times.push_back(full_ms);

        std::cout << "[TRT] " << fn
                  << " model-only: " << model_ms
                  << " ms, full: " << full_ms << " ms\n";

        YoloDetector::draw_image(img, res);
        // Prefix theo modelName để có thể chạy nhiều model chung một thư mục output.
        cv::imwrite(modelName + "_" + fn, img);
    }
    return 0;
}

static int run_ort(char* imageDir, const std::string& onnxPath, std::vector<double>& times_out) {
#ifdef ENABLE_ONNXRUNTIME
    std::vector<std::string> file_names;
    if (read_files_in_dir(imageDir, file_names) < 0) return -1;
    std::sort(file_names.begin(), file_names.end());

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ort_bench");
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    OrtCUDAProviderOptions cuda_opts{};
    cuda_opts.device_id = kGpuId;
    opts.AppendExecutionProvider_CUDA(cuda_opts);

    Ort::Session session(env, onnxPath.c_str(), opts);

    Ort::AllocatorWithDefaultOptions alloc;
    std::string in_name = session.GetInputNameAllocated(0, alloc).get();
    size_t output_count = session.GetOutputCount();
    std::vector<std::string> output_name_storage;
    std::vector<const char*> output_names;
    output_name_storage.reserve(output_count);
    output_names.reserve(output_count);
    for (size_t i = 0; i < output_count; i++) {
        output_name_storage.emplace_back(session.GetOutputNameAllocated(i, alloc).get());
        output_names.push_back(output_name_storage.back().c_str());
    }

    const char* input_names[] = { in_name.c_str() };
    std::vector<float> input_buf(3 * kInputH * kInputW);
    std::array<int64_t, 4> input_shape{1, 3, kInputH, kInputW};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    for (int i = 0; i < 10; i++) {
        cv::Mat dummy(kInputH, kInputW, CV_8UC3, cv::Scalar(114, 114, 114));
        preprocess_cpu(dummy, input_buf.data(), kInputH, kInputW);
        auto tensor = Ort::Value::CreateTensor<float>(mem, input_buf.data(), input_buf.size(),
                                                      input_shape.data(), input_shape.size());
        session.Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1, output_names.data(), output_names.size());
    }

    for (auto& fn : file_names) {
        cv::Mat img = cv::imread(std::string(imageDir) + "/" + fn, cv::IMREAD_COLOR);
        if (img.empty()) continue;

        preprocess_cpu(img, input_buf.data(), kInputH, kInputW);
        auto tensor = Ort::Value::CreateTensor<float>(mem, input_buf.data(), input_buf.size(),
                                                      input_shape.data(), input_shape.size());

        auto t0 = std::chrono::high_resolution_clock::now();
        session.Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1, output_names.data(), output_names.size());
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
    if (argc < 3 || argc > 4) {
        printf("Usage: ./detect [image dir] [onnx file|model name] [plan file optional]\n");
        printf("Example: ./detect ../images yolo11s\n");
        printf("Example: ./detect ../images ../onnx_model/yolo11s.onnx ./yolo11s.plan\n");
        return 1;
    }

    // Cho phép truyền "yolo11s" hoặc path ONNX đầy đủ. Điều này giữ CLI ngắn nhưng vẫn
    // hỗ trợ benchmark model ở vị trí tùy ý.
    std::string onnxPath = resolve_onnx_path(argv[2]);
    std::string modelName = basename_without_ext(onnxPath);
    // Nếu không truyền .plan thì sample sẽ tự dùng ./<model>.plan cạnh binary/build dir.
    std::string trtPath = argc == 4 ? argv[3] : "./" + modelName + ".plan";
    std::vector<double> trt_model_times, trt_full_times;
#ifdef ENABLE_ONNXRUNTIME
    std::vector<double> ort_times;
#endif

    std::cout << "ONNX: " << onnxPath << "\n";
    std::cout << "TensorRT plan: " << trtPath << "\n";
    std::cout << "Output prefix: " << modelName << "_\n";

    std::cout << "\n=== TensorRT (" << (bFP16Mode ? "FP16" : "FP32") << ") ===\n";
    run_trt_benchmark(argv[1], onnxPath, trtPath, modelName, trt_model_times, trt_full_times);

#ifdef ENABLE_ONNXRUNTIME
    std::cout << "\n=== ONNX Runtime (CUDA) ===\n";
    run_ort(argv[1], onnxPath, ort_times);
#endif

    auto avg = [](const std::vector<double>& v) {
        if (v.empty()) return 0.0;
        double s = std::accumulate(v.begin(), v.end(), 0.0);
        return s / v.size();
    };

    std::cout << "\n=== Summary (avg after explicit warm-up) ===\n";
    std::cout << "TRT model-only avg: " << avg(trt_model_times) << " ms\n";
#ifdef ENABLE_ONNXRUNTIME
    std::cout << "ORT model-only avg: " << avg(ort_times) << " ms\n";
#endif
    std::cout << "TRT full pipeline avg: " << avg(trt_full_times) << " ms\n";
#ifdef ENABLE_ONNXRUNTIME
    std::cout << "Model-only speedup TRT/ORT: "
              << avg(ort_times) / std::max(avg(trt_model_times), 1.0) << "x\n";
#endif

    return 0;
}
