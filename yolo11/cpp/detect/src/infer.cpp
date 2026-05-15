#include <iostream>
#include <fstream>

#include <NvOnnxParser.h>

#include "infer.h"
#include "preprocess.h"
#include "postprocess.h"
#include "calibrator.h"
#include "utils.h"

using namespace nvinfer1;

// File này giữ "xương sống" TensorRT của task detect:
// quản lý engine/context/buffer, thực hiện enqueue và nối các kernel CUDA hậu xử lý.

YoloDetector::YoloDetector(
        const std::string trtFile,
        const std::string onnxFile,
        int gpuId,
        float nmsThresh,
        float confThresh,
        int numClass
    ): trtFile_(trtFile), onnxFile_(onnxFile), nmsThresh_(nmsThresh), confThresh_(confThresh), numClass_(numClass)
{
    gLogger = Logger(ILogger::Severity::kERROR);
    cudaSetDevice(gpuId);

    CHECK(cudaStreamCreate(&stream));

    // Ưu tiên nạp engine serialize để khởi động nhanh; nếu chưa có sẽ build từ ONNX.
    get_engine();

    context = engine->createExecutionContext();

#if NV_TENSORRT_MAJOR >= 10
    // TRT10 truy cập I/O qua tensor name.
    inputIndex_ = 0;
    outputIndex_ = 1;
    for (int i = 0; i < engine->getNbIOTensors(); i++) {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
            inputName_ = name;
        } else {
            outputName_ = name;
        }
    }

    context->setInputShape(inputName_.c_str(), Dims {4, {1, 3, kInputH, kInputW}});

    // Head detect YOLO11 xuất [1, 4 + num_class, 8400].
    Dims outDims = context->getTensorShape(outputName_.c_str());
#else
    // TRT8/9 vẫn dùng binding index và binding dimensions.
    inputIndex_ = 0;
    outputIndex_ = 1;
    for (int i = 0; i < engine->getNbBindings(); i++) {
        if (engine->bindingIsInput(i)) {
            inputIndex_ = i;
        } else {
            outputIndex_ = i;
        }
    }

    context->setBindingDimensions(inputIndex_, Dims {4, {1, 3, kInputH, kInputW}});
    Dims outDims = context->getBindingDimensions(outputIndex_);
#endif
    OUTPUT_CANDIDATES = outDims.d[2];
    int outputSize = 1;
    for (int i = 0; i < outDims.nbDims; i++){
        outputSize *= outDims.d[i];
    }

    // outputData chỉ giữ kết quả đã decode + NMS, không phải raw head output.
    outputData = new float[1 + kMaxNumOutputBbox * kNumBoxElement];

    // vBufferD chứa đúng các binding TensorRT; các buffer còn lại là scratch buffer cho hậu xử lý GPU.
    vBufferD.resize(2, nullptr);
    CHECK(cudaMalloc(&vBufferD[inputIndex_], 3 * kInputH * kInputW * sizeof(float)));
    CHECK(cudaMalloc(&vBufferD[outputIndex_], outputSize * sizeof(float)));

    CHECK(cudaMalloc(&transposeDevice, outputSize * sizeof(float)));
    CHECK(cudaMalloc(&decodeDevice, (1 + kMaxNumOutputBbox * kNumBoxElement) * sizeof(float)));
}

void YoloDetector::get_engine(){
    if (access(trtFile_.c_str(), F_OK) == 0){
        // Nhánh nhanh: đọc plan có sẵn rồi deserialize.
        std::ifstream engineFile(trtFile_, std::ios::binary);
        long int fsize = 0;

        engineFile.seekg(0, engineFile.end);
        fsize = engineFile.tellg();
        engineFile.seekg(0, engineFile.beg);
        std::vector<char> engineString(fsize);
        engineFile.read(engineString.data(), fsize);
        if (engineString.size() == 0) { std::cout << "Failed getting serialized engine!" << std::endl; return; }
        std::cout << "Succeeded getting serialized engine!" << std::endl;

        runtime = createInferRuntime(gLogger);
        engine = runtime->deserializeCudaEngine(engineString.data(), fsize);
        if (engine == nullptr) { std::cout << "Failed loading engine!" << std::endl; return; }
        std::cout << "Succeeded loading engine!" << std::endl;
    } else {
        // Nhánh build: ONNX -> network -> optimization profile -> serialized engine -> runtime engine.
        IBuilder *            builder     = createInferBuilder(gLogger);
        INetworkDefinition *  network     = builder->createNetworkV2(
#if NV_TENSORRT_MAJOR >= 10
            0
#else
            1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)
#endif
        );
        IOptimizationProfile* profile     = builder->createOptimizationProfile();
        IBuilderConfig *      config      = builder->createBuilderConfig();
#if NV_TENSORRT_MAJOR >= 10
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1 << 30);
#else
        config->setMaxWorkspaceSize(1 << 30);
#endif
        IInt8Calibrator *     pCalibrator = nullptr;
        if (bFP16Mode){
            config->setFlag(BuilderFlag::kFP16);
        }
        if (bINT8Mode){
            config->setFlag(BuilderFlag::kINT8);
            // INT8 calibration chỉ xuất hiện ở lúc build, không liên quan inference runtime.
            int batchSize = 8;
            pCalibrator = new Int8EntropyCalibrator2(batchSize, kInputW, kInputH, calibrationDataPath.c_str(), cacheFile.c_str());
            config->setInt8Calibrator(pCalibrator);
        }

        nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
        if (!parser->parseFromFile(onnxFile_.c_str(), int(gLogger.reportableSeverity))){
            std::cout << std::string("Failed parsing .onnx file!") << std::endl;
            for (int i = 0; i < parser->getNbErrors(); ++i){
                auto *error = parser->getError(i);
                std::cout << std::to_string(int(error->code())) << std::string(":") << std::string(error->desc()) << std::endl;
            }
            return;
        }
        std::cout << std::string("Succeeded parsing .onnx file!") << std::endl;

        // Repo cố định profile ở đúng 1x3x640x640 nên sample này là static-shape.
        ITensor* inputTensor = network->getInput(0);
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims {4, {1, 3, kInputH, kInputW}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, Dims {4, {1, 3, kInputH, kInputW}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims {4, {1, 3, kInputH, kInputW}});
        config->addOptimizationProfile(profile);

        IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config);
        std::cout << "Succeeded building serialized engine!" << std::endl;

        runtime = createInferRuntime(gLogger);
        engine = runtime->deserializeCudaEngine(engineString->data(), engineString->size());
        if (engine == nullptr) { std::cout << "Failed building engine!" << std::endl; return; }
        std::cout << "Succeeded building engine!" << std::endl;

        if (bINT8Mode && pCalibrator != nullptr){
            delete pCalibrator;
        }

        // Serialize ra .plan để lần chạy sau bỏ qua parse/build ONNX.
        std::ofstream engineFile(trtFile_, std::ios::binary);
        engineFile.write(static_cast<char *>(engineString->data()), engineString->size());
        std::cout << "Succeeded saving .plan file!" << std::endl;

        delete engineString;
        delete parser;
        delete config;
        delete network;
        delete builder;
    }
}

YoloDetector::~YoloDetector(){
    // Giải phóng theo chiều ngược vòng đời: stream/buffer trước, TensorRT objects sau.
    cudaStreamDestroy(stream);

    for (int i = 0; i < 2; ++i)
    {
        CHECK(cudaFree(vBufferD[i]));
    }

    CHECK(cudaFree(transposeDevice));
    CHECK(cudaFree(decodeDevice));

    delete [] outputData;

    delete context;
    delete engine;
    delete runtime;
}

std::vector<Detection> YoloDetector::inference(cv::Mat& img){
    if (img.empty()) return {};

    // preprocess CUDA ghi thẳng tensor NCHW float vào input buffer của TensorRT.
    preprocess(img, (float*)vBufferD[inputIndex_], kInputH, kInputW, stream);

    // enqueue không đồng bộ trên cùng stream với preprocess/postprocess để tránh sync thừa.
#if NV_TENSORRT_MAJOR >= 10
    context->setTensorAddress(inputName_.c_str(), vBufferD[inputIndex_]);
    context->setTensorAddress(outputName_.c_str(), vBufferD[outputIndex_]);
    context->enqueueV3(stream);
#else
    context->enqueueV2(vBufferD.data(), stream, nullptr);
#endif

    // Tensor head xuất theo [C, N]. Các kernel decode/NMS thuận tiện hơn khi mỗi candidate
    // là một dải liên tiếp [cx, cy, w, h, cls...], nên cần transpose về [N, C].
    transpose((float*)vBufferD[outputIndex_], transposeDevice, OUTPUT_CANDIDATES, numClass_ + 4, stream);
    // decode gom class tốt nhất cho từng candidate, đổi bbox center-size thành xyxy và ghi
    // số lượng box hợp lệ vào phần tử đầu tiên bằng atomicAdd.
    decode(transposeDevice, decodeDevice, OUTPUT_CANDIDATES, numClass_, confThresh_, kMaxNumOutputBbox, kNumBoxElement, stream);
    // NMS trên GPU chỉ gạt keep_flag, nhờ đó host chỉ cần copy một buffer gọn đã hậu xử lý.
    nms(decodeDevice, nmsThresh_, kMaxNumOutputBbox, kNumBoxElement, stream);

    CHECK(cudaMemcpyAsync(outputData, decodeDevice, (1 + kMaxNumOutputBbox * kNumBoxElement) * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));

    std::vector<Detection> vDetections;
    int count = std::min((int)outputData[0], kMaxNumOutputBbox);
    for (int i = 0; i < count; i++){
        int pos = 1 + i * kNumBoxElement;
        int keepFlag = (int)outputData[pos + 6];
        if (keepFlag == 1){
            Detection det;
            memcpy(det.bbox, &outputData[pos], 4 * sizeof(float));
            det.conf = outputData[pos + 4];
            det.classId = (int)outputData[pos + 5];
            vDetections.push_back(det);
        }
    }

    // scale_bbox đảo ngược letterbox để bbox khớp hệ tọa độ ảnh đầu vào.
    for (size_t j = 0; j < vDetections.size(); j++){
        scale_bbox(img, vDetections[j].bbox);
    }

    return vDetections;
}

double YoloDetector::inference_model_only(cv::Mat& img){
    if (img.empty()) return 0.0;

    preprocess(img, (float*)vBufferD[inputIndex_], kInputH, kInputW, stream);

    // Dùng CUDA event trên cùng stream để chỉ đo thời gian enqueue + kernel nội bộ TensorRT.
    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start, stream));
#if NV_TENSORRT_MAJOR >= 10
    context->setTensorAddress(inputName_.c_str(), vBufferD[inputIndex_]);
    context->setTensorAddress(outputName_.c_str(), vBufferD[outputIndex_]);
    context->enqueueV3(stream);
#else
    context->enqueueV2(vBufferD.data(), stream, nullptr);
#endif
    CHECK(cudaEventRecord(stop, stream));
    CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    return milliseconds;
}

void YoloDetector::draw_image(cv::Mat& img, std::vector<Detection>& inferResult){
    // Hàm vẽ chỉ dùng cho sample/demo; không ảnh hưởng logic suy luận.
    for (size_t j = 0; j < inferResult.size(); j++)
    {
        cv::Scalar bboxColor(get_random_int(), get_random_int(), get_random_int());
        cv::Rect r(
            round(inferResult[j].bbox[0]),
            round(inferResult[j].bbox[1]),
            round(inferResult[j].bbox[2] - inferResult[j].bbox[0]),
            round(inferResult[j].bbox[3] - inferResult[j].bbox[1])
        );
        cv::rectangle(img, r, bboxColor, 2);

        std::string className = vClassNames[(int)inferResult[j].classId];
        std::string labelStr = className + " " + std::to_string(inferResult[j].conf).substr(0, 4);

        cv::Size textSize = cv::getTextSize(labelStr, cv::FONT_HERSHEY_PLAIN, 1.2, 2, NULL);
        cv::Point topLeft(r.x, r.y - textSize.height - 3);
        cv::Point bottomRight(r.x + textSize.width, r.y);
        cv::rectangle(img, topLeft, bottomRight, bboxColor, -1);
        cv::putText(img, labelStr, cv::Point(r.x, r.y - 2), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    }
}
