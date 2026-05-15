#include <iostream>
#include <fstream>

#include <NvOnnxParser.h>

#include "infer.h"
#include "preprocess.h"
#include "postprocess.h"
#include "calibrator.h"
#include "utils.h"

using namespace nvinfer1;

// Pose giữ cùng bộ khung TensorRT như detect; khác biệt chính nằm ở shape output,
// buffer decode lớn hơn và bước scale keypoint về ảnh gốc.

YoloDetector::YoloDetector(
        const std::string trtFile,
        const std::string onnxFile,
        Precision precision
    ): trtFile_(trtFile), onnxFile_(onnxFile), precision_(precision)
{
    gLogger = Logger(ILogger::Severity::kERROR);
    cudaSetDevice(kGpuId);

    CHECK(cudaStreamCreate(&stream));

    get_engine();

    context = engine->createExecutionContext();

#if NV_TENSORRT_MAJOR >= 10
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

    // 56 = 4 bbox + 1 class + 17 * 3 keypoint.
    Dims outDims = context->getTensorShape(outputName_.c_str());
#else
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

    outputData = new float[1 + kMaxNumOutputBbox * kNumBoxElement];
    vBufferD.resize(2, nullptr);
    CHECK(cudaMalloc(&vBufferD[inputIndex_], 3 * kInputH * kInputW * sizeof(float)));
    CHECK(cudaMalloc(&vBufferD[outputIndex_], outputSize * sizeof(float)));

    CHECK(cudaMalloc(&transposeDevice, outputSize * sizeof(float)));
    CHECK(cudaMalloc(&decodeDevice, (1 + kMaxNumOutputBbox * kNumBoxElement) * sizeof(float)));
}

void YoloDetector::get_engine(){
    if (access(trtFile_.c_str(), F_OK) == 0){
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
        if (precision_ == Precision::kFP16){
            config->setFlag(BuilderFlag::kFP16);
        }
        if (bINT8Mode){
            config->setFlag(BuilderFlag::kINT8);
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

    preprocess(img, (float*)vBufferD[inputIndex_], kInputH, kInputW, stream);

#if NV_TENSORRT_MAJOR >= 10
    context->setTensorAddress(inputName_.c_str(), vBufferD[inputIndex_]);
    context->setTensorAddress(outputName_.c_str(), vBufferD[outputIndex_]);
    context->enqueueV3(stream);
#else
    context->enqueueV2(vBufferD.data(), stream, nullptr);
#endif

    // [56, 8400] -> [8400, 56] để một thread decode đọc đủ bbox + class + keypoint của 1 candidate.
    transpose((float*)vBufferD[outputIndex_], transposeDevice, OUTPUT_CANDIDATES, 4 + kNumClass + kNumKpt * kKptDims, stream);
    int nk = kNumKpt * kKptDims;
    decode(transposeDevice, decodeDevice, OUTPUT_CANDIDATES, kNumClass, nk, kConfThresh, kMaxNumOutputBbox, kNumBoxElement, stream);
    nms(decodeDevice, kNmsThresh, kMaxNumOutputBbox, kNumBoxElement, stream);

    CHECK(cudaMemcpyAsync(outputData, decodeDevice, (1 + kMaxNumOutputBbox * kNumBoxElement) * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

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
            // 51 giá trị keypoint được giữ nguyên thứ tự [x, y, conf] * 17.
            memcpy(det.kpts, &outputData[pos + 7], kNumKpt * kKptDims * sizeof(float));
            vDetections.push_back(det);
        }
    }

    for (size_t j = 0; j < vDetections.size(); j++){
        scale_bbox(img, vDetections[j].bbox);
        vDetections[j].vKpts = scale_kpt_coords(img, vDetections[j].kpts);
    }

    return vDetections;
}

double YoloDetector::inference_model_only(cv::Mat& img){
    if (img.empty()) return 0.0;

    preprocess(img, (float*)vBufferD[inputIndex_], kInputH, kInputW, stream);

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

void YoloDetector::draw_image(cv::Mat& img, std::vector<Detection>& inferResult, bool drawBbox, bool kptLine){
    for (size_t j = 0; j < inferResult.size(); j++)
    {
        if (drawBbox){
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

        int x, y;
        float conf;
        int radius = std::min(img.rows, img.cols) / 100;
        cv::Scalar kptColor(get_random_int(), get_random_int(), get_random_int());

        std::vector<std::vector<float>> vScaledKpts = inferResult[j].vKpts;
        for (size_t k = 0; k < vScaledKpts.size(); k++){
            x = (int)vScaledKpts[k][0];
            y = (int)vScaledKpts[k][1];
            conf = vScaledKpts[k][2];
            if (x < 0 || x > img.cols || y < 0 || y > img.rows) continue;
            if (conf < 0.5) continue;
            cv::circle(img, cv::Point(x, y), radius, kptColor, -1);
        }

        if (kptLine){
            int kpt1_idx, kpt2_idx, kpt1_x, kpt1_y, kpt2_x, kpt2_y;
            float kpt1_conf, kpt2_conf;
            int skeleton_width = std::min(img.rows, img.cols) / 300;
            cv::Scalar skeletonColor(get_random_int(), get_random_int(), get_random_int());
            for (size_t m = 0; m < skeleton.size(); m++){
                kpt1_idx = skeleton[m][0] - 1;
                kpt2_idx = skeleton[m][1] - 1;
                kpt1_x = (int)vScaledKpts[kpt1_idx][0];
                kpt1_y = (int)vScaledKpts[kpt1_idx][1];
                kpt1_conf = vScaledKpts[kpt1_idx][2];
                kpt2_x = (int)vScaledKpts[kpt2_idx][0];
                kpt2_y = (int)vScaledKpts[kpt2_idx][1];
                kpt2_conf = vScaledKpts[kpt2_idx][2];
                if (kpt1_conf < 0.5 || kpt2_conf < 0.5) continue;
                if (kpt1_x > img.cols || kpt1_y > img.rows || kpt1_x < 0 || kpt1_y < 0) continue;
                if (kpt2_x > img.cols || kpt2_y > img.rows || kpt2_x < 0 || kpt2_y < 0) continue;
                cv::line(img, cv::Point(kpt1_x, kpt1_y), cv::Point(kpt2_x, kpt2_y), skeletonColor, skeleton_width, cv::LINE_AA);
            }
        }
    }
}
