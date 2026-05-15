#include <iostream>
#include <fstream>

#include <NvOnnxParser.h>

#include "infer.h"
#include "preprocess.h"
#include "postprocess.h"
#include "calibrator.h"
#include "utils.h"
#include "draw.h"

using namespace nvinfer1;

// Segment khác detect ở chỗ runtime phải quản lý thêm một output proto và bước
// tái tạo mask từ tổ hợp tuyến tính giữa proto với 32 hệ số của từng detection.

YoloDetector::YoloDetector(const std::string trtFile, const std::string onnxFile): trtFile_(trtFile), onnxFile_(onnxFile)
{
    gLogger = Logger(ILogger::Severity::kERROR);
    cudaSetDevice(kGpuId);

    CHECK(cudaStreamCreate(&stream));

    // Ưu tiên đọc .plan; nếu chưa có sẽ build từ ONNX giống detect/pose.
    get_engine();

    context = engine->createExecutionContext();

#if NV_TENSORRT_MAJOR >= 10
    inputIndex_ = 0;
    protoIndex_ = 1;
    outputIndex_ = 2;
    // TRT10: phân biệt hai output bằng số chiều.
    // - proto: 4D [1, 32, 160, 160]
    // - detect: 3D [1, 116, 8400] = 4 bbox + 80 class + 32 mask coefficient
    for (int i = 0; i < engine->getNbIOTensors(); i++) {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
            inputName_ = name;
        } else {
            auto dims = engine->getTensorShape(name);
            if (dims.nbDims == 4) protoName_ = name;
            else outputName_ = name;
        }
    }

    context->setInputShape(inputName_.c_str(), Dims {4, {1, 3, kInputH, kInputW}});

    protoOutDims = context->getTensorShape(protoName_.c_str());
    Dims outDims = context->getTensorShape(outputName_.c_str());
#else
    inputIndex_ = 0;
    protoIndex_ = 1;
    outputIndex_ = 2;

    for (int i = 0; i < engine->getNbBindings(); i++) {
        if (engine->bindingIsInput(i)) {
            inputIndex_ = i;
        }
    }

    context->setBindingDimensions(inputIndex_, Dims {4, {1, 3, kInputH, kInputW}});

    for (int i = 0; i < engine->getNbBindings(); i++) {
        if (engine->bindingIsInput(i)) continue;
        Dims dims = context->getBindingDimensions(i);
        if (dims.nbDims == 4) {
            protoIndex_ = i;
        } else {
            outputIndex_ = i;
        }
    }

    protoOutDims = context->getBindingDimensions(protoIndex_);
    Dims outDims = context->getBindingDimensions(outputIndex_);
#endif
    int protoOutputSize = 1;
    for (int i = 0; i < protoOutDims.nbDims; i++){
        protoOutputSize *= protoOutDims.d[i];
    }

    OUTPUT_CANDIDATES = outDims.d[2];
    int outputSize = 1;
    for (int i = 0; i < outDims.nbDims; i++){
        outputSize *= outDims.d[i];
    }

    // outputData chỉ giữ kết quả detect đã decode/NMS; proto vẫn nằm trên device.
    outputData = new float[1 + kMaxNumOutputBbox * kNumBoxElement];
    vBufferD.resize(3, nullptr);
    CHECK(cudaMalloc(&vBufferD[inputIndex_], 3 * kInputH * kInputW * sizeof(float)));
    CHECK(cudaMalloc(&vBufferD[protoIndex_], protoOutputSize * sizeof(float)));
    CHECK(cudaMalloc(&vBufferD[outputIndex_], outputSize * sizeof(float)));

    // transposeDevide: detect output [C, N] -> [N, C]
    // decodeDevice: buffer detect đã gói gọn cả 32 mask coefficient mỗi box.
    CHECK(cudaMalloc(&transposeDevide, outputSize * sizeof(float)));
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
        // Nhánh build giữ cùng quy trình với detect/pose, chỉ khác ở graph ONNX có 2 output.
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

    for (int i = 0; i < 3; ++i)
    {
        CHECK(cudaFree(vBufferD[i]));
    }

    CHECK(cudaFree(transposeDevide));
    CHECK(cudaFree(decodeDevice));

    delete [] outputData;

    delete context;
    delete engine;
    delete runtime;
}

std::vector<Detection> YoloDetector::inference(cv::Mat& img){
    if (img.empty()) return {};

    // preprocess CUDA ghi thẳng tensor input lên device.
    preprocess(img, (float*)vBufferD[inputIndex_], kInputH, kInputW, stream);

    // Cả detect head và proto head đều được enqueue trên cùng stream.
#if NV_TENSORRT_MAJOR >= 10
    context->setTensorAddress(inputName_.c_str(), vBufferD[inputIndex_]);
    context->setTensorAddress(protoName_.c_str(), vBufferD[protoIndex_]);
    context->setTensorAddress(outputName_.c_str(), vBufferD[outputIndex_]);
    context->enqueueV3(stream);
#else
    context->enqueueV2(vBufferD.data(), stream, nullptr);
#endif

    // detect output giữ bbox/class/mask coefficient; proto output dùng ở bước process_mask bên dưới.
    transpose((float*)vBufferD[outputIndex_], transposeDevide, OUTPUT_CANDIDATES, 4 + kNumClass + 32, stream);
    decode(transposeDevide, decodeDevice, OUTPUT_CANDIDATES, kNumClass, 32, kConfThresh, kMaxNumOutputBbox, kNumBoxElement, stream);
    nms(decodeDevice, kNmsThresh, kMaxNumOutputBbox, kNumBoxElement, stream);
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
            // 32 hệ số này sẽ được nhân với proto để tái tạo mask riêng cho detection.
            memcpy(det.mask, &outputData[pos + 7], 32 * sizeof(float));
            vDetections.push_back(det);
        }
    }

    // process_mask tiêu thụ proto tensor + 32 mask coefficient để tạo mask đúng kích thước ảnh gốc.
    process_mask((float*)vBufferD[protoIndex_], protoOutDims, vDetections, kInputH, kInputW, img, stream);
    cudaStreamSynchronize(stream);

    for (size_t j = 0; j < vDetections.size(); j++){
        scale_bbox(img, vDetections[j].bbox);
    }

    return vDetections;
}

double YoloDetector::inference_model_only(cv::Mat& img){
    if (img.empty()) return 0.0;

    preprocess(img, (float*)vBufferD[inputIndex_], kInputH, kInputW, stream);

#if NV_TENSORRT_MAJOR >= 10
    context->setTensorAddress(inputName_.c_str(), vBufferD[inputIndex_]);
    context->setTensorAddress(protoName_.c_str(), vBufferD[protoIndex_]);
    context->setTensorAddress(outputName_.c_str(), vBufferD[outputIndex_]);
#endif

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start, stream));
#if NV_TENSORRT_MAJOR >= 10
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

void YoloDetector::process_mask(
    float* protoDevice, Dims protoOutDims, std::vector<Detection>& vDetections, 
    int kInputH, int kInputW, cv::Mat& img, cudaStream_t stream
){
    int protoC = protoOutDims.d[1];
    int protoH = protoOutDims.d[2];
    int protoW = protoOutDims.d[3];

    int n = vDetections.size();
    if (n == 0) return;

    // Mỗi detection đóng góp một vector coefficient dài 32.
    float* maskCoefDevice = nullptr;
    CHECK(cudaMalloc(&maskCoefDevice, n * protoC * sizeof(float)));
    // maskDevice sẽ giữ n mask ở độ phân giải proto.
    float* maskDevice = nullptr;
    CHECK(cudaMalloc(&maskDevice, n * protoH * protoW * sizeof(float)));

    float* bboxDevice = nullptr;
    CHECK(cudaMalloc(&bboxDevice, n * 4 * sizeof(float)));

    for (size_t i = 0; i < n; i++){
        CHECK(cudaMemcpyAsync(&maskCoefDevice[i * protoC], vDetections[i].mask, protoC * sizeof(float), cudaMemcpyHostToDevice, stream));
        CHECK(cudaMemcpyAsync(&bboxDevice[i * 4], vDetections[i].bbox, 4 * sizeof(float), cudaMemcpyHostToDevice, stream));
    }

    // mask = sigmoid(mask_coef [n,32] x proto [32,H*W]) -> [n,H*W].
    matrix_multiply(maskCoefDevice, n, protoC, protoDevice, protoC, protoH * protoW, maskDevice, stream, true);

    // Bbox đang nằm ở hệ 640x640, còn proto ở 160x160 nên phải downsample theo cùng ratio.
    float heightRatio = (float)protoH / (float)kInputH;
    float widthRatio = (float)protoW / (float)kInputW;
    downsample_bbox(bboxDevice, n * 4, heightRatio, widthRatio, stream);

    // Zero-out phần mask nằm ngoài bbox để hạn chế nhiễu nền.
    crop_mask(maskDevice, n, protoH, protoW, bboxDevice, stream);

    // scale mask từ hệ proto về ảnh gốc:
    // 1) cắt bỏ vùng padding do letterbox
    // 2) bilinear resize về kích thước ảnh thật
    float r_w = protoW / (img.cols * 1.0);
    float r_h = protoH / (img.rows * 1.0);
    float r = std::min(r_w, r_h);
    float pad_h = (protoH - r * img.rows) / 2;
    float pad_w = (protoW - r * img.cols) / 2;
    int cutMaskLeft = (int)pad_w;
    int cutMaskTop = (int)pad_h;
    int cutMaskRight = (int)(protoW - pad_w);
    int cutMaskBottom = (int)(protoH - pad_h);
    int cutMaskWidth = cutMaskRight - cutMaskLeft;
    int cutMaskHeight = cutMaskBottom - cutMaskTop;
    float* cutMaskDevice = nullptr;
    CHECK(cudaMalloc(&cutMaskDevice, n * cutMaskHeight * cutMaskWidth * sizeof(float)));
    cut_mask(maskDevice, n, protoH, protoW, cutMaskDevice, cutMaskTop, cutMaskLeft, cutMaskHeight, cutMaskWidth, stream);

    float* scaledMaskDevice = nullptr;
    CHECK(cudaMalloc(&scaledMaskDevice, n * img.rows * img.cols * sizeof(float)));
    resize(cutMaskDevice, n, cutMaskHeight, cutMaskWidth, scaledMaskDevice, img.rows, img.cols, stream);

    for (size_t i = 0; i < n; i++){
        vDetections[i].maskMatrix.resize(img.rows * img.cols);
        CHECK(cudaMemcpyAsync(vDetections[i].maskMatrix.data(), &scaledMaskDevice[i * img.rows * img.cols], img.rows * img.cols * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }

    CHECK(cudaFree(maskCoefDevice));
    CHECK(cudaFree(maskDevice));
    CHECK(cudaFree(bboxDevice));
    CHECK(cudaFree(cutMaskDevice));
    CHECK(cudaFree(scaledMaskDevice));
}


void YoloDetector::draw_image(cv::Mat& img, std::vector<Detection>& inferResult, bool drawBbox){
    // draw inference result on image
    for (size_t i = 0; i < inferResult.size(); i++){
        // draw bboxes
        if (drawBbox){
            cv::Scalar bboxColor(get_random_int(), get_random_int(), get_random_int());
            cv::Rect r(
                round(inferResult[i].bbox[0]),
                round(inferResult[i].bbox[1]),
                round(inferResult[i].bbox[2] - inferResult[i].bbox[0]),
                round(inferResult[i].bbox[3] - inferResult[i].bbox[1])
            );
            cv::rectangle(img, r, bboxColor, 2);

            std::string className = vClassNames[(int)inferResult[i].classId];
            std::string labelStr = className + " " + std::to_string(inferResult[i].conf).substr(0, 4);

            cv::Size textSize = cv::getTextSize(labelStr, cv::FONT_HERSHEY_PLAIN, 1.2, 2, NULL);
            cv::Point topLeft(r.x, r.y - textSize.height - 3);
            cv::Point bottomRight(r.x + textSize.width, r.y);
            cv::rectangle(img, topLeft, bottomRight, bboxColor, -1);
            cv::putText(img, labelStr, cv::Point(r.x, r.y - 2), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        }

        // maskMatrix đã ở đúng kích thước ảnh gốc nên draw_mask chỉ còn nhiệm vụ tô đè.
        draw_mask(img, inferResult[i].maskMatrix.data());
    }
}
