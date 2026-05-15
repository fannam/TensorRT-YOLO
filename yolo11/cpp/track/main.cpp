#include <iostream>
#include <string>
#include <filesystem>
#include "infer.h"
#include "BYTETracker.h"

// Binary track ghép detector YOLO11 detect với shared ByteTrack:
// video frame -> detect -> lọc class -> BYTETracker.update() -> draw track id -> ghi video output.

// Chỉ track một tập con class COCO để demo dễ nhìn và giảm nhiễu ID switch cho object ít quan tâm.
std::vector<int>  trackClasses {0, 1, 2, 3, 5, 7};  // person, bicycle, car, motorcycle, bus, truck

bool isTrackingClass(int class_id){
    for (auto& c : trackClasses){
        if (class_id == c) return true;
    }
    return false;
}

int run(const std::filesystem::path& executablePath, char* videoPath){
    std::string inputVideoPath = std::string(videoPath);
    cv::VideoCapture cap(inputVideoPath);
    if ( !cap.isOpened() ) return 0;

    int img_w = cap.get(CAP_PROP_FRAME_WIDTH);
    int img_h = cap.get(CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(CAP_PROP_FPS);
    long nFrame = static_cast<long>(cap.get(CAP_PROP_FRAME_COUNT));
    cout << "Total frames: " << nFrame << endl;

    std::filesystem::path exeDir = std::filesystem::absolute(executablePath).parent_path();
    std::filesystem::path outputPath = exeDir / "../output/result.mp4";
    std::filesystem::create_directories(outputPath.parent_path());

    cv::VideoWriter writer(outputPath.string(), VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(img_w, img_h));

    // Tracker sample dùng detector detect thường, không dùng head tracking chuyên biệt.
    // Detector để confThresh thấp hơn detect demo vì ByteTrack muốn tiêu thụ cả box score thấp.
    std::string trtFile = (exeDir / "../../detect/build/yolo11s.plan").lexically_normal().string();
    std::string onnxFile = (exeDir / "../../detect/onnx_model/yolo11s.onnx").lexically_normal().string();
    YoloDetector detector(trtFile, onnxFile, 0, 0.45, 0.01);

    // ByteTrack dùng fps và track_buffer để quyết định một track được phép "mất tích" bao lâu.
    BYTETracker tracker(fps, 30);

    cv::Mat img;
    int num_frames = 0;
    int total_ms = 0;
    while (true){
        if ( !cap.read(img) ) break;
        num_frames++;
        if (num_frames % 20 == 0){
            cout << "Processing frame " << num_frames << " (" << num_frames * 1000000 / total_ms << " fps)" << endl;
        }
        if (img.empty()) break;

        auto start = std::chrono::system_clock::now();

        std::vector<Detection> res = detector.inference(img);

        // Chuyển output detector sang Object mà ByteTrack hiểu:
        // rect tlwh + label + score. Đồng thời lọc class không muốn track.
        std::vector<Object> objects;
        for (size_t j = 0; j < res.size(); j++){
            float* bbox = res[j].bbox;
            float conf = res[j].conf;
            int classId = res[j].classId;

            if (isTrackingClass(classId)){
                cv::Rect_<float> rect(bbox[0], bbox[1], (bbox[2] - bbox[0]), (bbox[3] - bbox[1]));
                Object obj {rect, classId, conf};
                objects.push_back(obj);
            }
        }

        // update() thực hiện toàn bộ state machine Tracked/Lost/Removed và trả về track đang hoạt động.
        std::vector<STrack> output_stracks = tracker.update(objects);

        auto end = std::chrono::system_clock::now();
        total_ms = total_ms + std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        for (int i = 0; i < output_stracks.size(); i++)
        {
            std::vector<float> tlwh = output_stracks[i].tlwh;
            if (tlwh[2] * tlwh[3] > 20)
            {
                cv::Scalar s = tracker.get_color(output_stracks[i].track_id);
                cv::putText(img, cv::format("%d", output_stracks[i].track_id), cv::Point(tlwh[0], tlwh[1] - 5),
                        0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                cv::rectangle(img, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
            }
        }
        cv::putText(img, cv::format("frame: %d fps: %d num: %ld", num_frames, num_frames * 1000000 / total_ms, output_stracks.size()),
                cv::Point(0, 30), 0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        writer.write(img);
    }

    cap.release();
    std::cout << "FPS: " << num_frames * 1000000 / total_ms << std::endl;

    return 0;
}

int main(int argc, char* argv[]){
    if (argc != 2 )
    {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "Usage: ./trt_yolo11_track_cli [video path]" << std::endl;
        std::cerr << "Example: ./trt_yolo11_track_cli ../../assets/street.mp4" << std::endl;
        return -1;
    }

    return run(argv[0], argv[1]);
}
