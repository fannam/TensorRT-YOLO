# -*- coding:utf-8 -*-

from pathlib import Path

kGpuId = 0
kNumClass = 80
kInputH = 640
kInputW = 640
kNmsThresh = 0.45
kConfThresh = 0.25
kMaxNumOutputBbox = 1000  # assume the box outputs no more than kMaxNumOutputBbox boxes that conf >= kNmsThresh;
kNumBoxElement = 7  # left, top, right, bottom, confidence, class, keep_flag(whether drop when NMS)

TASK_DIR = Path(__file__).resolve().parent
onnx_file = str(TASK_DIR / "onnx_model" / "yolo11s.onnx")
trt_file = str(TASK_DIR / "model.plan")

# for FP16 mode
use_fp16_mode = False
# for INT8 mode
use_int8_mode = False
n_calibration = 20
cache_file = str(TASK_DIR / "int8.cache")
calibration_data_dir = str(TASK_DIR / "calibrator")  # 存放用于 int8 量化校准的图像

class_name_list = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]
