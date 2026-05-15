# -*- coding:utf-8 -*-

import time
from pathlib import Path

import cv2


def main():
    try:
        from .infer import YoloDetector
    except ImportError:
        from infer import YoloDetector

    task_dir = Path(__file__).resolve().parent
    image_dir = task_dir / "images"
    output_dir = task_dir / "output"

    load_start = time.time()
    yolo_infer = YoloDetector(trt_plan=str(task_dir / "model.plan"), gpu_id=0)
    load_end = time.time()
    print("Model load cost: %.4f s" % (load_end - load_start))

    output_dir.mkdir(exist_ok=True)
    for img_path in sorted(image_dir.iterdir()):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        infer_start = time.time()
        bboxes, kpts = yolo_infer.inference(img)
        infer_end = time.time()
        print("Infer image %s cost %d ms." % (img_path.name, (infer_end - infer_start) * 1000))

        YoloDetector.draw_image(img, bboxes, kpts, draw_bbox=False)
        cv2.imwrite(str(output_dir / f"_{img_path.name}"), img)

    yolo_infer.release()


if __name__ == "__main__":
    main()
