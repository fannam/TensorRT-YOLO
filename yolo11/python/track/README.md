# YOLO11 Python ByteTrack

![street](../../../assets/result.gif)

## 运行

1. 先在 [../detect/README.md](../detect/README.md) 完成检测模型转换，确保 `yolo11/python/detect/model.plan` 已生成。
2. 在同一 Python 环境中安装额外依赖：

```bash
pip install lap
pip install cython_bbox
```

3. 在当前目录执行：

```bash
python main.py --video ../../assets/street.mp4
```

- 默认检测引擎路径为 `../detect/model.plan`
- 输出视频默认写入 `./output/result.mp4`
