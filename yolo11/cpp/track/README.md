# YOLO11 cpp ByteTrack

![street](../../../assets/result.gif)

## 运行

1. 先在 [../detect/README.md](../detect/README.md) 完成检测模型转换，确保 `yolo11/cpp/detect/build/yolo11s.plan` 已生成。
2. 安装额外依赖：

```bash
apt install libeigen3-dev
```

3. 在当前目录编译并运行：

```bash
mkdir -p build
cd build
cmake ..
make
./trt_yolo11_track_cli ../../assets/street.mp4
```

- 检测引擎默认从同一 model family 下的 `../detect/build/yolo11s.plan` 自动解析
- 输出视频写入 `../output/result.mp4`
