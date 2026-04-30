# 基于 YOLOv8 与注意力机制的疲劳驾驶面部识别系统

本项目用于毕业设计第一轮开发落地，目标是搭建一个可训练、可推理、可演示的疲劳驾驶面部识别系统。系统采用 YOLOv8 检测眼部/嘴部疲劳特征，并预留 LSTM + Attention 时序疲劳分类模块。

## 技术栈

- Python 3.10/3.11
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- PyQt5
- ONNX Runtime
- LSTM + Attention

## 环境初始化

请始终使用项目本地虚拟环境，不要直接使用系统 Python。

Windows 示例：

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果本机没有 Python 3.11，可使用已安装且兼容 PyTorch 的 Python 3.10：

```powershell
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 数据集放置

大型数据集不提交到仓库。推荐目录：

```text
data/
  fatigue_yolo/       # Roboflow 导出的 YOLOv8 数据集
  fatigue_sequence/   # UTA-RLDD 抽帧后生成的时序特征数据
```

YOLOv8 数据集目录需包含：

```text
data/fatigue_yolo/
  train/images
  train/labels
  valid/images
  valid/labels
  test/images
  test/labels
```

## 常用命令

训练 YOLOv8：

```powershell
.venv\Scripts\python.exe -m src.train.train_yolo --data configs/yolo_data.yaml --model yolov8n.pt --epochs 50
```

图片/视频/摄像头推理：

```powershell
.venv\Scripts\python.exe -m src.infer.run_infer --weights weights/best.pt --source path/to/image.jpg
.venv\Scripts\python.exe -m src.infer.run_infer --weights weights/best.pt --source path/to/video.mp4
.venv\Scripts\python.exe -m src.infer.run_infer --weights weights/best.pt --source 0
```

启动桌面演示系统：

```powershell
.venv\Scripts\python.exe -m src.app.main_window --weights weights/best.pt
```

## 第一轮成果

- 项目目录结构
- YOLOv8 训练入口
- 图片/视频/摄像头推理入口
- LSTM + Attention 时序模型骨架
- PyQt5 桌面演示原型
- 数据集、实验记录和运行说明模板

