# 疲劳驾驶面部识别系统

## 项目说明

本压缩包用于向最终用户交付疲劳驾驶面部识别程序及训练好的模型文件。系统基于 YOLOv8 实现对闭眼、睁眼和打哈欠三类疲劳相关面部特征的检测，并支持图片、视频和摄像头输入方式。

## 技术栈

- Python 3.11
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- PyQt5
- ONNX Runtime

## 包内结构

- `src/`：程序源码
- `configs/`：模型与数据配置文件
- `scripts/slurm/`：服务器训练脚本
- `weights/`：训练好的模型权重与 ONNX 文件
- `requirements.txt`：基础依赖
- `requirements-windows-cpu.txt`：Windows 本地演示依赖
- `requirements-server-cu121.txt`：服务器 GPU 训练依赖
- `README.md`：本说明文档

## 环境要求

- Windows 10/11 或 Linux
- Python 3.11
- 如需本地演示，推荐使用 Windows 并安装摄像头
- 如需重新训练，推荐使用带 NVIDIA GPU 的服务器环境

## 环境初始化

请使用项目本地虚拟环境运行程序，不要直接使用系统 Python。

### Windows

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements-windows-cpu.txt
```

### Linux / GPU 服务器

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-server-cu121.txt
```

## 模型文件说明

- `weights/best.pt`：YOLOv8n 基线模型权重
- `weights/best_cbam.pt`：加入 CBAM 注意力机制后的改进模型权重
- `weights/best.onnx`：基线模型 ONNX 版本
- `weights/best_cbam.onnx`：改进模型 ONNX 版本

如果只是进行推理演示，优先直接使用 `weights/best_cbam.pt`。

## 启动方法

### 图片推理

```powershell
.venv\Scripts\python.exe -m src.infer.run_infer --weights weights\best_cbam.pt --source path\to\image.jpg
```

### 视频推理

```powershell
.venv\Scripts\python.exe -m src.infer.run_infer --weights weights\best_cbam.pt --source path\to\video.mp4
```

### 摄像头推理

```powershell
.venv\Scripts\python.exe -m src.infer.run_infer --weights weights\best_cbam.pt --source 0
```

### 启动桌面演示程序

```powershell
.venv\Scripts\python.exe -m src.app.main_window
```

## 训练方法

如需重新训练，可使用以下方式：

### 本地训练

```powershell
.venv\Scripts\python.exe -m src.train.train_yolo --data configs/yolo_data.yaml --model yolov8n.pt --epochs 50
```

### 服务器 Slurm 训练

```bash
sbatch scripts/slurm/train_yolo.sbatch
```

如果训练改进模型，可使用 `scripts/slurm/train_yolo_cbam.sbatch`。

## 使用注意事项

- 程序首次运行前请确认依赖已经安装完成。
- 如不使用摄像头，可直接通过图片或视频测试推理功能。
- 运行结果截图默认保存在 `runs/screenshots/`。
- 检测日志默认保存在 `runs/app_logs/`。

## 常见问题

### 权重文件找不到

请确认 `weights/` 目录中的模型文件存在，并在启动命令中正确填写权重路径。

### 摄像头无法打开

请检查是否有其他程序占用了摄像头，或改用图片、视频进行测试。

### Windows 下依赖报错

请优先使用 `requirements-windows-cpu.txt` 安装本地演示依赖。
