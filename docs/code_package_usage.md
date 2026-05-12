# 疲劳驾驶面部识别系统代码包说明

## 1. 程序介绍

本程序是一个基于 YOLOv8 与注意力机制的疲劳驾驶面部识别系统，主要用于识别驾驶员面部图像中的疲劳相关特征。系统以闭眼、睁眼和打哈欠为核心检测目标，通过目标检测模型输出检测框、类别和置信度，并结合疲劳状态规则给出正常、疑似疲劳或疲劳提示。

项目围绕“模型训练 → 推理检测 → 可视化演示”的流程设计，既可以在服务器上训练模型，也可以在本地运行图片、视频或摄像头推理，并通过 PyQt5 桌面界面展示检测结果。

## 2. 代码包内容

本代码包主要包含以下内容：

- `src/`：项目核心源码
- `configs/`：数据集配置和模型结构配置
- `data/`：疲劳驾驶 YOLO 格式数据集
- `scripts/`：服务器 Slurm 训练脚本
- `requirements.txt`：基础依赖
- `requirements-windows-cpu.txt`：Windows CPU 推理依赖
- `requirements-server-cu121.txt`：服务器 CUDA 12.1 训练依赖
- `README.md`：项目整体说明

本代码包包含数据集，但不包含虚拟环境、训练输出和完整实验日志。如需直接运行推理演示，还需要准备模型权重文件。

## 3. 目录结构

```text
src/
  app/       # PyQt5 桌面演示程序
  infer/     # 图片、视频、摄像头推理入口
  models/    # LSTM + Attention 时序模型骨架
  train/     # YOLOv8 训练入口
  utils/     # 疲劳判定规则和 Ultralytics 兼容工具

configs/
  yolo_data.yaml       # YOLO 数据集配置
  yolov8n_cbam.yaml    # 加入 CBAM 注意力模块的 YOLOv8n 配置

data/
  fatigue_yolo/        # YOLO 格式疲劳驾驶数据集
  fatigue_sequence/    # 时序特征数据预留目录

scripts/
  slurm/               # GPU 服务器训练脚本
```

## 4. 环境安装

建议使用 Python 3.11，并创建项目本地虚拟环境。

### Windows 本地演示

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements-windows-cpu.txt
```

### Linux / GPU 服务器训练

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-server-cu121.txt
```

## 5. 准备数据集和权重

### 数据集

本代码包已包含训练数据，目录如下：

```text
data/fatigue_yolo/
  train/images
  train/labels
  valid/images
  valid/labels
  test/images
  test/labels
```

如果解压后数据路径发生变化，需要同步修改 `configs/yolo_data.yaml`。

### 模型权重

推理演示需要准备训练好的权重文件，例如：

```text
weights/best.pt
weights/best_cbam.pt
```

其中 `best.pt` 为 YOLOv8n 基线模型权重，`best_cbam.pt` 为加入 CBAM 注意力机制后的模型权重。

## 6. 使用方法

### 图片推理

```powershell
.venv\Scripts\python.exe -m src.infer.run_infer --weights weights\best.pt --source path\to\image.jpg
```

### 视频推理

```powershell
.venv\Scripts\python.exe -m src.infer.run_infer --weights weights\best.pt --source path\to\video.mp4
```

### 摄像头推理

```powershell
.venv\Scripts\python.exe -m src.infer.run_infer --weights weights\best.pt --source 0
```

### 启动桌面演示程序

```powershell
.venv\Scripts\python.exe -m src.app.main_window
```

桌面程序支持选择图片、选择视频、打开摄像头、开始检测、停止检测、显示疲劳状态、保存截图和记录检测日志。

## 7. 训练方法

### 本地训练

```powershell
.venv\Scripts\python.exe -m src.train.train_yolo --data configs\yolo_data.yaml --model yolov8n.pt --epochs 50
```

### Slurm GPU 服务器训练

```bash
sbatch scripts/slurm/train_yolo.sbatch
```

如果要训练注意力机制模型，可使用：

```bash
sbatch scripts/slurm/train_yolo_cbam.sbatch
```

## 8. 常见问题

### 找不到数据集

检查 `data/fatigue_yolo/` 是否存在，并确认 `configs/yolo_data.yaml` 中的路径是否正确。

### 找不到模型权重

检查 `weights/` 目录下是否存在 `best.pt` 或 `best_cbam.pt`。如果没有权重，需要先完成训练或从已有实验结果中复制权重。

### 摄像头打不开

请确认摄像头没有被其他软件占用。若本机没有摄像头，可以改用图片或视频进行测试。

### Windows 下 PyTorch 报 DLL 错误

建议安装 Windows CPU 版本依赖：

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-windows-cpu.txt
```

## 9. 说明

本代码包主要用于项目交付、程序查看、训练复现和二次开发。若要直接复现已有推理效果，还需要配套模型权重和训练结果文件。
