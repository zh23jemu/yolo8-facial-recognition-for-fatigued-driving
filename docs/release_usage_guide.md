# 疲劳驾驶面部识别系统使用说明

## 1. 包内内容

本压缩包包含以下内容：

- `src/`：程序源码
- `configs/`：数据集与模型配置
- `weights/`：训练好的模型权重
- `data.zip`：数据集压缩包
- `requirements.txt`、`requirements-server-cu121.txt`、`requirements-windows-cpu.txt`：依赖说明
- `README.md`：项目概述

## 2. 环境建议

- Windows 10/11 或 Linux
- Python 3.11
- 如需训练，建议使用带 NVIDIA GPU 的服务器
- 如需本地演示，建议安装摄像头并使用 CPU 版本依赖

## 3. 数据集说明

数据集已打包为 `data.zip`。解压后应保持如下结构：

```text
data/
  fatigue_yolo/
    train/
    valid/
    test/
```

如果你只是做推理演示，也可以直接使用包内的 `weights/` 目录中的模型文件，无需重新训练。

## 4. 安装依赖

建议先创建虚拟环境，再安装依赖。

### Windows

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-windows-cpu.txt
```

### Linux / 服务器 GPU

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-server-cu121.txt
```

## 5. 推理运行

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

## 6. 桌面程序

直接运行：

```powershell
.venv\Scripts\python.exe -m src.app.main_window
```

界面支持：

- 选择图片
- 打开摄像头
- 开始检测
- 停止检测
- 显示疲劳状态
- 保存截图

## 7. 训练说明

如果需要重新训练，请优先在服务器 GPU 节点上运行 Slurm 脚本：

```bash
bash scripts/slurm/train_yolo.sbatch
```

训练配置默认读取 `configs/yolo_data.yaml`。

## 8. 常见问题

### 找不到数据集

请确认 `data/fatigue_yolo/` 已正确解压，且 `configs/yolo_data.yaml` 中的数据路径与实际目录一致。

### 模型权重缺失

请确认 `weights/best.pt` 或 `weights/best_cbam.pt` 文件存在。

### 摄像头打不开

请检查是否有其他软件占用摄像头，或者改用图片/视频测试。

## 9. 备注

本项目当前用于毕业设计演示与实验展示。如需实际部署到车载环境，建议进一步补充夜间、逆光、遮挡和侧脸样本，并结合连续帧时序模型进一步优化。
