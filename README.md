# 基于 YOLOv8 与注意力机制的疲劳驾驶面部识别系统

本项目用于毕业设计，目标是搭建一个可训练、可推理、可演示的疲劳驾驶面部识别系统。系统采用 YOLOv8 检测眼部和嘴部疲劳特征，并预留 LSTM + Attention 时序疲劳分类模块。

## 包内内容

本次发给用户的压缩包建议包含：

- `src/`：程序源码
- `configs/`：数据集与模型配置
- `weights/`：训练好的模型权重
- `data.zip`：数据集压缩包
- `requirements.txt`、`requirements-server-cu121.txt`、`requirements-windows-cpu.txt`：依赖说明
- `README.md`：项目使用说明

## 技术栈

- Python 3.10/3.11
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- PyQt5
- ONNX Runtime
- LSTM + Attention

## 环境建议

- Windows 10/11 或 Linux
- Python 3.11
- 如需训练，建议使用带 NVIDIA GPU 的服务器
- 如需本地演示，建议安装摄像头并使用 CPU 版本依赖

## 环境初始化

请始终使用项目本地虚拟环境，不要直接使用系统 Python。

### Windows

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements-windows-cpu.txt
```

### Linux / 服务器 GPU

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-server-cu121.txt
```

如果本机没有 Python 3.11，也可以使用已安装且兼容 PyTorch 的 Python 3.10。

## 数据集放置

大型数据集不提交到仓库。数据集已单独打包为 `data.zip`，解压后建议保持如下结构：

```text
data/
  fatigue_yolo/
    train/
    valid/
    test/
  fatigue_sequence/
```

其中 `data/fatigue_yolo/` 是 Roboflow 导出的 YOLOv8 数据集，`data/fatigue_sequence/` 用于后续连续帧时序实验。

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

## 使用方法

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

### 启动桌面演示系统

```powershell
.venv\Scripts\python.exe -m src.app.main_window
```

桌面演示系统支持：

- 选择本地图片进行单张检测
- 选择本地视频进行检测
- 打开摄像头实时检测
- 显示正常、疑似疲劳、疲劳状态
- 疲劳状态红色报警
- 保存当前检测截图到 `runs/screenshots/`
- 自动记录检测日志到 `runs/app_logs/`

## 训练方法

### 本地训练

```powershell
.venv\Scripts\python.exe -m src.train.train_yolo --data configs/yolo_data.yaml --model yolov8n.pt --epochs 50
```

### 服务器 Slurm GPU 节点训练

```bash
sbatch scripts/slurm/train_yolo.sbatch
```

详细说明见 `docs/server_slurm_training.md`。

## 常见问题

### 找不到数据集

请确认 `data/fatigue_yolo/` 已正确解压，且 `configs/yolo_data.yaml` 中的数据路径与实际目录一致。

### 模型权重缺失

请确认 `weights/best.pt` 或 `weights/best_cbam.pt` 文件存在。

### 摄像头打不开

请检查是否有其他软件占用摄像头，或者改用图片和视频测试。

### Windows 下 PyTorch 报 DLL 错误

可改用 CPU 版依赖：

```powershell
.venv\Scripts\python.exe -m pip uninstall -y torch torchvision torchaudio
.venv\Scripts\python.exe -m pip install -r requirements-windows-cpu.txt
```

## 第一轮成果

- 项目目录结构
- YOLOv8 训练入口
- 图片/视频/摄像头推理入口
- LSTM + Attention 时序模型骨架
- PyQt5 桌面演示原型
- 数据集、实验记录和运行说明模板

## 文档材料

- `docs/materials_index.md`：毕业设计材料总索引
- `docs/thesis_chapter_materials.md`：论文第 4、5、6 章写作素材
- `docs/experiment_report.md`：YOLOv8n 与 YOLOv8n + CBAM 实验报告
- `docs/system_test_report.md`：桌面系统测试记录
- `docs/defense_ppt_outline.md`：答辩 PPT 提纲
- `docs/demo_guide.md`：系统演示与复现实验指南
- `docs/defense_presentation.pptx`：答辩 PPT 初稿
- `docs/defense_presentation_preview.png`：答辩 PPT 总览预览图
- `docs/thesis_draft.md`：论文正文 Markdown 初稿
- `docs/thesis_draft.docx`：论文正文 Word 草稿
