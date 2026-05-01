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

服务器 GPU 节点如果使用 V100，推荐安装服务器专用依赖：

```bash
.venv/bin/python -m pip install -r requirements-server-cu121.txt
```

Windows 本地只做演示推理时，推荐使用 CPU 版 PyTorch。若出现 `c10.dll` 或 PyTorch DLL 加载失败，执行：

```powershell
.venv\Scripts\python.exe -m pip uninstall -y torch torchvision torchaudio
.venv\Scripts\python.exe -m pip install -r requirements-windows-cpu.txt
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

服务器 Slurm GPU 节点训练：

```bash
sbatch scripts/slurm/train_yolo.sbatch
```

详细说明见：

```text
docs/server_slurm_training.md
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

桌面演示系统功能：

- 选择本地图片进行单张检测
- 选择本地视频进行检测
- 打开摄像头实时检测
- 显示正常、疑似疲劳、疲劳状态
- 疲劳状态红色报警
- 保存当前检测截图到 `runs/screenshots/`
- 自动记录检测日志到 `runs/app_logs/`

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
