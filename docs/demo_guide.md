# 系统演示与复现实验指南

本文档用于答辩前快速复现模型推理、桌面演示和训练结果查看。

## 1. 本地环境检查

在 Windows 本地项目根目录执行：

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

当前本地演示环境使用 CPU 版 PyTorch，预期输出类似：

```text
2.5.1+cpu
False
```

如果出现 `c10.dll` 或 PyTorch DLL 加载错误，执行：

```powershell
.venv\Scripts\python.exe -m pip uninstall -y torch torchvision torchaudio
.venv\Scripts\python.exe -m pip install -r requirements-windows-cpu.txt
```

## 2. 模型文件

当前项目保留两组模型：

| 模型 | 权重路径 | 说明 |
| --- | --- | --- |
| YOLOv8n | `weights/best.pt` | 基线模型 |
| YOLOv8n + CBAM | `weights/best_cbam.pt` | 注意力机制模型 |
| YOLOv8n ONNX | `weights/best.onnx` | 基线 ONNX 模型 |
| YOLOv8n + CBAM ONNX | `weights/best_cbam.onnx` | 注意力模型 ONNX |

答辩演示优先使用 `weights/best_cbam.pt`，因为它对应注意力机制改进模型。

## 3. 启动桌面演示系统

使用 CBAM 模型启动：

```powershell
.venv\Scripts\python.exe -m src.app.main_window --weights weights/best_cbam.pt
```

使用基线模型启动：

```powershell
.venv\Scripts\python.exe -m src.app.main_window --weights weights/best.pt
```

界面操作流程：

1. 点击“选择图片/视频”。
2. 选择测试集图片，例如 `data\fatigue_yolo\test\images\0_jpg.rf.8f6a5837638646b3083c01fd0938d5f6.jpg`。
3. 点击“开始检测”。
4. 查看检测框、类别、置信度和疲劳状态。
5. 点击“保存截图”，保存答辩用截图。

## 4. 命令行图片推理

基线模型推理：

```powershell
.venv\Scripts\python.exe -m src.infer.run_infer --weights weights\best.pt --source data\fatigue_yolo\test\images\0_jpg.rf.8f6a5837638646b3083c01fd0938d5f6.jpg --save runs\baseline_demo.jpg
```

CBAM 模型推理：

```powershell
.venv\Scripts\python.exe -m src.infer.run_infer --weights weights\best_cbam.pt --source data\fatigue_yolo\test\images\0_jpg.rf.8f6a5837638646b3083c01fd0938d5f6.jpg --save runs\cbam_demo.jpg
```

## 5. 训练结果位置

基线训练结果：

```text
runs/yolo/fatigue_yolov8n_slurm/
```

CBAM 训练结果：

```text
runs/yolo/fatigue_yolov8n_cbam/
```

重点查看文件：

- `results.csv`
- `confusion_matrix.png`
- `BoxPR_curve.png`
- `BoxF1_curve.png`

## 6. 论文和 PPT 推荐引用材料

| 材料 | 路径 |
| --- | --- |
| 实验报告 | `docs/experiment_report.md` |
| 系统测试记录 | `docs/system_test_report.md` |
| 论文章节素材 | `docs/thesis_chapter_materials.md` |
| 答辩 PPT 提纲 | `docs/defense_ppt_outline.md` |
| 系统演示截图 1 | `docs/images/system_demo_yawn.jpg` |
| 系统演示截图 2 | `docs/images/system_demo_second.jpg` |

## 7. 服务器训练复现

短跑测试：

```bash
sbatch scripts/slurm/train_yolo_cbam_test.sbatch
```

正式训练：

```bash
sbatch scripts/slurm/train_yolo_cbam.sbatch
```

服务器训练完成后，如需同步产物到 Git：

```bash
cp runs/yolo/fatigue_yolov8n_cbam/weights/best.pt weights/best_cbam.pt
cp runs/yolo/fatigue_yolov8n_cbam/weights/best.onnx weights/best_cbam.onnx
git add weights/best_cbam.pt weights/best_cbam.onnx runs/yolo/fatigue_yolov8n_cbam/results.csv runs/yolo/fatigue_yolov8n_cbam/confusion_matrix.png runs/yolo/fatigue_yolov8n_cbam/BoxPR_curve.png runs/yolo/fatigue_yolov8n_cbam/BoxF1_curve.png
git commit -m "feat: 添加CBAM模型和实验结果

同步YOLOv8n + CBAM训练权重、ONNX模型和核心实验曲线。
用于论文注意力机制对比实验和本地演示复现。"
git push origin master
```

