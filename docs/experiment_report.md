# YOLOv8 疲劳驾驶面部识别实验报告

## 1. 实验目的

本实验用于验证“基于 YOLOv8 与注意力机制的疲劳驾驶面部识别系统”的核心检测模块是否能够完成眼部闭合、眼部睁开和打哈欠状态识别。实验重点包括模型检测精度、推理速度、系统演示可用性和后续论文实验材料整理。

## 2. 数据集说明

本阶段使用 Roboflow 导出的 Driver_Drowsiness_YOLO 数据集，数据集已整理为 YOLOv8 格式，类别数量为 3 类。

| 数据划分 | 图片数量 | 标签数量 |
| --- | ---: | ---: |
| 训练集 | 3790 | 3790 |
| 验证集 | 542 | 542 |
| 测试集 | 1081 | 1081 |
| 合计 | 5413 | 5413 |

类别顺序如下：

| 类别编号 | 类别名称 | 中文含义 |
| ---: | --- | --- |
| 0 | Eyeclosed | 闭眼 |
| 1 | Eyeopen | 睁眼 |
| 2 | Yawn | 打哈欠 |

数据集配置文件为 `configs/yolo_data.yaml`。训练时必须保持类别顺序不变，否则会造成睁眼、闭眼等标签错位。

## 3. 实验环境

服务器训练环境如下：

| 项目 | 配置 |
| --- | --- |
| 调度系统 | Slurm |
| GPU 节点 | plcyf-com-prod-gpu001 |
| GPU | Tesla V100-PCIE-32GB |
| Python | 3.11.11 |
| PyTorch | 2.5.1+cu121 |
| CUDA 可用性 | True |
| YOLO 框架 | Ultralytics 8.4.45 |

本地演示环境如下：

| 项目 | 配置 |
| --- | --- |
| 操作系统 | Windows |
| Python | 3.11 |
| PyTorch | 2.5.1+cpu |
| 推理方式 | CPU 推理 |
| 界面框架 | PyQt5 |

## 4. 训练参数

正式训练使用 Slurm GPU 节点完成，训练脚本为 `scripts/slurm/train_yolo.sbatch`。

| 参数 | 取值 |
| --- | --- |
| 基础模型 | YOLOv8n |
| 输入尺寸 | 640 |
| 训练轮数 | 50 |
| batch size | 16 |
| workers | 4 |
| 优化器 | AdamW |
| 学习率策略 | Cosine LR |
| 数据增强 | Mosaic、MixUp、HSV 色彩扰动 |
| 输出目录 | `runs/yolo/fatigue_yolov8n_slurm` |

## 5. 实验结果

第 50 轮验证集整体指标如下：

| 指标 | 数值 |
| --- | ---: |
| Precision | 0.6868 |
| Recall | 0.9114 |
| mAP50 | 0.8347 |
| mAP50-95 | 0.5368 |

训练完成后的最终验证日志中，模型在融合后结构下的推理速度约为：

| 阶段 | 单图耗时 |
| --- | ---: |
| preprocess | 0.1 ms |
| inference | 0.9 ms |
| postprocess | 1.0 ms |

短跑测试阶段曾得到如下类别指标，可作为模型可行性的辅助说明：

| 类别 | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 542 | 578 | 0.687 | 0.908 | 0.837 | 0.539 |
| Eyeclosed | 278 | 353 | 0.865 | 0.940 | 0.968 | 0.586 |
| Eyeopen | 107 | 144 | 0.658 | 0.861 | 0.849 | 0.530 |
| Yawn | 81 | 81 | 0.539 | 0.923 | 0.693 | 0.502 |

实验图文件：

- 混淆矩阵：`runs/yolo/fatigue_yolov8n_slurm/confusion_matrix.png`
- PR 曲线：`runs/yolo/fatigue_yolov8n_slurm/BoxPR_curve.png`
- F1 曲线：`runs/yolo/fatigue_yolov8n_slurm/BoxF1_curve.png`
- 训练日志：`runs/yolo/fatigue_yolov8n_slurm/results.csv`

模型文件：

- PyTorch 权重：`weights/best.pt`
- ONNX 权重：`weights/best.onnx`

## 6. 系统演示测试

本地 PyQt5 桌面系统已完成基础演示闭环，支持选择图片/视频、摄像头检测、保存截图和生成检测日志。当前测试样例为单张打哈欠图片，系统检测结果如下：

| 项目 | 结果 |
| --- | --- |
| 检测目标 | Yawn |
| 置信度 | 0.95 |
| 疲劳状态 | 疑似疲劳 |
| 闭眼比例 | 0.0 |
| 打哈欠次数 | 1 |
| 本地 CPU 推理 FPS | 14.14 |
| 检测日志 | `runs/app_logs/detection_20260501_105950.csv` |

该结果说明系统能够在本地 CPU 环境下加载服务器训练得到的模型，并完成图像推理、状态判断、界面显示和日志记录。

## 7. 阶段结论

当前阶段已经完成目标检测模型训练和系统演示闭环。模型在验证集上达到 mAP50 约 0.835，能够较好识别闭眼、睁眼和打哈欠三类疲劳相关面部状态。其中闭眼类别表现最好，打哈欠类别精度相对较低，后续可通过补充打哈欠样本、调整数据增强策略或增加训练轮数进一步优化。

从工程实现角度看，系统已经具备毕业设计演示所需的核心功能，包括模型加载、图像/视频输入、疲劳状态显示、报警状态标识、截图保存和日志记录。

## 8. 后续训练建议

如果论文题目需要突出“注意力机制”改进效果，建议继续补充至少一组对比实验：

1. **YOLOv8n 基线模型**：当前已完成，可作为 baseline。
2. **YOLOv8n + 注意力机制模型**：采用 YOLOv8n + CBAM，并在相同数据集、相同训练轮数下训练。
3. **对比指标**：Precision、Recall、mAP50、mAP50-95、模型大小、推理速度。

已新增训练脚本：

- 短跑测试：`scripts/slurm/train_yolo_cbam_test.sbatch`
- 正式训练：`scripts/slurm/train_yolo_cbam.sbatch`

建议先提交短跑测试，确认自定义模型配置可训练后，再提交正式训练。
