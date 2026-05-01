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

正式训练使用 Slurm GPU 节点完成。基线模型训练脚本为 `scripts/slurm/train_yolo.sbatch`，注意力机制模型训练脚本为 `scripts/slurm/train_yolo_cbam.sbatch`。

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

### 5.1 YOLOv8n 基线模型

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

### 5.2 YOLOv8n + CBAM 注意力机制模型

为体现题目中的注意力机制，本研究在 YOLOv8n 主干网络末端加入 CBAM 模块。CBAM 同时包含通道注意力和空间注意力，有助于增强模型对闭眼、睁眼和打哈欠等局部疲劳特征的关注能力。

CBAM 模型结构与训练设置：

| 项目 | 取值 |
| --- | --- |
| 模型配置 | `configs/yolov8n_cbam.yaml` |
| 注意力模块 | CBAM |
| 注意力位置 | 主干网络 SPPF 后 |
| 参数量 | 3,077,323 |
| GFLOPs | 8.2 |
| 训练轮数 | 50 |
| batch size | 16 |
| 预训练迁移 | 从 `yolov8n.pt` 迁移初始化 |
| 输出目录 | `runs/yolo/fatigue_yolov8n_cbam` |

CBAM 模型最终验证指标如下：

| 类别 | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 542 | 578 | 0.684 | 0.937 | 0.858 | 0.547 |
| Eyeclosed | 278 | 353 | 0.871 | 0.946 | 0.973 | 0.574 |
| Eyeopen | 107 | 144 | 0.657 | 0.889 | 0.839 | 0.531 |
| Yawn | 81 | 81 | 0.524 | 0.975 | 0.762 | 0.535 |

CBAM 模型推理速度如下：

| 阶段 | 单图耗时 |
| --- | ---: |
| preprocess | 0.1 ms |
| inference | 1.1 ms |
| postprocess | 0.6 ms |

CBAM 实验图文件：

- 混淆矩阵：`runs/yolo/fatigue_yolov8n_cbam/confusion_matrix.png`
- PR 曲线：`runs/yolo/fatigue_yolov8n_cbam/BoxPR_curve.png`
- F1 曲线：`runs/yolo/fatigue_yolov8n_cbam/BoxF1_curve.png`
- 训练日志：`runs/yolo/fatigue_yolov8n_cbam/results.csv`

### 5.3 对比分析

| 模型 | Precision | Recall | mAP50 | mAP50-95 | 参数量 | 推理耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n | 0.6868 | 0.9114 | 0.8347 | 0.5368 | 约 3.01M | 0.9 ms |
| YOLOv8n + CBAM | 0.6840 | 0.9370 | 0.8580 | 0.5470 | 约 3.08M | 1.1 ms |

与 YOLOv8n 基线模型相比，YOLOv8n + CBAM 的 mAP50 提升约 0.0233，mAP50-95 提升约 0.0102，Recall 提升约 0.0256。CBAM 模型参数量略有增加，推理耗时从约 0.9 ms 增加到约 1.1 ms，但整体仍满足实时检测需求。

从类别结果看，CBAM 对打哈欠类别提升较明显，Yawn 类别 mAP50 从约 0.693 提升到 0.762，mAP50-95 从约 0.502 提升到 0.535。这说明注意力机制能够增强模型对嘴部张开等局部疲劳特征的关注能力。

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

当前阶段已经完成目标检测模型训练、注意力机制对比实验和系统演示闭环。YOLOv8n 基线模型在验证集上达到 mAP50 约 0.835，YOLOv8n + CBAM 模型达到 mAP50 约 0.858。对比结果表明，引入注意力机制后模型整体检测精度有所提升，尤其对打哈欠类别的识别能力提升更明显。

从工程实现角度看，系统已经具备毕业设计演示所需的核心功能，包括模型加载、图像/视频输入、疲劳状态显示、报警状态标识、截图保存和日志记录。

## 8. 后续工作建议

当前 YOLOv8n 与 YOLOv8n + CBAM 的核心对比实验已经完成。后续若时间允许，可继续开展视频序列层面的 LSTM + Attention 疲劳状态分类实验；若时间紧张，可优先围绕当前目标检测模型和桌面系统完成论文撰写与答辩材料制作。
