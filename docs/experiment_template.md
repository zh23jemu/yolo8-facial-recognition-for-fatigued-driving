# 实验记录模板

## 实验基本信息

- 实验日期：2026-05-01
- 数据集版本：Roboflow Driver_Drowsiness_YOLO
- 模型结构：YOLOv8n
- 输入尺寸：640
- 训练轮数：50
- 批大小：16
- 优化器：AdamW
- 学习率策略：Cosine LR

## 检测模型指标

| 模型 | Precision | Recall | mAP50 | mAP50-95 | FPS | 模型大小 |
| --- | --- | --- | --- | --- | --- | --- |
| YOLOv8n | 0.6868 | 0.9114 | 0.8347 | 0.5368 | 约 14.14（本地 CPU 单图演示） | best.pt 约 6.0 MB |
| YOLOv8n + Attention |  |  |  |  |  |  |
| ONNX Runtime |  |  |  |  |  |  |

## 时序分类指标

| 模型 | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| LSTM + Attention |  |  |  |  |

## 结论

当前 YOLOv8n 基线模型已经完成训练，并能够支持本地桌面系统演示。后续如需突出题目中的注意力机制，应继续补充 YOLOv8n + Attention 对比实验。
