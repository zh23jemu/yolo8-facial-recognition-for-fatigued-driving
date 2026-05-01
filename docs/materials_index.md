# 毕业设计材料索引

本文档汇总当前项目中可用于论文、答辩 PPT 和系统演示的全部关键材料。

## 1. 代码与配置

| 材料 | 路径 | 用途 |
| --- | --- | --- |
| YOLO 数据配置 | `configs/yolo_data.yaml` | 训练数据集路径与类别定义 |
| CBAM 模型配置 | `configs/yolov8n_cbam.yaml` | 注意力机制模型结构 |
| 训练入口 | `src/train/train_yolo.py` | 训练基线和 CBAM 模型 |
| 推理入口 | `src/infer/run_infer.py` | 图片、视频、摄像头推理 |
| 桌面系统 | `src/app/main_window.py` | PyQt5 演示系统 |
| 疲劳规则 | `src/utils/fatigue_rules.py` | 疲劳状态判定 |
| Ultralytics 补丁 | `src/utils/ultralytics_patches.py` | 注册 CBAM 注意力模块 |

## 2. 模型文件

| 材料 | 路径 | 用途 |
| --- | --- | --- |
| 基线 PyTorch 权重 | `weights/best.pt` | YOLOv8n 推理演示 |
| 基线 ONNX 权重 | `weights/best.onnx` | ONNX 部署实验 |
| CBAM PyTorch 权重 | `weights/best_cbam.pt` | 注意力模型推理演示 |
| CBAM ONNX 权重 | `weights/best_cbam.onnx` | 注意力模型 ONNX 部署实验 |

## 3. 实验结果

| 材料 | 路径 |
| --- | --- |
| 基线训练日志 | `runs/yolo/fatigue_yolov8n_slurm/results.csv` |
| 基线混淆矩阵 | `runs/yolo/fatigue_yolov8n_slurm/confusion_matrix.png` |
| 基线 PR 曲线 | `runs/yolo/fatigue_yolov8n_slurm/BoxPR_curve.png` |
| 基线 F1 曲线 | `runs/yolo/fatigue_yolov8n_slurm/BoxF1_curve.png` |
| CBAM 训练日志 | `runs/yolo/fatigue_yolov8n_cbam/results.csv` |
| CBAM 混淆矩阵 | `runs/yolo/fatigue_yolov8n_cbam/confusion_matrix.png` |
| CBAM PR 曲线 | `runs/yolo/fatigue_yolov8n_cbam/BoxPR_curve.png` |
| CBAM F1 曲线 | `runs/yolo/fatigue_yolov8n_cbam/BoxF1_curve.png` |

## 4. 文档材料

| 文档 | 路径 | 用途 |
| --- | --- | --- |
| 数据集说明 | `docs/dataset_notes.md` | 写论文数据集部分 |
| 实验报告 | `docs/experiment_report.md` | 写实验章节 |
| 系统测试记录 | `docs/system_test_report.md` | 写测试章节 |
| 论文章节素材 | `docs/thesis_chapter_materials.md` | 写第 4、5、6 章 |
| 论文正文初稿 | `docs/thesis_draft.md` | Markdown 版正文，便于继续扩写 |
| 论文 Word 草稿 | `docs/thesis_draft.docx` | Word 版正文，便于套学校模板 |
| 答辩 PPT 提纲 | `docs/defense_ppt_outline.md` | 制作答辩 PPT |
| 答辩 PPT 初稿 | `docs/defense_presentation.pptx` | 答辩展示与后续精修 |
| PPT 总览预览图 | `docs/defense_presentation_preview.png` | 快速检查 PPT 页面布局 |
| 演示指南 | `docs/demo_guide.md` | 答辩前复现演示 |

## 5. 系统截图

| 图片 | 路径 | 用途 |
| --- | --- | --- |
| 打哈欠检测截图 | `docs/images/system_demo_yawn.jpg` | 论文和 PPT 展示 |
| 系统演示截图 | `docs/images/system_demo_second.jpg` | 论文和 PPT 展示 |
| 单图推理截图 | `docs/images/test_predict.jpg` | 检测效果展示 |

## 6. 当前可写入论文的核心结论

- YOLOv8n 基线模型 mAP50 为 0.8347，mAP50-95 为 0.5368。
- YOLOv8n + CBAM 模型 mAP50 为 0.8580，mAP50-95 为 0.5470。
- 引入 CBAM 注意力机制后，模型整体 mAP50 提升约 0.0233，mAP50-95 提升约 0.0102。
- 打哈欠类别提升明显，Yawn 类别 mAP50 从约 0.693 提升到 0.762。
- 系统已支持图片、视频、摄像头输入，并完成截图保存和日志记录。
