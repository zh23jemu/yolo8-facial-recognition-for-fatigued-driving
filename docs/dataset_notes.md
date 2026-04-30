# 数据集说明

## 推荐数据集

1. Roboflow Driver_Drowsiness_YOLO  
   用于 YOLOv8 目标检测训练，类别建议统一为 `eye_open`、`eye_closed`、`yawn`。

2. UTA-RLDD  
   用于时序疲劳分类，建议从视频中抽取连续帧，生成 `features.csv`。

3. NTHU Driver Drowsiness Detection Dataset  
   作为论文增强和复杂场景补充数据集，需通过官方许可申请获取。

## 标注规范

- `eye_open`：眼睛明显睁开。
- `eye_closed`：眼睛闭合或接近闭合。
- `yawn`：嘴部明显张开并符合打哈欠状态。

## 划分比例

默认按 70% / 15% / 15% 划分训练集、验证集和测试集。若使用 Roboflow 已划分数据集，应在论文中说明实际划分来源。

