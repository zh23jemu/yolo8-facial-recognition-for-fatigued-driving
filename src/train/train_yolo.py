from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """解析 YOLOv8 训练参数。

    参数尽量保持和 Ultralytics 原生训练接口一致，便于后续在论文实验中复现实验配置。
    """

    parser = argparse.ArgumentParser(description="训练疲劳驾驶 YOLOv8 检测模型")
    parser.add_argument("--data", default="configs/yolo_data.yaml", help="YOLO 数据集配置文件路径")
    parser.add_argument("--model", default="yolov8n.pt", help="预训练模型或自定义模型配置")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸")
    parser.add_argument("--batch", type=int, default=16, help="批大小")
    parser.add_argument("--workers", type=int, default=4, help="数据加载进程数量，Slurm 环境建议不超过分配的 CPU 核数")
    parser.add_argument("--device", default=None, help="训练设备，例如 0、cpu；不填则由框架自动选择")
    parser.add_argument("--project", default="runs/yolo", help="训练输出目录")
    parser.add_argument("--name", default="fatigue_yolov8n", help="本次实验名称")
    parser.add_argument("--export-onnx", action="store_true", help="训练结束后导出 ONNX 模型")
    return parser.parse_args()


def main() -> None:
    """YOLOv8 训练入口。

    训练输出会保存在 runs/yolo 下，包含指标、权重和可视化结果，
    可直接作为毕业论文实验记录和模型文件来源。
    """

    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"数据配置文件不存在：{data_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "未安装 ultralytics，请先执行：.venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        ) from exc

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "optimizer": "AdamW",
        "cos_lr": True,
        "mosaic": 1.0,
        "mixup": 0.15,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)

    if args.export_onnx:
        # ONNX 模型用于后续 ONNX Runtime 推理和轻量化部署实验。
        model.export(format="onnx", imgsz=args.imgsz, simplify=True)

    print("训练完成，结果已保存到：", results.save_dir)


if __name__ == "__main__":
    main()
