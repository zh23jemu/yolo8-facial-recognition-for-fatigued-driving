from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2

from src.utils.fatigue_rules import FatigueRuleEvaluator, feature_from_detections
from src.utils.ultralytics_patches import register_attention_modules


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def parse_args() -> argparse.Namespace:
    """解析推理脚本参数。"""

    parser = argparse.ArgumentParser(description="疲劳驾驶 YOLOv8 图片/视频/摄像头推理")
    parser.add_argument("--weights", required=True, help="YOLOv8 权重路径，例如 weights/best.pt")
    parser.add_argument("--source", required=True, help="图片路径、视频路径或摄像头编号，例如 0")
    parser.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值")
    parser.add_argument("--show", action="store_true", help="是否弹窗显示检测结果")
    parser.add_argument("--save", default=None, help="可选：保存推理视频或图片的路径")
    return parser.parse_args()


def load_model(weights: str):
    """延迟导入并加载 YOLOv8 模型。

    延迟导入可以让缺少依赖时的错误提示更清晰，也方便文档环境先进行静态检查。
    """

    register_attention_modules()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "未安装 ultralytics，请先执行：.venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        ) from exc
    return YOLO(weights)


def source_kind(source: str) -> str:
    """判断输入源类型。"""

    if source.isdigit():
        return "camera"
    suffix = Path(source).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    raise ValueError(f"无法识别输入源类型：{source}")


def detections_from_result(result) -> List[Tuple[str, float]]:
    """从 Ultralytics 单帧结果中提取 `(类别名, 置信度)` 列表。"""

    detections: List[Tuple[str, float]] = []
    names = result.names
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        detections.append((str(names[class_id]), confidence))
    return detections


def draw_status(frame, state_info: dict, fps: float):
    """在画面左上角绘制 FPS 和疲劳状态。"""

    state = str(state_info["state"])
    alarm = bool(state_info["alarm"])
    color = (0, 0, 255) if alarm else (0, 180, 255) if state == "suspected_fatigue" else (0, 180, 0)
    text = f"State: {state} | FPS: {fps:.1f} | Closed: {state_info['closed_ratio']}"
    cv2.rectangle(frame, (8, 8), (780, 48), (0, 0, 0), thickness=-1)
    cv2.putText(frame, text, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    return frame


def run_frame_infer(model, frame, evaluator: FatigueRuleEvaluator, conf: float):
    """对单帧执行检测并叠加疲劳状态。"""

    start = time.perf_counter()
    result = model.predict(frame, conf=conf, verbose=False)[0]
    annotated = result.plot()
    detections = detections_from_result(result)
    feature = feature_from_detections(detections)
    state_info = evaluator.update(feature)
    fps = 1.0 / max(time.perf_counter() - start, 1e-6)
    return draw_status(annotated, state_info, fps), detections, state_info, fps


def infer_image(model, source: str, conf: float, show: bool, save: str | None) -> None:
    """图片推理入口。"""

    image = cv2.imread(source)
    if image is None:
        raise FileNotFoundError(f"无法读取图片：{source}")
    evaluator = FatigueRuleEvaluator(window_size=1)
    output, detections, state_info, fps = run_frame_infer(model, image, evaluator, conf)
    print(f"FPS={fps:.2f}, state={state_info['state']}, detections={detections}")

    if save:
        cv2.imwrite(save, output)
    if show:
        cv2.imshow("Fatigue Detection", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def infer_stream(model, source: str | int, conf: float, show: bool, save: str | None) -> None:
    """视频或摄像头推理入口。"""

    capture = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not capture.isOpened():
        raise RuntimeError(f"无法打开输入源：{source}")

    writer = None
    evaluator = FatigueRuleEvaluator()
    if save:
        fps = capture.get(cv2.CAP_PROP_FPS) or 25
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save, fourcc, fps, (width, height))

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            output, detections, state_info, fps = run_frame_infer(model, frame, evaluator, conf)
            print(f"FPS={fps:.2f}, state={state_info['state']}, detections={detections}")

            if writer is not None:
                writer.write(output)
            if show:
                cv2.imshow("Fatigue Detection", output)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()


def main() -> None:
    """命令行推理总入口。"""

    args = parse_args()
    model = load_model(args.weights)
    kind = source_kind(args.source)
    if kind == "image":
        infer_image(model, args.source, args.conf, args.show, args.save)
    else:
        infer_stream(model, args.source, args.conf, args.show, args.save)


if __name__ == "__main__":
    main()
