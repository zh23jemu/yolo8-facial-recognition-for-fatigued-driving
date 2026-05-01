from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import cv2
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.utils.fatigue_rules import FatigueRuleEvaluator, feature_from_detections


STATE_TEXT = {
    "normal": "正常",
    "suspected_fatigue": "疑似疲劳",
    "fatigue": "疲劳",
}


def parse_args() -> argparse.Namespace:
    """解析桌面演示系统参数。"""

    parser = argparse.ArgumentParser(description="疲劳驾驶检测桌面演示系统")
    parser.add_argument("--weights", default="weights/best.pt", help="YOLOv8 权重路径")
    parser.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值")
    return parser.parse_args()


def load_yolo(weights: str):
    """加载 YOLOv8 模型。"""

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "未安装 ultralytics，请先执行：.venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        ) from exc

    if not Path(weights).exists():
        raise FileNotFoundError(
            f"未找到模型权重：{weights}\n\n"
            "请先把服务器训练得到的 best.pt 放到项目的 weights/best.pt，"
            "或启动时通过 --weights 指定实际权重路径。"
        )
    return YOLO(weights)


def detections_from_result(result) -> List[Tuple[str, float]]:
    """从单帧 YOLOv8 结果中提取类别名和置信度。"""

    detections: List[Tuple[str, float]] = []
    names = result.names
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        detections.append((str(names[class_id]), confidence))
    return detections


def cv_frame_to_pixmap(frame) -> QPixmap:
    """把 OpenCV BGR 图像转换为 PyQt 可显示的 QPixmap。"""

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channel = rgb.shape
    bytes_per_line = channel * width
    image = QImage(rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(image.copy())


class VideoWorker(QThread):
    """视频检测线程。

    推理过程放到独立线程中执行，避免 YOLOv8 检测阻塞主界面。
    """

    frame_ready = pyqtSignal(object)
    status_ready = pyqtSignal(str, bool, str)
    log_path_ready = pyqtSignal(str)
    error_ready = pyqtSignal(str)

    def __init__(self, model, source: str | int, conf: float) -> None:
        super().__init__()
        self.model = model
        self.source = source
        self.conf = conf
        self._running = True
        self.evaluator = FatigueRuleEvaluator()
        self.frame_index = 0

    def stop(self) -> None:
        """请求线程安全停止。"""

        self._running = False

    def run(self) -> None:
        """持续读取视频帧，执行检测并发送给界面。"""

        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            self.error_ready.emit(f"无法打开输入源：{self.source}")
            return

        log_file = None
        log_writer = None
        try:
            log_dir = Path("runs/app_logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            log_file = log_path.open("w", newline="", encoding="utf-8-sig")
            log_writer = csv.writer(log_file)
            log_writer.writerow(
                [
                    "frame_index",
                    "state",
                    "closed_ratio",
                    "yawn_count",
                    "alarm",
                    "fps",
                    "detections",
                ]
            )
            self.log_path_ready.emit(str(log_path))

            while self._running:
                ok, frame = capture.read()
                if not ok:
                    break

                start = time.perf_counter()
                result = self.model.predict(frame, conf=self.conf, verbose=False)[0]
                annotated = result.plot()
                detections = detections_from_result(result)
                feature = feature_from_detections(detections)
                state_info = self.evaluator.update(feature)
                fps = 1.0 / max(time.perf_counter() - start, 1e-6)
                state = str(state_info["state"])
                state_cn = STATE_TEXT.get(state, state)
                detection_text = "；".join(
                    f"{name}:{conf:.2f}" for name, conf in detections
                ) or "未检测到目标"

                state_text = (
                    f"状态：{state_cn}    "
                    f"闭眼比例：{state_info['closed_ratio']}    "
                    f"打哈欠次数：{state_info['yawn_count']}    "
                    f"FPS：{fps:.1f}"
                )
                if log_writer is not None:
                    log_writer.writerow(
                        [
                            self.frame_index,
                            state,
                            state_info["closed_ratio"],
                            state_info["yawn_count"],
                            int(bool(state_info["alarm"])),
                            round(fps, 2),
                            detection_text,
                        ]
                    )
                self.status_ready.emit(state_text, bool(state_info["alarm"]), detection_text)
                self.frame_ready.emit(annotated)
                self.frame_index += 1
        except Exception as exc:  # noqa: BLE001
            self.error_ready.emit(str(exc))
        finally:
            capture.release()
            if log_file is not None:
                log_file.close()


class MainWindow(QMainWindow):
    """疲劳驾驶检测桌面主窗口。"""

    def __init__(self, weights: str, conf: float) -> None:
        super().__init__()
        self.weights = weights
        self.conf = conf
        self.model = None
        self.worker: VideoWorker | None = None
        self.video_path: str | None = None
        self.latest_frame = None
        self.current_log_path: str | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        """创建界面控件和布局。"""

        self.setWindowTitle("疲劳驾驶面部识别系统")
        self.resize(1100, 760)

        self.video_label = QLabel("请选择视频或打开摄像头")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(960, 540)
        self.video_label.setStyleSheet("background:#111;color:#ddd;font-size:22px;")

        self.status_label = QLabel("状态：未开始")
        self.status_label.setStyleSheet("font-size:18px;color:#1b5e20;font-weight:bold;")

        self.detail_label = QLabel(f"权重：{self.weights}    置信度阈值：{self.conf}")
        self.detail_label.setWordWrap(True)
        self.detail_label.setStyleSheet("font-size:14px;color:#444;")

        self.select_button = QPushButton("选择视频")
        self.camera_button = QPushButton("打开摄像头")
        self.start_button = QPushButton("开始检测")
        self.stop_button = QPushButton("停止检测")
        self.screenshot_button = QPushButton("保存截图")
        self.stop_button.setEnabled(False)
        self.screenshot_button.setEnabled(False)

        self.select_button.clicked.connect(self.select_video)
        self.camera_button.clicked.connect(self.start_camera)
        self.start_button.clicked.connect(self.start_video)
        self.stop_button.clicked.connect(self.stop_detection)
        self.screenshot_button.clicked.connect(self.save_screenshot)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.select_button)
        button_layout.addWidget(self.camera_button)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.screenshot_button)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.detail_label)
        layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def ensure_model(self) -> bool:
        """按需加载模型，避免程序启动时因未准备权重而直接退出。"""

        if self.model is not None:
            return True
        try:
            self.model = load_yolo(self.weights)
            return True
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "模型加载失败", str(exc))
            return False

    def select_video(self) -> None:
        """选择本地视频文件。"""

        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv)",
        )
        if path:
            self.video_path = path
            self.status_label.setText(f"已选择视频：{path}")
            self.detail_label.setText(f"当前视频：{path}")

    def start_video(self) -> None:
        """开始检测已选择的视频。"""

        if not self.video_path:
            QMessageBox.information(self, "提示", "请先选择视频文件。")
            return
        self._start_source(self.video_path)

    def start_camera(self) -> None:
        """打开默认摄像头并开始检测。"""

        self._start_source(0)

    def _start_source(self, source: str | int) -> None:
        """启动指定输入源的检测线程。"""

        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "提示", "检测正在运行，请先停止当前任务。")
            return
        if not self.ensure_model():
            return

        self.worker = VideoWorker(self.model, source, self.conf)
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.status_ready.connect(self.update_status)
        self.worker.log_path_ready.connect(self.update_log_path)
        self.worker.error_ready.connect(self.show_error)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

        self.start_button.setEnabled(False)
        self.camera_button.setEnabled(False)
        self.select_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.screenshot_button.setEnabled(True)

    def stop_detection(self) -> None:
        """停止当前检测任务。"""

        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(2000)
        self.on_worker_finished()

    def update_frame(self, frame) -> None:
        """刷新视频画面。"""

        self.latest_frame = frame
        pixmap = cv_frame_to_pixmap(frame)
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    def update_status(self, text: str, alarm: bool, detection_text: str) -> None:
        """刷新疲劳状态文本和报警颜色。"""

        if alarm:
            self.status_label.setStyleSheet("font-size:18px;color:#b71c1c;font-weight:bold;")
        else:
            self.status_label.setStyleSheet("font-size:18px;color:#1b5e20;font-weight:bold;")
        self.status_label.setText(text)
        log_text = f"日志：{self.current_log_path}" if self.current_log_path else "日志：准备中"
        self.detail_label.setText(f"检测目标：{detection_text}    {log_text}")

    def update_log_path(self, path: str) -> None:
        """显示当前检测日志保存位置。"""

        self.current_log_path = path
        self.detail_label.setText(f"检测日志：{path}")

    def save_screenshot(self) -> None:
        """保存当前检测画面截图，便于论文和答辩使用。"""

        if self.latest_frame is None:
            QMessageBox.information(self, "提示", "当前还没有可保存的检测画面。")
            return

        screenshot_dir = Path("runs/screenshots")
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        default_path = screenshot_dir / f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存检测截图",
            str(default_path),
            "Image Files (*.jpg *.png *.bmp)",
        )
        if not path:
            return

        success = cv2.imwrite(path, self.latest_frame)
        if success:
            QMessageBox.information(self, "保存成功", f"截图已保存到：\n{path}")
        else:
            QMessageBox.warning(self, "保存失败", f"无法保存截图：\n{path}")

    def show_error(self, message: str) -> None:
        """显示检测线程错误。"""

        QMessageBox.critical(self, "运行错误", message)

    def on_worker_finished(self) -> None:
        """检测结束后恢复按钮状态。"""

        self.start_button.setEnabled(True)
        self.camera_button.setEnabled(True)
        self.select_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def closeEvent(self, event) -> None:  # noqa: N802
        """窗口关闭时确保后台线程退出。"""

        self.stop_detection()
        event.accept()


def main() -> None:
    """桌面演示系统入口。"""

    args = parse_args()
    app = QApplication(sys.argv)
    window = MainWindow(args.weights, args.conf)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
