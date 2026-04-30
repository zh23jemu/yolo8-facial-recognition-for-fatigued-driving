from __future__ import annotations

import argparse
import sys
import time
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
        raise FileNotFoundError(f"未找到模型权重：{weights}")
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

    frame_ready = pyqtSignal(QPixmap)
    status_ready = pyqtSignal(str, bool)
    error_ready = pyqtSignal(str)

    def __init__(self, model, source: str | int, conf: float) -> None:
        super().__init__()
        self.model = model
        self.source = source
        self.conf = conf
        self._running = True
        self.evaluator = FatigueRuleEvaluator()

    def stop(self) -> None:
        """请求线程安全停止。"""

        self._running = False

    def run(self) -> None:
        """持续读取视频帧，执行检测并发送给界面。"""

        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            self.error_ready.emit(f"无法打开输入源：{self.source}")
            return

        try:
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

                state_text = (
                    f"状态：{state_info['state']}    "
                    f"闭眼比例：{state_info['closed_ratio']}    "
                    f"打哈欠次数：{state_info['yawn_count']}    "
                    f"FPS：{fps:.1f}"
                )
                self.status_ready.emit(state_text, bool(state_info["alarm"]))
                self.frame_ready.emit(cv_frame_to_pixmap(annotated))
        except Exception as exc:  # noqa: BLE001
            self.error_ready.emit(str(exc))
        finally:
            capture.release()


class MainWindow(QMainWindow):
    """疲劳驾驶检测桌面主窗口。"""

    def __init__(self, weights: str, conf: float) -> None:
        super().__init__()
        self.weights = weights
        self.conf = conf
        self.model = None
        self.worker: VideoWorker | None = None
        self.video_path: str | None = None
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

        self.select_button = QPushButton("选择视频")
        self.camera_button = QPushButton("打开摄像头")
        self.start_button = QPushButton("开始检测")
        self.stop_button = QPushButton("停止检测")
        self.stop_button.setEnabled(False)

        self.select_button.clicked.connect(self.select_video)
        self.camera_button.clicked.connect(self.start_camera)
        self.start_button.clicked.connect(self.start_video)
        self.stop_button.clicked.connect(self.stop_detection)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.select_button)
        button_layout.addWidget(self.camera_button)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.status_label)
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
        self.worker.error_ready.connect(self.show_error)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

        self.start_button.setEnabled(False)
        self.camera_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_detection(self) -> None:
        """停止当前检测任务。"""

        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(2000)
        self.on_worker_finished()

    def update_frame(self, pixmap: QPixmap) -> None:
        """刷新视频画面。"""

        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    def update_status(self, text: str, alarm: bool) -> None:
        """刷新疲劳状态文本和报警颜色。"""

        if alarm:
            self.status_label.setStyleSheet("font-size:18px;color:#b71c1c;font-weight:bold;")
        else:
            self.status_label.setStyleSheet("font-size:18px;color:#1b5e20;font-weight:bold;")
        self.status_label.setText(text)

    def show_error(self, message: str) -> None:
        """显示检测线程错误。"""

        QMessageBox.critical(self, "运行错误", message)

    def on_worker_finished(self) -> None:
        """检测结束后恢复按钮状态。"""

        self.start_button.setEnabled(True)
        self.camera_button.setEnabled(True)
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

