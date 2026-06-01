from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import cv2

from src.utils.fatigue_rules import FatigueRuleEvaluator, feature_from_detections
from src.utils.ultralytics_patches import register_attention_modules


def configure_qt_runtime_paths() -> None:
    """在导入 Qt 图形后端前补齐插件和运行库路径。

    部分 Windows 环境下，用户把项目复制到新的目录后，PyQt5 虽然已经安装，
    但 Qt 平台插件目录不会被自动发现，进而出现
    “Could not find the Qt platform plugin 'windows'” 的报错。
    这里根据当前 Python 环境中的 PyQt5 安装位置，主动设置插件目录和 DLL 搜索路径，
    让桌面程序在交付包场景下更稳定。
    """

    pyqt_spec = importlib.util.find_spec("PyQt5")
    if pyqt_spec is None or pyqt_spec.origin is None:
        return

    pyqt_dir = Path(pyqt_spec.origin).resolve().parent
    qt_roots = [pyqt_dir / "Qt5", pyqt_dir / "Qt"]
    plugin_dir = next((root / "plugins" for root in qt_roots if (root / "plugins" / "platforms").exists()), None)
    platforms_dir = plugin_dir / "platforms" if plugin_dir is not None else None
    qt_bin_dir = next((root / "bin" for root in qt_roots if (root / "bin").exists()), None)

    # 客户机器上 QT_QPA_PLATFORM_PLUGIN_PATH 可能已经存在但值为空字符串。
    # setdefault 不会覆盖这种空值，因此这里显式判断并写入真实 platforms 目录。
    if platforms_dir is not None and platforms_dir.exists():
        if not os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH"):
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platforms_dir)
    if plugin_dir is not None and plugin_dir.exists():
        if not os.environ.get("QT_PLUGIN_PATH"):
            os.environ["QT_PLUGIN_PATH"] = str(plugin_dir)

    # qwindows.dll 依赖的 Qt5Core/Qt5Gui 等 DLL 也需要能被系统找到，
    # 因此把 Qt 自带 bin 目录放入 PATH 和 add_dll_directory 搜索范围。
    if qt_bin_dir is not None and qt_bin_dir.exists():
        current_path = os.environ.get("PATH", "")
        qt_bin = str(qt_bin_dir)
        if qt_bin not in current_path.split(os.pathsep):
            os.environ["PATH"] = qt_bin + os.pathsep + current_path if current_path else qt_bin
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(qt_bin)
            except OSError:
                # 某些受限环境可能不允许重复或额外注册 DLL 目录，此时保留 PATH 即可。
                pass


configure_qt_runtime_paths()

from PyQt5.QtCore import QPoint, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


STATE_TEXT = {
    "normal": "正常",
    "suspected_fatigue": "疑似疲劳",
    "fatigue": "疲劳",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

APP_STYLESHEET = """
QWidget#appRoot {
    background: #eef2f5;
    color: #17212b;
    font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
}

QWidget#appRoot[alarm="true"] {
    background: #fff1f2;
}

QLabel#titleLabel {
    color: #17212b;
    font-size: 30px;
    font-weight: 800;
}

QLabel#subtitleLabel {
    color: #627183;
    font-size: 14px;
}

QFrame#videoFrame {
    background: #111827;
    border: 1px solid #263241;
    border-radius: 12px;
}

QLabel#videoLabel {
    background: #0b1118;
    border: 1px solid #202b38;
    border-radius: 8px;
    color: #cbd5e1;
    font-size: 22px;
    font-weight: 600;
}

QFrame#statusFrame {
    background: #ffffff;
    border: 1px solid #d9e2ea;
    border-radius: 10px;
}

QFrame#statusFrame[alarm="true"] {
    background: #fff7f7;
    border: 2px solid #dc2626;
}

QLabel#statusLabel {
    font-size: 19px;
    font-weight: 800;
    padding: 4px 0;
}

QLabel#statusLabel[status="idle"] {
    color: #475569;
}

QLabel#statusLabel[status="running"] {
    color: #1d4ed8;
}

QLabel#statusLabel[status="normal"] {
    color: #15803d;
}

QLabel#statusLabel[status="alarm"] {
    color: #b91c1c;
}

QLabel#detailLabel {
    color: #526173;
    font-size: 14px;
    line-height: 1.35;
}

QPushButton {
    min-height: 38px;
    padding: 8px 18px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 700;
}

QPushButton#primaryButton {
    background: #0f766e;
    color: #ffffff;
    border: 1px solid #0f766e;
}

QPushButton#primaryButton:hover {
    background: #115e59;
}

QPushButton#secondaryButton {
    background: #ffffff;
    color: #334155;
    border: 1px solid #cbd5e1;
}

QPushButton#secondaryButton:hover {
    background: #f8fafc;
    border-color: #94a3b8;
}

QPushButton#successButton {
    background: #2563eb;
    color: #ffffff;
    border: 1px solid #2563eb;
}

QPushButton#successButton:hover {
    background: #1d4ed8;
}

QPushButton#dangerButton {
    background: #dc2626;
    color: #ffffff;
    border: 1px solid #dc2626;
}

QPushButton#dangerButton:hover {
    background: #b91c1c;
}

QPushButton:disabled {
    background: #d8dee6;
    color: #8a97a6;
    border: 1px solid #cbd5e1;
}
"""


def parse_args() -> argparse.Namespace:
    """解析桌面演示系统参数。"""

    parser = argparse.ArgumentParser(description="疲劳驾驶检测桌面演示系统")
    parser.add_argument("--weights", default="weights/best.pt", help="YOLOv8 权重路径")
    parser.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值")
    return parser.parse_args()


def load_yolo(weights: str):
    """加载 YOLOv8 模型。"""

    register_attention_modules()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "未安装 ultralytics，请先执行：.venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            "模型依赖加载失败，通常是 Windows 本地 PyTorch 安装不完整或 CUDA/CPU 版本不匹配。\n\n"
            "建议修复命令：\n"
            ".venv\\Scripts\\python.exe -m pip uninstall -y torch torchvision torchaudio\n"
            ".venv\\Scripts\\python.exe -m pip install -r requirements-windows-cpu.txt"
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

        log_file = None
        log_writer = None
        capture = None
        try:
            source_path = Path(str(self.source))
            is_image_source = (
                isinstance(self.source, str)
                and source_path.suffix.lower() in IMAGE_SUFFIXES
            )
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

            if is_image_source:
                frame = cv2.imread(str(source_path))
                if frame is None:
                    self.error_ready.emit(f"无法读取图片：{self.source}")
                    return
                self._process_frame(frame, log_writer)
                return

            capture = cv2.VideoCapture(self.source)
            if not capture.isOpened():
                self.error_ready.emit(f"无法打开输入源：{self.source}")
                return

            while self._running:
                ok, frame = capture.read()
                if not ok:
                    break

                self._process_frame(frame, log_writer)
        except Exception as exc:  # noqa: BLE001
            self.error_ready.emit(str(exc))
        finally:
            if capture is not None:
                capture.release()
            if log_file is not None:
                log_file.close()

    def _process_frame(self, frame, log_writer) -> None:
        """处理一帧图像并发送界面更新。

        图片、视频和摄像头最终都会走这个函数，保证检测显示、日志记录和状态判断逻辑一致。
        """

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


class MainWindow(QMainWindow):
    """疲劳驾驶检测桌面主窗口。"""

    def __init__(self, weights: str, conf: float) -> None:
        super().__init__()
        self.weights = weights
        self.conf = conf
        self.model = None
        self.worker: VideoWorker | None = None
        self.media_path: str | None = None
        self.latest_frame = None
        self.current_log_path: str | None = None
        self.is_alarm_active = False
        self._alarm_shake_origin: QPoint | None = None
        self._alarm_shake_index = 0
        self._alarm_shake_offsets = [
            QPoint(-12, 0),
            QPoint(12, 0),
            QPoint(-9, 0),
            QPoint(9, 0),
            QPoint(-5, 0),
            QPoint(5, 0),
            QPoint(0, 0),
        ]
        self._alarm_shake_timer = QTimer(self)
        self._alarm_shake_timer.setInterval(35)
        self._alarm_shake_timer.timeout.connect(self._run_alarm_shake_step)
        self._build_ui()

    def _build_ui(self) -> None:
        """创建界面控件和布局。"""

        self.setWindowTitle("疲劳驾驶面部识别系统")
        self.resize(1180, 820)
        self.setMinimumSize(1040, 720)

        title_label = QLabel("疲劳驾驶面部识别系统")
        title_label.setObjectName("titleLabel")
        subtitle_label = QLabel(
            "YOLOv8 + CBAM 注意力机制 · 图片自动检测 · 视频/摄像头实时监测"
        )
        subtitle_label.setObjectName("subtitleLabel")

        header_layout = QVBoxLayout()
        header_layout.setSpacing(4)
        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)

        self.video_label = QLabel("请选择图片/视频或打开摄像头")
        self.video_label.setObjectName("videoLabel")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(960, 540)

        video_frame = QFrame()
        video_frame.setObjectName("videoFrame")
        video_layout = QVBoxLayout()
        video_layout.setContentsMargins(12, 12, 12, 12)
        video_layout.addWidget(self.video_label)
        video_frame.setLayout(video_layout)

        self.status_label = QLabel("状态：未开始")
        self.status_label.setObjectName("statusLabel")

        self.detail_label = QLabel(f"权重：{self.weights}    置信度阈值：{self.conf}")
        self.detail_label.setObjectName("detailLabel")
        self.detail_label.setWordWrap(True)

        self.status_frame = QFrame()
        self.status_frame.setObjectName("statusFrame")
        status_layout = QVBoxLayout()
        status_layout.setContentsMargins(18, 14, 18, 14)
        status_layout.setSpacing(8)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.detail_label)
        self.status_frame.setLayout(status_layout)

        self.select_button = QPushButton("选择图片/视频")
        self.camera_button = QPushButton("打开摄像头")
        self.start_button = QPushButton("开始检测")
        self.stop_button = QPushButton("停止检测")
        self.screenshot_button = QPushButton("保存截图")
        self.select_button.setObjectName("primaryButton")
        self.camera_button.setObjectName("secondaryButton")
        self.start_button.setObjectName("successButton")
        self.stop_button.setObjectName("dangerButton")
        self.screenshot_button.setObjectName("secondaryButton")
        self.stop_button.setEnabled(False)
        self.screenshot_button.setEnabled(False)

        self.select_button.clicked.connect(self.select_media)
        self.camera_button.clicked.connect(self.start_camera)
        self.start_button.clicked.connect(self.start_video)
        self.stop_button.clicked.connect(self.stop_detection)
        self.screenshot_button.clicked.connect(self.save_screenshot)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        button_layout.addWidget(self.select_button)
        button_layout.addWidget(self.camera_button)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.screenshot_button)

        layout = QVBoxLayout()
        layout.setContentsMargins(26, 22, 26, 22)
        layout.setSpacing(18)
        layout.addLayout(header_layout)
        layout.addWidget(video_frame)
        layout.addWidget(self.status_frame)
        layout.addLayout(button_layout)

        self.app_container = QWidget()
        self.app_container.setObjectName("appRoot")
        self.app_container.setLayout(layout)
        self.setCentralWidget(self.app_container)
        self.setStyleSheet(APP_STYLESHEET)
        self._set_status_style("idle")
        self._set_alarm_visual(False)

    def _set_status_style(self, status: str) -> None:
        """按检测状态切换状态栏视觉样式。

        使用动态属性统一管理颜色，避免在多个回调里重复拼接大段样式。
        """

        self.status_label.setProperty("status", status)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def _refresh_dynamic_style(self, widget: QWidget) -> None:
        """刷新使用动态属性控制的 Qt 样式。"""

        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def _set_alarm_visual(self, active: bool) -> None:
        """切换疲劳报警的主界面颜色。

        报警状态只改变主容器和状态区域的动态属性，具体颜色仍集中在
        APP_STYLESHEET 中维护，避免在业务回调里拼接样式字符串。
        """

        alarm_value = "true" if active else "false"
        self.app_container.setProperty("alarm", alarm_value)
        self.status_frame.setProperty("alarm", alarm_value)
        self._refresh_dynamic_style(self.app_container)
        self._refresh_dynamic_style(self.status_frame)

    def _reset_alarm_state(self) -> None:
        """恢复非报警状态，并确保抖动动画停在原始窗口位置。"""

        self.is_alarm_active = False
        self._set_alarm_visual(False)
        if self._alarm_shake_timer.isActive():
            self._alarm_shake_timer.stop()
        if self._alarm_shake_origin is not None:
            self.move(self._alarm_shake_origin)
            self._alarm_shake_origin = None

    def _trigger_alarm_effect(self) -> None:
        """触发一次疲劳报警窗口抖动。

        持续疲劳期间不会重复调用本函数；只有从非疲劳进入疲劳时，
        update_status 才会触发一次，避免窗口长时间晃动影响操作。
        """

        if self.isMaximized() or self.isFullScreen():
            return
        if self._alarm_shake_timer.isActive():
            return
        self._alarm_shake_origin = self.pos()
        self._alarm_shake_index = 0
        self._alarm_shake_timer.start()

    def _run_alarm_shake_step(self) -> None:
        """按预设偏移量执行一次窗口抖动动画步骤。"""

        if self._alarm_shake_origin is None:
            self._alarm_shake_timer.stop()
            return
        if self._alarm_shake_index >= len(self._alarm_shake_offsets):
            self._alarm_shake_timer.stop()
            self.move(self._alarm_shake_origin)
            self._alarm_shake_origin = None
            return

        self.move(self._alarm_shake_origin + self._alarm_shake_offsets[self._alarm_shake_index])
        self._alarm_shake_index += 1

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

    def select_media(self) -> None:
        """选择本地图片或视频文件。

        没有演示视频时，可以直接选择数据集测试图片进行单张检测，
        这样答辩时也能展示检测框、疲劳状态、截图和日志功能。
        选择文件后仅更新当前输入源，用户点击“开始检测”后才真正启动模型推理。
        """

        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片或视频文件",
            "",
            "Media Files (*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mov *.mkv *.wmv)",
        )
        if path:
            self.media_path = path
            self._set_status_style("idle")
            self.status_label.setText("已选择文件，点击“开始检测”后开始识别")
            self.detail_label.setText(f"当前文件：{path}")

    def start_video(self) -> None:
        """开始检测已选择的图片或视频。"""

        if not self.media_path:
            QMessageBox.information(self, "提示", "请先选择图片或视频文件。")
            return
        self._start_source(self.media_path)

    def start_camera(self) -> None:
        """打开默认摄像头并开始检测。"""

        self._start_source(0)

    def _start_source(self, source: str | int) -> None:
        """启动指定输入源的检测线程。"""

        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "提示", "检测正在运行，请先停止当前任务。")
            return
        # 自动打开图片时，首次加载 YOLO 模型可能耗时较长。
        # 因此先把“正在检测/准备模型”的提示显示出来，并立即刷新事件循环，
        # 避免用户误以为选择图片后没有响应。
        source_text = "摄像头" if isinstance(source, int) else str(source)
        self._reset_alarm_state()
        self._set_status_style("running")
        self.status_label.setText(f"正在检测：{source_text}")
        self.detail_label.setText("正在准备模型并启动检测，请稍候...")
        QApplication.processEvents()

        if not self.ensure_model():
            self._set_status_style("idle")
            self.status_label.setText("状态：模型加载失败")
            return

        self.worker = VideoWorker(self.model, source, self.conf)
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.status_ready.connect(self.update_status)
        self.worker.log_path_ready.connect(self.update_log_path)
        self.worker.error_ready.connect(self.show_error)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

        # 检测线程启动后更新为推理提示，等待第一帧结果返回。
        self._set_status_style("running")
        self.status_label.setText(f"正在检测：{source_text}")
        self.detail_label.setText("模型正在推理，请稍候...")

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
        self._reset_alarm_state()

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
            if not self.is_alarm_active:
                self._trigger_alarm_effect()
            self.is_alarm_active = True
            self._set_alarm_visual(True)
            self._set_status_style("alarm")
        else:
            self.is_alarm_active = False
            self._set_alarm_visual(False)
            self._set_status_style("normal")
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
