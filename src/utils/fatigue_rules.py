from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, Tuple


FATIGUE_LABELS = ("normal", "suspected_fatigue", "fatigue")

# 不同数据集导出的类别命名可能不同，例如 Roboflow 常见 Eyeclosed/Eyeopen/Yawn。
# 在进入规则判断前统一映射到项目内部类别，避免大小写或下划线差异影响疲劳判断。
CLASS_NAME_ALIASES = {
    "eyeopen": "eye_open",
    "eye_open": "eye_open",
    "open_eye": "eye_open",
    "openeye": "eye_open",
    "eyeclosed": "eye_closed",
    "eye_closed": "eye_closed",
    "closed_eye": "eye_closed",
    "closedeye": "eye_closed",
    "yawn": "yawn",
    "yawning": "yawn",
}


@dataclass
class FrameFatigueFeature:
    """单帧疲劳特征。

    该结构用于把 YOLOv8 的检测输出压缩为时序模型或规则判断可使用的数值特征。
    三个置信度字段取值范围为 0~1，分别表示当前帧中睁眼、闭眼和打哈欠目标的最高置信度。
    """

    eye_open_conf: float = 0.0
    eye_closed_conf: float = 0.0
    yawn_conf: float = 0.0

    def as_tuple(self) -> Tuple[float, float, float]:
        """返回固定顺序的三维特征，便于送入 LSTM 等时序模型。"""

        return (self.eye_open_conf, self.eye_closed_conf, self.yawn_conf)


class FatigueRuleEvaluator:
    """基于滑动窗口的疲劳状态规则判断器。

    第一轮开发先提供可解释、可演示的规则判断逻辑；后续训练好 LSTM + Attention 后，
    可以把该规则作为兜底策略，或把时序模型输出与规则输出进行融合。
    """

    def __init__(
        self,
        window_size: int = 90,
        eye_closed_threshold: float = 0.45,
        yawn_threshold: float = 0.50,
        fatigue_ratio: float = 0.35,
        suspected_ratio: float = 0.20,
    ) -> None:
        self.window_size = window_size
        self.eye_closed_threshold = eye_closed_threshold
        self.yawn_threshold = yawn_threshold
        self.fatigue_ratio = fatigue_ratio
        self.suspected_ratio = suspected_ratio
        self._features: Deque[FrameFatigueFeature] = deque(maxlen=window_size)

    def reset(self) -> None:
        """清空历史窗口，通常在切换视频源或重新开始检测时调用。"""

        self._features.clear()

    def update(self, feature: FrameFatigueFeature) -> Dict[str, float | str | bool]:
        """加入一帧特征并返回当前疲劳状态。

        返回值包含状态标签、闭眼比例、打哈欠次数和是否需要报警，便于 UI 或命令行直接展示。
        """

        self._features.append(feature)
        return self.evaluate()

    def evaluate(self) -> Dict[str, float | str | bool]:
        """根据当前滑动窗口计算疲劳状态。"""

        if not self._features:
            return {
                "state": "normal",
                "closed_ratio": 0.0,
                "yawn_count": 0,
                "alarm": False,
            }

        closed_flags = [
            item.eye_closed_conf >= self.eye_closed_threshold
            and item.eye_closed_conf >= item.eye_open_conf
            for item in self._features
        ]
        yawn_flags = [item.yawn_conf >= self.yawn_threshold for item in self._features]

        closed_ratio = sum(closed_flags) / len(closed_flags)
        yawn_count = self._count_events(yawn_flags)

        if closed_ratio >= self.fatigue_ratio or yawn_count >= 2:
            state = "fatigue"
        elif closed_ratio >= self.suspected_ratio or yawn_count >= 1:
            state = "suspected_fatigue"
        else:
            state = "normal"

        return {
            "state": state,
            "closed_ratio": round(closed_ratio, 4),
            "yawn_count": yawn_count,
            "alarm": state == "fatigue",
        }

    @staticmethod
    def _count_events(flags: Iterable[bool]) -> int:
        """统计连续 True 片段数量，用于避免一个打哈欠动作被每帧重复计数。"""

        count = 0
        in_event = False
        for flag in flags:
            if flag and not in_event:
                count += 1
                in_event = True
            elif not flag:
                in_event = False
        return count


def feature_from_detections(
    detections: Iterable[Tuple[str, float]],
) -> FrameFatigueFeature:
    """从检测结果中提取单帧最高置信度特征。

    参数 detections 的元素格式为 `(class_name, confidence)`，调用方可从 Ultralytics
    的检测框结果中生成该列表。函数会按类别取最高置信度，忽略未知类别。
    """

    feature = FrameFatigueFeature()
    for class_name, confidence in detections:
        normalized_name = CLASS_NAME_ALIASES.get(
            class_name.strip().lower().replace("-", "_").replace(" ", "_")
        )
        conf = float(confidence)
        if normalized_name == "eye_open":
            feature.eye_open_conf = max(feature.eye_open_conf, conf)
        elif normalized_name == "eye_closed":
            feature.eye_closed_conf = max(feature.eye_closed_conf, conf)
        elif normalized_name == "yawn":
            feature.yawn_conf = max(feature.yawn_conf, conf)
    return feature
