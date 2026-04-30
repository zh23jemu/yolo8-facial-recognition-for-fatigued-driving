from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch
from torch import Tensor, nn


@dataclass
class SequencePrediction:
    """时序疲劳分类输出结果。"""

    label: str
    confidence: float
    probabilities: Dict[str, float]


class TemporalAttention(nn.Module):
    """LSTM 输出序列上的注意力层。

    LSTM 会为每一帧输出一个隐藏向量，注意力层学习不同帧的重要程度，
    使模型在判断疲劳时更关注闭眼持续片段、打哈欠片段等关键帧。
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, sequence_output: Tensor) -> Tensor:
        """计算注意力加权后的序列表示。

        参数形状为 `[batch, sequence_length, hidden_size]`，
        返回形状为 `[batch, hidden_size]` 的上下文向量。
        """

        weights = torch.softmax(self.score(sequence_output), dim=1)
        context = torch.sum(sequence_output * weights, dim=1)
        return context


class LSTMAttentionClassifier(nn.Module):
    """疲劳驾驶时序分类模型。

    输入特征默认为三维：睁眼置信度、闭眼置信度、打哈欠置信度。
    输出三类：正常、疑似疲劳、疲劳。
    """

    labels = ("normal", "suspected_fatigue", "fatigue")

    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_classes: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=False,
        )
        self.attention = TemporalAttention(hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """前向计算分类 logits。

        x 的形状为 `[batch, sequence_length, 3]`。
        """

        output, _ = self.lstm(x)
        context = self.attention(output)
        return self.classifier(context)

    @torch.no_grad()
    def predict(
        self,
        features: Iterable[Iterable[float]],
        device: str | torch.device = "cpu",
    ) -> SequencePrediction:
        """对单个时序窗口进行预测。

        features 是连续帧特征列表，例如 30 帧的 `[eye_open, eye_closed, yawn]`。
        该方法主要供演示系统或推理脚本快速调用。
        """

        self.eval()
        self.to(device)
        feature_list: List[List[float]] = [list(item) for item in features]
        if not feature_list:
            raise ValueError("features 不能为空，至少需要一帧时序特征。")

        tensor = torch.tensor(feature_list, dtype=torch.float32, device=device).unsqueeze(0)
        logits = self.forward(tensor)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        index = int(torch.argmax(probs).item())
        probabilities = {
            label: round(float(probs[i].item()), 4)
            for i, label in enumerate(self.labels)
        }
        return SequencePrediction(
            label=self.labels[index],
            confidence=probabilities[self.labels[index]],
            probabilities=probabilities,
        )

