from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from model.cdnet import CDNet


Tensor4 = torch.Tensor
AdapterOutput = Tuple[Tensor4, Optional[Tensor4], Optional[Tensor4], Optional[Tensor4]]


@dataclass(frozen=True)
class AdapterRule:
    """将原始输出映射到统一输出元组的规则。

    Args:
        fn (Callable[[object], AdapterOutput]): 转换函数，用于将 baseline
            原始输出映射为 ``(logits, edge, feat1, feat2)``。
    """

    fn: Callable[[object], AdapterOutput]


def _as_4d(x: object, name: str) -> Tensor4:
    """校验输入并返回四维张量。

    Args:
        x (object): 待校验对象。
        name (str): 变量名，用于报错信息。

    Returns:
        Tensor4: 形状为 ``[B, C, H, W]`` 的张量。

    Raises:
        TypeError: 当 ``x`` 不是 ``torch.Tensor`` 时抛出。
        ValueError: 当张量维度不是 4 时抛出。
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor, got {type(x)}")
    if x.dim() != 4:
        raise ValueError(f"{name} must be 4D [B,C,H,W], got shape {tuple(x.shape)}")
    return x


def _single_to_unified(raw: object) -> AdapterOutput:
    """将单一 logits 输出转换为统一格式。

    Args:
        raw (object): 模型原始输出，期望为 logits 张量。

    Returns:
        AdapterOutput: ``(logits, None, None, None)``。
    """
    logits = _as_4d(raw, "logits")
    return logits, None, None, None


def _edge_logits_pair_to_unified(raw: object) -> AdapterOutput:
    """将 ``(edge, logits)`` 输出转换为统一格式。

    Args:
        raw (object): 模型原始输出，期望为 ``tuple(edge, logits)``。

    Returns:
        AdapterOutput: ``(logits, edge, None, None)``。
    """
    if not (isinstance(raw, tuple) and len(raw) == 2):
        raise ValueError("Expected tuple(edge, logits)")
    edge = _as_4d(raw[0], "edge")
    logits = _as_4d(raw[1], "logits")
    return logits, edge, None, None


def _triple_to_unified(raw: object) -> AdapterOutput:
    """将 ``(logits, aux1, aux2)`` 输出转换为统一格式。

    Args:
        raw (object): 模型原始输出，期望为 ``tuple(logits, aux1, aux2)``。

    Returns:
        AdapterOutput: ``(logits, None, None, None)``。
    """
    if not (isinstance(raw, tuple) and len(raw) == 3):
        raise ValueError("Expected tuple(logits, aux1, aux2)")
    logits = _as_4d(raw[0], "logits")
    return logits, None, None, None


def _bit_changeformer_to_unified(raw: object) -> AdapterOutput:
    """将 BIT/ChangeFormer 输出转换为统一格式。

    BIT/ChangeFormer 不同实现可能返回单个 logits 张量，或返回张量列表/元组。
    该函数会直接使用该张量，或在序列输出时取最后一个张量作为 logits。

    Args:
        raw (object): 模型原始输出。

    Returns:
        AdapterOutput: ``(logits, None, None, None)``。
    """
    if isinstance(raw, torch.Tensor):
        logits = _as_4d(raw, "logits")
        return logits, None, None, None
    if isinstance(raw, (list, tuple)) and len(raw) > 0 and isinstance(raw[0], torch.Tensor):
        logits = _as_4d(raw[-1], "logits")
        return logits, None, None, None
    raise ValueError("Unexpected output type for BIT/ChangeFormer adapter")


def _cdnet_to_unified(raw: object) -> AdapterOutput:
    """将 ``model.cdnet.CDNet`` 的三分支输出映射为统一格式。

    Args:
        raw (object): ``CDNet`` 原始输出，期望为 ``(x, x8, x16)``。

    Returns:
        AdapterOutput: ``(x, None, None, None)``。
    """
    return _triple_to_unified(raw)


DEFAULT_RULES: Dict[str, AdapterRule] = {
    "USSFCNet": AdapterRule(_single_to_unified),
    "Net": AdapterRule(_single_to_unified),
    "SiamUnet_diff": AdapterRule(_single_to_unified),
    "SiamUnet_conc": AdapterRule(_single_to_unified),
    "mynet3": AdapterRule(_single_to_unified),
    "EATDer": AdapterRule(_edge_logits_pair_to_unified),
    "BASE_Transformer": AdapterRule(_bit_changeformer_to_unified),
}


class UnifiedCDAdapter(nn.Module):
    """将 baseline 模型包装为统一变化检测接口。

    统一 forward 输出格式：
        (logits, edge, feat1, feat2)
    当 baseline 不提供对应分支时，edge/feat1/feat2 可为 ``None``。
    """

    def __init__(self, base_model: nn.Module, rule: Optional[Callable[[object], AdapterOutput]] = None) -> None:
        """初始化统一适配器。

        Args:
            base_model (nn.Module): 要包装的 baseline 模型实例。
            rule (Optional[Callable[[object], AdapterOutput]]): 可选的自定义
                转换规则。若为 ``None``，则按 ``base_model`` 类名从
                ``DEFAULT_RULES`` 自动选择。

        Raises:
            KeyError: 未提供自定义规则且默认规则中找不到该类名时抛出。
        """
        super().__init__()
        self.base_model = base_model

        if rule is not None:
            self._rule = rule
            return

        # 明确使用包化后的 CDNet 类型判断，避免依赖旧的 model/CDNet.py。
        if isinstance(base_model, CDNet):
            self._rule = _cdnet_to_unified
            return

        cls_name = type(base_model).__name__
        if cls_name not in DEFAULT_RULES:
            raise KeyError(
                f"No default adapter rule for class '{cls_name}'. "
                "Please pass `rule=...` explicitly."
            )
        self._rule = DEFAULT_RULES[cls_name].fn

    def forward(self, image_a: Tensor4, image_b: Tensor4) -> AdapterOutput:
        """执行包装模型并转换为统一输出元组。

        Args:
            image_a (Tensor4): 时相 A 图像，形状 ``[B, C, H, W]``。
            image_b (Tensor4): 时相 B 图像，形状 ``[B, C, H, W]``。

        Returns:
            AdapterOutput: ``(logits, edge, feat1, feat2)``。
        """
        raw = self.base_model(image_a, image_b)
        logits, edge, feat1, feat2 = self._rule(raw)
        return logits, edge, feat1, feat2


def wrap_unified(base_model: nn.Module, rule: Optional[Callable[[object], AdapterOutput]] = None) -> UnifiedCDAdapter:
    """使用 ``UnifiedCDAdapter`` 包装 baseline 模型。

    Args:
        base_model (nn.Module): baseline 模型实例。
        rule (Optional[Callable[[object], AdapterOutput]]): 可选的自定义输出
            转换函数。

    Returns:
        UnifiedCDAdapter: 具有统一输出接口的包装模型。
    """

    return UnifiedCDAdapter(base_model=base_model, rule=rule)
