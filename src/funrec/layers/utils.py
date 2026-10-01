from typing import Any

import numpy as np
import torch


def concat_fun(inputs: list[torch.Tensor], axis: int = -1) -> torch.Tensor:
    """沿指定维度拼接张量；只有一个张量时直接返回该张量。

    参数:
        inputs: 待拼接的张量列表。
        axis: 拼接维度，默认使用最后一个维度。

    返回:
        拼接后的张量。
    """
    if len(inputs) == 1:
        return inputs[0]
    else:
        return torch.cat(inputs, dim=axis)


def slice_arrays(
    arrays: Any | list[Any] | None,
    start: int | list[int] | np.ndarray | None = None,
    stop: int | None = None,
) -> Any | list[Any]:
    """按范围或索引集合切分一个数组或一组数组。

    参数:
        arrays: 单个类数组对象、类数组对象列表或 ``None``。
        start: 起始位置，或需要选取的索引列表/数组。
        stop: 结束位置；``start`` 为索引列表时必须为 ``None``。

    返回:
        与输入结构对应的切片；输入为 ``None`` 时返回 ``[None]``。

    异常:
        ValueError: ``start`` 为列表且同时提供了 ``stop``。
    """

    if arrays is None:
        return [None]

    if isinstance(arrays, np.ndarray):
        arrays = [arrays]

    if isinstance(start, list) and stop is not None:
        raise ValueError(
            "The stop argument has to be None if the value of start is a list."
        )
    elif isinstance(arrays, list):
        if hasattr(start, "__len__"):
            # HDF5 数据集仅支持使用列表作为索引
            if hasattr(start, "shape"):
                start = start.tolist()
            return [None if x is None else x[start] for x in arrays]
        else:
            if len(arrays) == 1:
                return arrays[0][start:stop]
            return [None if x is None else x[start:stop] for x in arrays]
    else:
        if hasattr(start, "__len__"):
            if hasattr(start, "shape"):
                start = start.tolist()
            return arrays[start]
        elif hasattr(start, "__getitem__"):
            return arrays[start:stop]
        else:
            return [None]
