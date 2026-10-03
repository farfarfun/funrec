from funrec.models.b2000 import BaseModel


class SDM(BaseModel):
    """序列深度匹配模型（SDM）的占位实现，当前直接复用 ``BaseModel`` 的构造逻辑，
    尚未实现短期/长期兴趣融合等 SDM 专属结构。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
