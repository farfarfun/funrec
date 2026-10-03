from funrec.models.b2000 import BaseModel


class Empty(BaseModel):
    """空白占位模型，直接复用 ``BaseModel`` 的构造逻辑，不添加任何额外网络结构，
    可用于快速验证输入管道或作为自定义模型的起始模板。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
