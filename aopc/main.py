import pydantic
import typing as typ


class AopcConfig(pydantic.BaseModel):
    """AOPC configuration"""

    model: nn.module = pydantic.Field(..., description="Model")  # OR typ.Callable
    normalization: typ.Literal["beam", "exact"] | None = pydantic.Field(
        default=None, description="Normalization method"
    )
    word_map: list[int] = pydantic.Field(default=[], description="Word map")


# TODO: implement joblib Caching functionallity


class Aopc:
    def __init__(self, *args: typ.Any, **kwargs: typ.Any): ...

    def __call__(self, *args: typ.Any, **kwargs: typ.Any) -> typ.Any: ...
