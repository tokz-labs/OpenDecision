from .common import CROSS_ENCODER, PACKED, ModelConfig, ModelOutput, pad_groups
from .cross_encoder import CrossEncoder
from .packed import PackedEncoder


def build_model(cfg: ModelConfig, backbone_config=None):
    """Instantiate the architecture named in ``cfg`` (pretrained backbone unless a config is given)."""
    cls = CrossEncoder if cfg.architecture == CROSS_ENCODER else PackedEncoder
    return cls(cfg, backbone_config=backbone_config)


__all__ = [
    "CROSS_ENCODER",
    "PACKED",
    "CrossEncoder",
    "ModelConfig",
    "ModelOutput",
    "PackedEncoder",
    "build_model",
    "pad_groups",
]
