from torch import Tensor, nn

from lib.config.schema import ModelConfig
from lib.reflection import build_object_from_class_name
from modules.forced_alignment import ARCHITECTURE_ERROR_MSG


class VoiceEventDetectionModel(nn.Module):
    """Single-stream acoustic model with a softmax frame classification head."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        expected = type(self).__name__
        if config.arch != expected:
            raise ValueError(ARCHITECTURE_ERROR_MSG.format(expected=expected, actual=config.arch))
        self.num_classes = config.num_classes
        self.backbone = build_object_from_class_name(
            config.backbone.cls, nn.Module,
            in_dim=config.in_dim,
            out_dim=self.num_classes,
            **config.backbone.kwargs,
        )

    def forward(self, spectrogram: Tensor, mask: Tensor | None = None) -> Tensor:
        """Return [B, T, num_classes] logits, with background at index 0."""
        return self.backbone(spectrogram, mask=mask)
