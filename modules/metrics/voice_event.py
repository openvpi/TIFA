from collections.abc import Sequence

import torch
from torch import Tensor
from torchmetrics import Metric


def frame_labels_to_events(labels: Sequence[int]) -> list[tuple[int, int, int]]:
    """Decode non-background contiguous runs into (class, onset, offset)."""
    events = []
    previous = 0
    start = 0
    for index, label in enumerate(labels):
        if label != previous:
            if previous != 0:
                events.append((previous, start, index))
            previous, start = label, index
    if previous != 0:
        events.append((previous, start, len(labels)))
    return events


def match_events(
    predicted: Sequence[tuple[int, int]],
    target: Sequence[tuple[int, int]],
    collar: float,
    require_offset: bool = False,
) -> tuple[int, int, int]:
    """BreathLab matching: nearest available onset, greedily one-to-one."""
    used = set()
    for onset, offset in sorted(predicted, key=lambda event: event[0]):
        best_index, best_distance = None, float("inf")
        for index, (gt_onset, gt_offset) in enumerate(target):
            if index in used:
                continue
            distance = abs(onset - gt_onset)
            if distance > collar or (require_offset and abs(offset - gt_offset) > collar):
                continue
            if distance < best_distance:
                best_index, best_distance = index, distance
        if best_index is not None:
            used.add(best_index)
    tp = len(used)
    return tp, len(predicted) - tp, len(target) - tp


def _prf(counts: Tensor) -> Tensor:
    tp, fp, fn = counts.double().unbind(dim=-1)
    return torch.stack((
        tp / (tp + fp).clamp_min(1),
        tp / (tp + fn).clamp_min(1),
        2 * tp / (2 * tp + fp + fn).clamp_min(1),
    ), dim=-1).float()


class VoiceEventPRF(Metric):
    """Pool frame/event counts across clips and ranks before computing P/R/F1.

    Event IDs start at 1. Background (0) is excluded from per-class metrics
    and micro/macro summaries.
    """

    full_state_update = False
    is_differentiable = False
    higher_is_better = True

    def __init__(self, event_classes, collar: float = 20, require_offset: bool = False, postfix: str = ""):
        super().__init__()
        self.event_classes = tuple(event_classes)
        self.collar = collar
        self.require_offset = require_offset
        self.postfix = postfix
        shape = (len(event_classes), 3)
        self.add_state("frame_counts", default=torch.zeros(shape, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("event_counts", default=torch.zeros(shape, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, prediction: Tensor, target: Tensor, mask: Tensor, event_labels: Tensor, event_spans: Tensor):
        for class_id in range(1, len(self.event_classes) + 1):
            predicted = (prediction == class_id) & mask
            actual = (target == class_id) & mask
            self.frame_counts[class_id - 1] += torch.stack((
                (predicted & actual).sum(), (predicted & ~actual).sum(), (~predicted & actual).sum(),
            ))
        predictions = prediction.masked_fill(~mask, 0).detach().cpu().tolist()
        labels = event_labels.detach().cpu().tolist()
        spans = event_spans.detach().cpu().tolist()
        counts = torch.zeros_like(self.event_counts, device="cpu")
        for frames, classes, intervals in zip(predictions, labels, spans):
            events = frame_labels_to_events(frames)
            for class_id in range(1, len(self.event_classes) + 1):
                pred_events = [(start, end) for label, start, end in events if label == class_id]
                gt_events = [(start, end) for label, (start, end) in zip(classes, intervals)
                             if label == class_id and end > start]
                counts[class_id - 1] += torch.tensor(match_events(
                    pred_events, gt_events, self.collar, self.require_offset,
                ))
        self.event_counts += counts.to(self.event_counts.device)

    def compute(self) -> dict[str, Tensor]:
        result = {}
        for prefix, counts in (("Frame", self.frame_counts), ("Event", self.event_counts)):
            per_class = _prf(counts)
            for index, name in enumerate(self.event_classes):
                for score, value in zip(("P", "R", "F1"), per_class[index]):
                    result[f"{prefix}_{score}_{name}{self.postfix}"] = value
            for average, values in (("micro", _prf(counts.sum(dim=0))), ("macro", per_class.mean(dim=0))):
                for score, value in zip(("P", "R", "F1"), values):
                    result[f"{prefix}_{score}_{average}{self.postfix}"] = value
        return result
