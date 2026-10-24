import json
import pathlib

import numpy

from lib.config.io import save_raw_config
from .binarizer_base import DataSample
from .phonemes_binarizer import PhonemeMetadataItem, PhonemeTimingBinarizer


class VoiceEventBinarizer(PhonemeTimingBinarizer):
    __data_attrs__ = ["frame_targets", "event_labels", "event_spans"]

    def resolve_data_dir(self) -> pathlib.Path:
        return self.config.voice_event_data_dir_resolved

    def build_vocabulary(self, metadata_list: list[PhonemeMetadataItem]):
        self.event_to_id = {event: index for index, event in enumerate(self.config.vocabulary.events, start=1)}
        with open(self.data_dir / "vocabulary.json", "w", encoding="utf8") as f:
            json.dump({"events": self.config.vocabulary.events}, f, ensure_ascii=False, indent=2)

    def save_auxiliary_files(self):
        save_raw_config(self.config.features.model_dump(), self.data_dir / "feature.yaml")

    def process_item(self, item: PhonemeMetadataItem) -> DataSample:
        length = self.get_frame_count(item.waveform_fn)
        durations = self.sec_dur_to_frame_dur(numpy.array(item.raw_durations, dtype=numpy.float64), length)
        targets = numpy.zeros(length, dtype=numpy.int64)
        labels, spans = [], []
        cursor = 0
        for symbol, duration in zip(item.raw_symbols, durations):
            start, end = cursor, cursor + int(duration)
            cursor = end
            event = self.event_to_id.get(symbol, 0)
            if event == 0 or end <= start:
                continue
            targets[start:end] = event
            labels.append(event)
            spans.append((start, end))
        return DataSample(
            path=item.waveform_fn.relative_to(self.data_dir).as_posix(),
            name=item.name,
            length=length,
            data={
                "frame_targets": targets,
                "event_labels": numpy.array(labels, dtype=numpy.int64),
                "event_spans": numpy.array(spans, dtype=numpy.int64).reshape((-1, 2)),
            },
        )
