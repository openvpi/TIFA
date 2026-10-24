import abc
import math
import pathlib
import random
from collections.abc import Callable

import librosa
import numpy
import torch

from lib.audio import load_audio
from lib.config.io import load_raw_config
from lib.config.schema import AugmentationConfig, BinarizerFeaturesConfig
from lib.feature.mel import StretchableMelSpectrogram
from lib.indexed_dataset import IndexedDataset
from lib.path_traversal import path_lengths
from lib.sequence_mutation import apply_mask_mutations, apply_sequence_edits
from lib.vocabulary import MASK_TOKEN, SPACE_TOKEN, NUM_RESERVED_TOKENS, Vocabulary
from .augmentation import (
    AugmentationContext,
    ComposedAugmentation,
    SpectrogramStretching,
    generate_seed,
    build_augmentation_chain,
)

__all__ = [
    "collate_nd",
    "concat_phoneme_timing_fields",
    "plan_concat_groups",
    "BaseDataset",
    "PhonemeTimingDataset",
    "VoiceEventDataset",
    "TextOnlyDataset",
    "DynamicBatchSampler",
    "ZippedDataLoader",
]


def _align_length(t: torch.Tensor, expected: int) -> torch.Tensor:
    """Slice or pad *t* along dim 0 to exactly *expected* frames."""
    n = t.shape[0]
    if n > expected:
        return t[:expected]
    if n < expected:
        return torch.nn.functional.pad(t, (0, 0, 0, expected - n), value=math.log(1e-5))
    return t


def collate_nd(values, pad_value=0, max_len=None, ndim=1):
    """
    Pad a list of Nd tensors on their first ``ndim`` dimensions and stack them
    into a (N+1)d tensor.
    """
    max_sizes = [
        max(v.size(d) for v in values)
        for d in range(ndim)
    ]
    if max_len is not None:
        max_sizes[0] = max_len
    remaining = values[0].shape[ndim:]
    size = (*max_sizes, *remaining)
    res = torch.full((len(values), *size), fill_value=pad_value, dtype=values[0].dtype, device=values[0].device)

    for i, v in enumerate(values):
        idx = [i] + [slice(v.size(d)) for d in range(ndim)]
        res[tuple(idx)] = v
    return res


def concat_phoneme_timing_fields(samples: list[dict]) -> dict:
    """Concatenate prepared phoneme-timing fields with shifted coordinates."""
    if not samples:
        raise ValueError("Cannot concatenate an empty phoneme-timing group.")

    frame_offsets = []
    token_offsets = []
    frame_cursor = 0
    token_cursor = 0
    for sample in samples:
        frame_offsets.append(frame_cursor)
        token_offsets.append(token_cursor)
        frame_cursor += int(sample["T"].item())
        token_cursor += int(sample["N"].item())

    shifted_spans = []
    shifted_regions = []
    for sample, frame_offset, token_offset in zip(
        samples,
        frame_offsets,
        token_offsets,
    ):
        spans = sample["spans"]
        shifted_spans.append(spans + frame_offset if frame_offset > 0 else spans)

        regions = sample["regions"]
        if token_offset > 0:
            regions = regions.clone()
            active = regions > 0
            regions[active] += token_offset
        shifted_regions.append(regions)

    return {
        "tokens": torch.cat([sample["tokens"] for sample in samples]),
        "spans": torch.cat(shifted_spans),
        "regions": torch.cat(shifted_regions),
        "frame_targets": torch.cat([sample["frame_targets"] for sample in samples]),
    }


def plan_concat_groups(
    indices: list[int],
    num_frames: Callable[[int], int],
    *,
    max_concat_size: int | None,
    max_concat_frames: int | None,
    dynamic_size: bool = False,
    rng: random.Random | None = None,
) -> list[list[int]]:
    """Partition ordered indices into concat groups under size and frame limits."""
    if not indices:
        return []
    if max_concat_size is None and max_concat_frames is None:
        return [[index] for index in indices]
    if dynamic_size and max_concat_size is not None and rng is None:
        raise ValueError("Dynamic concat grouping requires a random generator.")

    def next_target_size() -> int | None:
        if dynamic_size and max_concat_size is not None:
            return rng.randint(1, max_concat_size)
        return max_concat_size

    target_size = next_target_size()
    groups = []
    current = []
    current_frames = 0
    for index in indices:
        frames = num_frames(index)
        exceed_size = target_size is not None and len(current) >= target_size
        exceed_frames = max_concat_frames is not None and current_frames + frames > max_concat_frames
        if current and (exceed_size or exceed_frames):
            groups.append(current)
            current = []
            current_frames = 0
            target_size = next_target_size()
        current.append(index)
        current_frames += frames
    if current:
        groups.append(current)
    return groups


class BaseDataset(torch.utils.data.Dataset, abc.ABC):
    __non_zero_paddings__ = {
        "spectrogram": math.log(1e-5),
        "spectrogram_dirty": math.log(1e-5),
    }
    __multi_dims__: dict[str, int] = {}

    def __init__(
        self,
        data_dir: pathlib.Path,
        prefix: str,
        *,
        augmentation_config: AugmentationConfig = None,
        augmentation_deterministic: bool = False,
        augmentation_return_dirty: bool = False,
        augmentation_seed: int | None = None,
        max_concat_size: int | None = None,
        max_concat_frames: int | None = None,
        concat_dynamic_size: bool = True,
        concat_deterministic: bool = False,
    ):
        super().__init__()
        self.info = {
            k: v
            for k, v in numpy.load(data_dir / f"{prefix}.info.npz").items()
        }
        self.data_dir = data_dir
        self.data = IndexedDataset(data_dir, prefix)
        self.epoch = torch.multiprocessing.Value("i", 0)
        self.augmentation_config = augmentation_config
        self.augmentation_deterministic = augmentation_deterministic
        self.augmentation_return_dirty = augmentation_return_dirty
        self.augmentation_seed = augmentation_seed
        self.augmentation_chains: dict[int, ComposedAugmentation] = {}
        self.mel_spectrogram = None
        self.max_concat_size = max_concat_size
        self.max_concat_frames = max_concat_frames
        self.concat_dynamic_size = concat_dynamic_size
        self.concat_deterministic = concat_deterministic
        self._n_original = len(self.info["lengths"])
        self._group_indices: list[list[int]] | None = None
        self._group_epoch: int = 0
        self._chain_epoch: int = -1
        self._setup()
        if self.max_concat_size is not None or self.max_concat_frames is not None:
            self._form_groups(0)
            self._group_epoch = 0

    def __getitem__(self, index):
        if self._group_indices is not None:
            current_epoch = self.epoch.value
            if self._group_epoch != current_epoch:
                self._form_groups(current_epoch)
                self._group_epoch = current_epoch
            if self._should_rebuild_chains(current_epoch):
                self._build_chains(current_epoch)
            samples = [self._get_single_item(i) for i in self._group_indices[index]]
            result = self.concat_samples(samples)
            result["_idx"] = torch.tensor(index, dtype=torch.long)
            result["_augmentation"] = {}
            return result
        if self._should_rebuild_chains(self.epoch.value):
            self._build_chains(self.epoch.value)
        return self._get_single_item(index)

    def __len__(self):
        if self._group_indices is not None:
            return len(self._group_indices)
        return self._n_original

    def set_epoch(self, epoch: int):
        self.epoch.value = epoch
        if self.max_concat_size is not None or self.max_concat_frames is not None:
            self._form_groups(epoch)
            self._group_epoch = epoch
        if self.augmentation_config is not None and not self.augmentation_deterministic:
            self._build_chains(epoch)

    def num_frames(self, index: int) -> int:
        if self._group_indices is not None:
            return sum(
                self._single_num_frames(i) for i in self._group_indices[index]
            )
        return self._single_num_frames(index)

    def _single_num_frames(self, index: int) -> int:
        base_len = int(self.info["lengths"][index])
        chain = self.augmentation_chains.get(index)
        if chain is not None:
            for t in chain.transforms:
                if isinstance(t, SpectrogramStretching) and t.speed is not None:
                    return max(1, int(base_len / t.speed))
        return base_len

    def get_metadata(self, key: str, index: int):
        """Proxy for info[key][index] that handles group-to-original mapping."""
        if self._group_indices is None:
            return self.info[key][index]

        values = [self.info[key][orig_idx] for orig_idx in self._group_indices[index]]
        if key == "item_paths":
            return "Concatenation of " + ", ".join(str(org_idx) for org_idx in self._group_indices[index])
        return sum(int(v) for v in values)

    def concat_samples(self, samples: list[dict]) -> dict:
        """Prepare and merge individual samples into one concatenated sample.

        Common acoustic fields are owned here. Subclasses only merge their
        dataset-specific alignment or path fields.
        """
        if not samples:
            raise ValueError("Cannot concatenate an empty sample list.")

        processed = [self._prepare_item(sample) for sample in samples]
        if len(processed) == 1:
            return processed[0]

        result = self._concat_common_samples(processed)
        specific = self._concat_specific_samples(processed)
        overlapping = result.keys() & specific.keys()
        if overlapping:
            raise ValueError(f"Dataset-specific concatenation returned common fields: " f"{sorted(overlapping)}")
        result.update(specific)
        return result

    def _concat_common_samples(self, samples: list[dict]) -> dict:
        result = {}
        for key in ("spectrogram", "spectrogram_dirty", "f0"):
            present = [key in sample for sample in samples]
            if not any(present):
                if key == "spectrogram":
                    raise KeyError("Every sample must contain 'spectrogram'.")
                continue
            if not all(present):
                raise ValueError(f"Common field '{key}' is missing from some samples.")
            for sample in samples:
                if sample[key].shape[0] != int(sample["T"].item()):
                    raise ValueError(f"Common field '{key}' does not match sample frame length.")
            result[key] = torch.cat([sample[key] for sample in samples], dim=0)

        result["T"] = torch.stack([sample["T"] for sample in samples]).sum()
        result["N"] = torch.stack([sample["N"] for sample in samples]).sum()
        return result

    @abc.abstractmethod
    def _prepare_item(self, sample: dict) -> dict:
        """Add derived fields required before concatenation."""

    @abc.abstractmethod
    def _concat_specific_samples(self, samples: list[dict]) -> dict:
        """Merge dataset-specific fields from two or more prepared samples."""

    def _setup(self):
        feature_raw = load_raw_config(self.data_dir / "feature.yaml")
        feature_cfg = BinarizerFeaturesConfig.model_validate(feature_raw)
        self.sample_rate = feature_cfg.audio_sample_rate
        self.mel_spectrogram = StretchableMelSpectrogram(
            sample_rate=feature_cfg.audio_sample_rate,
            n_mels=feature_cfg.spectrogram.num_bins,
            n_fft=feature_cfg.fft_size,
            win_length=feature_cfg.win_size,
            hop_length=feature_cfg.hop_size,
            fmin=feature_cfg.spectrogram.fmin,
            fmax=feature_cfg.spectrogram.fmax,
            clip_val=1e-9,
        ).eval()

        if self.augmentation_config is None:
            return

        if self.augmentation_deterministic:
            self._build_chains(0)

    def _form_groups(self, epoch: int) -> None:
        seed = 42 if self.concat_deterministic else 42 + epoch
        rng = random.Random(seed)
        indices = list(range(self._n_original))
        rng.shuffle(indices)
        self._group_indices = plan_concat_groups(
            indices,
            self._single_num_frames,
            max_concat_size=self.max_concat_size,
            max_concat_frames=self.max_concat_frames,
            dynamic_size=self.concat_dynamic_size,
            rng=rng,
        )

    def _build_chains(self, epoch: int) -> None:
        if self.augmentation_deterministic:
            seed = generate_seed(sorted(self.info.keys()))
            generator = numpy.random.default_rng(seed)
        elif self.augmentation_seed is not None:
            generator = numpy.random.default_rng(self.augmentation_seed + epoch)
        else:
            generator = numpy.random.default_rng()
        self.augmentation_chains.clear()
        for index in range(self._n_original):
            self.augmentation_chains[index] = build_augmentation_chain(
                self.augmentation_config,
                mel_spectrogram=self.mel_spectrogram,
                generator=generator,
            )
        self._chain_epoch = epoch

    def _should_rebuild_chains(self, epoch: int) -> bool:
        return (
            self.augmentation_config is not None
            and not self.augmentation_deterministic
            and self._chain_epoch != epoch
        )

    def _get_single_item(self, index):
        sample = self.data[index]
        waveform = self._load_waveform(index)
        spectrogram_clean = None

        augmentation = {}
        if self.augmentation_config is not None:
            chain = self.augmentation_chains[index]

            wf_tensor = torch.from_numpy(waveform).unsqueeze(0)
            if self.augmentation_return_dirty:
                spectrogram_clean = self.mel_spectrogram(wf_tensor).squeeze(0).T

            ctx = AugmentationContext(waveform=waveform, sr=self.sample_rate)
            chain.apply(ctx)
            spectrogram = ctx.spectrogram
            augmentation = chain.args_dict()
        else:
            wf_tensor = torch.from_numpy(waveform).unsqueeze(0)
            spectrogram = self.mel_spectrogram(wf_tensor).squeeze(0).T

        spectrogram = torch.clamp(spectrogram, min=math.log(1e-5))
        expected_len = int(self.info["lengths"][index])
        spectrogram = _align_length(spectrogram, expected_len)

        if self.augmentation_config is not None and self.augmentation_return_dirty:
            spectrogram_clean = torch.clamp(spectrogram_clean, min=math.log(1e-5))
            spectrogram_clean = _align_length(spectrogram_clean, expected_len)
            sample["spectrogram"] = spectrogram_clean
            sample["spectrogram_dirty"] = spectrogram
        else:
            sample["spectrogram"] = spectrogram

        if "f0" in sample:
            f0_val = sample["f0"].float()
            if f0_val.shape[0] != spectrogram.shape[0]:
                f0_val = torch.nn.functional.interpolate(
                    f0_val.view(1, 1, -1), size=spectrogram.shape[0],
                    mode="linear", align_corners=True,
                ).view(-1)
            sample["f0"] = f0_val

        return {
            "_idx": torch.tensor(index, dtype=torch.long),
            "_augmentation": augmentation,
            **sample,
        }

    def _load_waveform(self, index: int) -> numpy.ndarray:
        waveform_fn = self.data_dir / str(self.info["item_paths"][index])
        waveform, sr = load_audio(waveform_fn)
        if sr != self.sample_rate:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)
        return waveform

    @classmethod
    def collate(cls, samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        for s in samples:
            s.pop("_augmentation", None)
        batch = {
            "size": len(samples),
            "indices": torch.stack([s.pop("_idx") for s in samples]),
        }
        if len(samples) == 0:
            return batch
        for key, value in samples[0].items():
            if value.ndim == 0:
                batch[key] = torch.stack([s[key] for s in samples])
            else:
                pad_value = cls.__non_zero_paddings__.get(key, 0)
                ndim = cls.__multi_dims__.get(key, 1)
                batch[key] = collate_nd([s[key] for s in samples], pad_value=pad_value, ndim=ndim)
        return batch


class PhonemeTimingDataset(BaseDataset):
    def __init__(
            self,
            *args,
            augmentation_return_mutated: bool = False,
            augmentation_config=None,
            **kwargs,
    ):
        # Time stretching is not supported for token-spectrogram aligned datasets.
        if augmentation_config is not None:
            augmentation_config = augmentation_config.drop("time_stretching")
        super().__init__(
            *args,
            augmentation_config=augmentation_config,
            **kwargs,
        )
        self._augmentation_return_mutated = augmentation_return_mutated
        self._vocab_size: int | None = None
        if (
                self.augmentation_config is not None
                and self.augmentation_config.sequence_edit.enabled
        ):
            self._vocab_size = Vocabulary.from_file(
                self.data_dir / "vocabulary.json"
            ).vocab_size

    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)
        if self._group_indices is not None:
            return self._apply_mutation(sample)
        sample = self._prepare_item(sample)
        return self._apply_mutation(sample)

    def _prepare_item(self, sample: dict) -> dict:
        # Per-frame ground-truth token IDs, computed before any token edits.
        regions_orig = sample["regions"]
        tokens_orig = sample["tokens"]
        T = int(regions_orig.shape[0])
        frame_targets = torch.zeros(T, dtype=torch.long)
        valid = regions_orig > 0
        if valid.any():
            idx = regions_orig[valid] - 1
            frame_targets[valid] = tokens_orig[idx]
        sample["frame_targets"] = frame_targets

        sample["T"] = torch.tensor(sample["spectrogram"].shape[0], dtype=torch.long)
        sample["N"] = torch.tensor(sample["tokens"].shape[0], dtype=torch.long)
        return sample

    def _apply_mutation(self, sample: dict) -> dict:
        seed = sample["_idx"].item() if self.augmentation_deterministic else None
        rng = numpy.random.default_rng(seed)

        edit_cfg = self.augmentation_config and self.augmentation_config.sequence_edit
        mask_cfg = self.augmentation_config and self.augmentation_config.token_masking
        # Determine whether and how to mutate
        mutation_type = None
        if self._vocab_size is not None and edit_cfg and rng.random() < edit_cfg.prob:
            mutation_type = "edit"
        elif mask_cfg is not None and mask_cfg.enabled and rng.random() < mask_cfg.prob:
            mutation_type = "mask"

        sample["is_mlm"] = torch.tensor(False, dtype=torch.bool)

        if self._augmentation_return_mutated:
            # Keep originals clean; emit mutated copy
            sample["token_targets"] = sample["tokens"].clone()
            if mutation_type == "edit":
                mutated_tok, _, _, mutated_targets = apply_sequence_edits(
                    tokens=sample["tokens"],
                    spans=sample["spans"],
                    regions=sample["regions"],
                    min_token=NUM_RESERVED_TOKENS,
                    max_token=self._vocab_size - 1,
                    p_sub=edit_cfg.p_sub,
                    p_del=edit_cfg.p_del,
                    p_ins=edit_cfg.p_ins,
                    rng=rng,
                )
                sample["tokens_mutated"] = mutated_tok
                sample["token_targets_mutated"] = mutated_targets
            elif mutation_type == "mask":
                mutated_tok, _, _, mutated_targets = apply_mask_mutations(
                    tokens=sample["tokens"],
                    spans=sample["spans"],
                    regions=sample["regions"],
                    p_mask=mask_cfg.p_mask,
                    p_insert=mask_cfg.p_insert,
                    p_chain=mask_cfg.p_chain,
                    max_chain=mask_cfg.max_chain,
                    mask_token=MASK_TOKEN,
                    space_token=SPACE_TOKEN,
                    rng=rng,
                )
                sample["tokens_mutated"] = mutated_tok
                sample["token_targets_mutated"] = mutated_targets
            else:
                sample["tokens_mutated"] = sample["tokens"].clone()
                sample["token_targets_mutated"] = sample["tokens"].clone()
        else:
            # Mutate in-place
            if mutation_type == "edit":
                (
                    sample["tokens"], sample["spans"], sample["regions"],
                    sample["token_targets"],
                ) = apply_sequence_edits(
                    tokens=sample["tokens"],
                    spans=sample["spans"],
                    regions=sample["regions"],
                    min_token=NUM_RESERVED_TOKENS,
                    max_token=self._vocab_size - 1,
                    p_sub=edit_cfg.p_sub,
                    p_del=edit_cfg.p_del,
                    p_ins=edit_cfg.p_ins,
                    rng=rng,
                )
                sample["N"] = torch.tensor(sample["tokens"].shape[0], dtype=torch.long)
            elif mutation_type == "mask":
                (
                    sample["tokens"], sample["spans"], sample["regions"],
                    sample["token_targets"],
                ) = apply_mask_mutations(
                    tokens=sample["tokens"],
                    spans=sample["spans"],
                    regions=sample["regions"],
                    p_mask=mask_cfg.p_mask,
                    p_insert=mask_cfg.p_insert,
                    p_chain=mask_cfg.p_chain,
                    max_chain=mask_cfg.max_chain,
                    mask_token=MASK_TOKEN,
                    space_token=SPACE_TOKEN,
                    rng=rng,
                )
                sample["is_mlm"] = torch.tensor(True, dtype=torch.bool)
                sample["N"] = torch.tensor(sample["tokens"].shape[0], dtype=torch.long)
            else:
                sample["token_targets"] = sample["tokens"].clone()
        return sample

    def _concat_specific_samples(self, samples: list[dict]) -> dict:
        return concat_phoneme_timing_fields(samples)


class VoiceEventDataset(BaseDataset):
    """Frame and event targets from voice-event binary data."""

    def __init__(self, *args, augmentation_config=None, **kwargs):
        if augmentation_config is not None:
            # Follow supervised timing-data augmentation: preserve frame coordinates.
            augmentation_config = augmentation_config.drop("time_stretching", "sequence_edit", "token_masking")
        super().__init__(*args, augmentation_config=augmentation_config, **kwargs)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        return sample if self._group_indices is not None else self._prepare_item(sample)

    def _prepare_item(self, sample: dict) -> dict:
        result = {key: sample[key] for key in ("_idx", "_augmentation", "spectrogram", "spectrogram_dirty") if key in sample}
        result.update(
            frame_targets=sample["frame_targets"],
            event_labels=sample["event_labels"],
            event_spans=sample["event_spans"],
            T=torch.tensor(len(sample["frame_targets"]), dtype=torch.long),
            N=torch.tensor(len(sample["event_labels"]), dtype=torch.long),
        )
        return result

    def _concat_specific_samples(self, samples: list[dict]) -> dict:
        spans = []
        offset = 0
        for sample in samples:
            spans.append(sample["event_spans"] + offset)
            offset += int(sample["T"])
        return {
            "frame_targets": torch.cat([sample["frame_targets"] for sample in samples]),
            "event_labels": torch.cat([sample["event_labels"] for sample in samples]),
            "event_spans": torch.cat(spans),
        }


class TextOnlyDataset(BaseDataset):
    __multi_dims__ = {
        **BaseDataset.__multi_dims__,
        "paths": 2,
        "groups": 2,
        "candidates": 2,
    }

    def _setup(self):
        first = self.data.dset.get("0")
        if first is not None and (
            not {"paths", "words", "groups", "candidates"}.issubset(first.keys())
            or first["paths"].ndim != 2
            or first["candidates"].dtype != numpy.dtype(bool)
        ):
            raise ValueError("Text-only data requires complete candidate tensor grids; re-binarize the dataset.")
        super()._setup()

    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)
        if self._group_indices is not None:
            return sample  # already processed by concat_samples
        return self._prepare_item(sample)

    # noinspection PyMethodMayBeStatic
    def _prepare_item(self, sample: dict) -> dict:
        sample["T"] = torch.tensor(sample["spectrogram"].shape[0], dtype=torch.long)
        sample["N"] = path_lengths(
            sample["paths"].unsqueeze(0), sample["words"].unsqueeze(0), sample["candidates"].unsqueeze(0),
        )[0]
        return sample

    def _concat_specific_samples(self, samples: list[dict]) -> dict:
        width = max(sample["paths"].shape[1] for sample in samples)
        paths, words, groups, candidates = [], [], [], []
        offset = 0
        for sample in samples:
            padding = (0, width - sample["paths"].shape[1])
            paths.append(torch.nn.functional.pad(sample["paths"], padding))
            groups.append(torch.nn.functional.pad(sample["groups"], padding))
            candidates.append(torch.nn.functional.pad(sample["candidates"], padding))
            owners = sample["words"]
            words.append(owners + (owners > 0).long() * offset)
            offset += sample["candidates"].shape[0]
        return {
            "paths": torch.cat(paths),
            "words": torch.cat(words),
            "groups": torch.cat(groups),
            "candidates": torch.cat(candidates),
        }


class DynamicBatchSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(
            self,
            dataset: BaseDataset,
            max_batch_size: int,
            max_batch_frames: int,
            sort_by_len: bool = True,
            frame_count_grid: int = 1,
            batch_count_multiple_of: int = 1,
            reassign_batches: bool = True,
            shuffle_batches: bool = True,
            seed: int = 0,
            target_num_batches: int | None = None,
    ):
        if torch.distributed.is_initialized():
            num_replicas = None
            rank = None
        else:
            num_replicas = 1
            rank = 0
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=False,
            seed=seed,
            drop_last=False,
        )
        self.dataset = dataset
        self.max_batch_size = max_batch_size
        self.max_batch_frames = max_batch_frames
        self.sort_by_len = sort_by_len
        self.frame_count_grid = frame_count_grid
        self.batch_count_multiple_of = batch_count_multiple_of
        self.reassign_batches = reassign_batches
        self.shuffle_batches = shuffle_batches
        self.generator: torch.Generator = torch.Generator().manual_seed(seed)
        self.batches: list[list[int]] = None
        self.formed = None
        self.target_num_batches = target_num_batches

    def __iter__(self):
        self.form_batches()
        return iter(self.batches)

    def __len__(self):
        self.form_batches()
        return len(self.batches)

    def set_epoch(self, epoch: int):
        super().set_epoch(epoch)
        self.generator = torch.Generator().manual_seed(self.seed + epoch)

    def permutation(self, n: int) -> list[int]:
        perm = torch.randperm(n, generator=self.generator).tolist()
        return perm

    def form_batches(self):
        if self.formed == self.epoch + self.seed:
            return

        self.dataset.set_epoch(self.epoch)
        lengths = [self.dataset.num_frames(i) for i in range(len(self.dataset))]
        if self.sort_by_len:
            sorted_indices = sorted(
                self.permutation(len(lengths)),
                key=lambda x: lengths[x] // self.frame_count_grid, reverse=True
            )
        else:
            sorted_indices = list(range(len(lengths)))

        total_items = len(sorted_indices)
        if self.target_num_batches is not None:
            target_items_per_batch = max(1, math.ceil(total_items / self.target_num_batches))
            effective_max_batch_size = min(self.max_batch_size, target_items_per_batch)
        else:
            effective_max_batch_size = self.max_batch_size

        def batch_full(batch_: list[int], new_index_: int):
            if len(batch_) >= effective_max_batch_size:
                return True
            max_len = max(lengths[new_index_], max((lengths[i] for i in batch_), default=0))
            if max_len * (len(batch_) + 1) > self.max_batch_frames:
                return True
            return False

        def _greedy_pack() -> list[list[int]]:
            batches_: list[list[int]] = []
            current_batch = []
            for _idx in sorted_indices:
                sample_length = lengths[_idx]
                if sample_length > self.max_batch_frames:
                    raise ValueError(
                        f"Sample length {sample_length} exceeds max batch frames {self.max_batch_frames}."
                    )
                if batch_full(current_batch, _idx):
                    batches_.append(current_batch)
                    current_batch = []
                current_batch.append(_idx)
            if current_batch:
                batches_.append(current_batch)
            return batches_

        batches = _greedy_pack()

        if self.target_num_batches is not None:
            # Merge: too many batches  --  merge the smallest adjacent pairs
            while len(batches) > self.target_num_batches:
                best_i = -1
                best_size = float("inf")
                for i in range(len(batches) - 1):
                    combined = len(batches[i]) + len(batches[i + 1])
                    if combined > self.max_batch_size:
                        continue
                    combined_len = max(
                        max((lengths[j] for j in batches[i]), default=0),
                        max((lengths[j] for j in batches[i + 1]), default=0),
                    )
                    if combined_len * combined > self.max_batch_frames:
                        continue
                    if combined < best_size:
                        best_size = combined
                        best_i = i
                if best_i < 0:
                    break  # no more mergeable pairs
                batches[best_i].extend(batches[best_i + 1])
                del batches[best_i + 1]

            # Reduce: too few batches  --  halve effective size and repack
            while len(batches) < self.target_num_batches:
                if effective_max_batch_size <= 1:
                    raise RuntimeError(
                        f"Cannot form {self.target_num_batches} batches from "
                        f"{total_items} items: aux dataset too small. "
                        f"Reduce aux_ratio or add more data."
                    )
                effective_max_batch_size = max(1, effective_max_batch_size // 2)
                batches = _greedy_pack()

        multiple_of = self.num_replicas * self.batch_count_multiple_of
        remainder = (multiple_of - (len(batches) % multiple_of)) % multiple_of
        if self.reassign_batches:
            new_batch = []
            while remainder > 0:
                num_batches = len(batches)
                perm = self.permutation(num_batches)
                batches = [batches[i] for i in perm]
                modified = False
                idx = 0
                while remainder > 0 and idx < num_batches:
                    batch = batches[idx]
                    if len(batch) > 1:
                        item = batch[-1]
                        if batch_full(new_batch, item):
                            batches.append(new_batch)
                            new_batch = []
                            modified = True
                            remainder -= 1
                            if remainder == 0:
                                break
                        batch.pop()
                        new_batch.append(item)
                    idx += 1
                if not modified:
                    if len(new_batch) > 0:
                        batches.append(new_batch)
                        new_batch = []
                        remainder -= 1
                    if remainder > 0:
                        raise RuntimeError(
                            f"Unable to reassign batches to meet the required multiple count of {multiple_of}."
                        )
        else:
            batches += [[]] * remainder

        if self.shuffle_batches and self.epoch != 0:
            perm = self.permutation(len(batches))
            batches = [batches[i] for i in perm]
        elif self.sort_by_len:
            batches = sorted(
                batches,
                key=lambda b: len(b) * max(lengths[i] for i in b),
                reverse=True
            )

        batches = [b for i, b in enumerate(batches) if i % self.num_replicas == self.rank]

        self.batches = batches
        self.formed = self.epoch + self.seed


class ZippedDataLoader:
    """
    Wraps a main DataLoader, optionally zipped with an aux DataLoader.
    Batches are always wrapped as ``{"main": ..., "size": ...}``, where
    ``"size"`` is the main batch item count. When aux is active, ``"aux"`` and
    ``"aux_size"`` are included, with ``"aux_size"`` counting raw aux items
    before filtering or concatenation. ``num_batches`` keeps the epoch length
    fixed when the main sampler switches to its smaller active-phase budget.
    """

    def __init__(
        self,
        main_dataloader: torch.utils.data.DataLoader,
        aux_dataloader: torch.utils.data.DataLoader | None = None,
        aux_sampler: DynamicBatchSampler | None = None,
        aux_warmup_epochs: int = 0,
        num_batches: int | None = None,
    ):
        if aux_dataloader is not None and aux_sampler is None:
            raise ValueError("aux_sampler is required when aux_dataloader is provided.")
        self.main_dl = main_dataloader
        self.aux_dl = aux_dataloader
        self.aux_sampler = aux_sampler
        self.aux_warmup_epochs = aux_warmup_epochs
        self.num_batches = num_batches
        if self.num_batches is not None and self.num_batches <= 0:
            raise ValueError("num_batches must be positive.")

    def __iter__(self):
        num_batches = len(self.main_dl) if self.num_batches is None else self.num_batches
        if num_batches <= 0:
            raise RuntimeError("Main dataloader must produce at least one batch.")
        main_iter = iter(self.main_dl)
        aux_iter = None
        aux_active = self.aux_dl is not None and self.aux_sampler.epoch >= self.aux_warmup_epochs
        if aux_active:
            self.aux_sampler.target_num_batches = num_batches
            self.aux_sampler.formed = None
            aux_iter = iter(self.aux_dl)

        for batch_index in range(num_batches):
            try:
                main_batch = next(main_iter)
            except StopIteration as exc:
                raise RuntimeError(
                    f"Main dataloader produced {batch_index} batches; " f"expected {num_batches}."
                ) from exc
            if aux_iter is not None:
                try:
                    aux_batch = next(aux_iter)
                except StopIteration:
                    self.aux_sampler.formed = None
                    aux_iter = iter(self.aux_dl)
                    aux_batch = next(aux_iter)
                yield {
                    "main": main_batch,
                    "aux": aux_batch,
                    "size": main_batch["size"],
                    "aux_size": aux_batch["size"],
                }
            else:
                yield {
                    "main": main_batch,
                    "size": main_batch["size"],
                }

    def __len__(self):
        return len(self.main_dl) if self.num_batches is None else self.num_batches
