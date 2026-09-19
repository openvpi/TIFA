from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TypeAlias

from ..preprocessors.base import Preprocessor


@dataclass
class G2PGroup:
    """A contiguous pronunciation group and its pronunciation script."""

    script: str
    phonemes: list[str]


G2PPath: TypeAlias = list[G2PGroup]


@dataclass
class G2PReading:
    """Complete legal phoneme realizations of one reading."""

    paths: list[G2PPath] = field(default_factory=list)


@dataclass
class G2PWord:
    """One converter-defined output word, with its alternative readings.

    Its text and boundaries may differ from the input tokenizer's words.
    """

    text: str
    language: str | None = None
    readings: list[G2PReading] = field(default_factory=list)


class Converter(ABC):
    language: tuple[str, ...] | None = None

    @abstractmethod
    def claim(self, token: str) -> bool:
        ...

    # noinspection PyMethodMayBeStatic
    def preprocessors(self) -> list[Preprocessor]:
        """Transform a claimed input run; word text and count may change."""
        return []

    @abstractmethod
    def convert(self, words: list[str]) -> list[G2PWord]:
        """Convert a claimed run into zero or more output words.

        The pipeline preserves output text and order and sets each word's
        language. Implementations may merge, split, rewrite, or omit inputs.
        """
        ...


class G2PConversionError(Exception):
    def __init__(self, unconverted_tokens: list[str]) -> None:
        self.unconverted_tokens = unconverted_tokens
        super().__init__(
            f"The following tokens could not be converted "
            f"by any converter in the chain: {unconverted_tokens}"
        )


def resolve_language(
    language: tuple[str, ...] | None,
    language_set: set[str] | None,
) -> str | None:
    """Return the single language tag that matched, or *None*."""
    if language is None:
        return None
    if language_set is None:
        return language[0]
    for tag in language:
        if tag in language_set:
            return tag
    return None
