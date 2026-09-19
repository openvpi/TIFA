from dataclasses import dataclass

from .converters.base import (
    Converter,
    G2PConversionError,
    G2PWord,
    resolve_language,
)
from .preprocessors.base import Preprocessor
from .tokenizers.base import Tokenizer


@dataclass
class _TokenState:
    """An unconverted token or the complete output of a converted run."""

    text: str
    results: list[G2PWord] | None = None


class G2PPipeline:
    def __init__(
        self,
        preprocessors: list[Preprocessor] | None = None,
        tokenizers: list[Tokenizer] | None = None,
        converters: list[Converter] | None = None,
    ) -> None:
        self._preprocessors = preprocessors or []
        self._tokenizers = tokenizers or []
        self._converters = converters or []

    def convert(
        self, text: str, *, languages: list[str] | None = None,
    ) -> list[G2PWord]:
        language_set = set(languages) if languages else None
        active = [
            c for c in self._converters
            if language_set is None
            or c.language is None
            or any(ln in language_set for ln in c.language)
        ]
        if not active:
            raise ValueError("No converter matches the requested languages.")

        tokens = [text]
        for pp in self._preprocessors:
            tokens = pp.process(tokens)
        for tok in self._tokenizers:
            tokens = tok.tokenize(tokens)

        states = [_TokenState(text=t) for t in tokens]
        for converter in active:
            next_states: list[_TokenState] = []
            i = 0
            while i < len(states):
                if states[i].results is not None or not converter.claim(states[i].text):
                    next_states.append(states[i])
                    i += 1
                    continue
                j = i + 1
                while (
                    j < len(states)
                    and states[j].results is None
                    and converter.claim(states[j].text)
                ):
                    j += 1
                run_states = states[i:j]
                run_texts = [s.text for s in run_states]
                for pp in converter.preprocessors():
                    run_texts = pp.process(run_texts)
                results = converter.convert(run_texts)
                resolved = resolve_language(converter.language, language_set)
                for result in results:
                    result.language = resolved
                # Keep even an empty output block so later converters cannot
                # join input tokens across a run that was already handled.
                next_states.append(_TokenState(
                    text="".join(s.text for s in run_states), results=results,
                ))
                i = j
            states = next_states

        unconverted = [s for s in states if s.results is None]
        if unconverted:
            raise G2PConversionError([s.text for s in unconverted])

        return [word for s in states if s.results is not None for word in s.results]
