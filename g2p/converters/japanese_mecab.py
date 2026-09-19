"""Japanese text -> UniDic pronunciations -> the existing kana G2P."""

from itertools import product
from pathlib import Path

from g2p.registry import converter
from g2p.tokenizers.cjk import CJKTokenizer
from .base import Converter, G2PConversionError, G2PReading, G2PWord
from .japanese import JapaneseKanaConverter, _is_kana


def _is_japanese_char(char: str) -> bool:
    cp = ord(char)
    return (
        _is_kana(char)
        or char in "々〆〇"
        or 0x3400 <= cp <= 0x4DBF
        or 0x4E00 <= cp <= 0x9FFF
        or 0xF900 <= cp <= 0xFAFF
        or 0x20000 <= cp <= 0x2FA1F
        or 0x30000 <= cp <= 0x323AF
    )


@converter(id="japanese-mecab", language="ja,jpn")
class JapaneseMecabConverter(Converter):
    """Segment full Japanese word forms and enumerate whole-word readings.

    MeCab supplies only kana pronunciations (UniDic's ``pron`` field).
    Romaji, phoneme groups, dictionary alternatives and long vowels are
    handled by :class:`JapaneseKanaConverter` without additional rules.
    The full PyPI ``unidic`` dictionary is loaded only on first conversion.
    """

    def __init__(
        self,
        dict_path: str,
        *,
        nbest: int = 32,
        double_written_sokuon: bool = False,
        unidic_dir: str | None = None,
    ) -> None:
        if isinstance(nbest, bool) or not isinstance(nbest, int) or nbest < 1:
            raise ValueError("nbest must be a positive integer.")
        self._nbest = nbest
        self._unidic_dir = unidic_dir
        self._tagger = None
        self._tokenizer = CJKTokenizer()
        self._kana = JapaneseKanaConverter(
            dict_path=dict_path, double_written_sokuon=double_written_sokuon,
        )

    def __getstate__(self):
        # Binarization and data loaders use spawned worker processes. MeCab's
        # native tagger cannot be pickled; each worker creates its own instance.
        state = self.__dict__.copy()
        state["_tagger"] = None
        return state

    def claim(self, token: str) -> bool:
        # In particular, never take ASCII romaji/phoneme input from the
        # Japanese dictionary converter used by existing training datasets.
        return bool(token) and all(_is_japanese_char(char) for char in token)

    def _get_tagger(self):
        if self._tagger is not None:
            return self._tagger
        try:
            import fugashi
            if self._unidic_dir is None:
                import unidic
                dicdir = Path(unidic.DICDIR)
            else:
                dicdir = Path(self._unidic_dir)
        except ImportError as exc:
            raise RuntimeError(
                "Japanese MeCab G2P requires fugashi and unidic. Run "
                "`python -m pip install fugashi unidic`, then "
                "`python -m unidic download` in the active environment."
            ) from exc
        if not (dicdir / "sys.dic").is_file() or not (dicdir / "mecabrc").is_file():
            raise RuntimeError(
                f"UniDic dictionary files are missing from {dicdir}. "
                "Run `python -m unidic download` in the active environment, "
                "or set unidic_dir to a full UniDic dictionary directory."
            )
        # Explicit paths avoid system MeCab dictionaries and support spaces
        # in Windows environment paths.
        dicdir = dicdir.resolve()
        self._tagger = fugashi.Tagger(
            f'-r "{(dicdir / "mecabrc").as_posix()}" -d "{dicdir.as_posix()}"'
        )
        return self._tagger

    def _pronunciations(self, word: str) -> list[str]:
        readings: list[str] = []
        seen: set[str] = set()
        for nodes in self._get_tagger().nbestToNodeList(word, self._nbest):
            if len(nodes) != 1 or nodes[0].surface != word:
                continue
            pron = getattr(nodes[0].feature, "pron", None)
            if not pron or pron in ("*", "-") or pron in seen:
                continue
            seen.add(pron)
            readings.append(pron)
        # Unknown kana still has a usable pronunciation. Unknown kanji and
        # numbers have no invented reading or romanization fallback.
        if not readings and all(_is_kana(char) for char in word):
            readings.append(word)
        return readings

    def _reading(self, kana: str) -> G2PReading:
        kana_words = self._kana.convert(self._tokenizer.tokenize([kana]))
        alternatives = [
            [path for reading in word.readings for path in reading.paths]
            for word in kana_words
        ]
        # Keep each candidate as a complete word path. Per-kana dictionary
        # alternatives combine within this reading, not across readings.
        paths = []
        seen = set()
        for parts in product(*alternatives):
            path = [group for part in parts for group in part]
            key = tuple((group.script, tuple(group.phonemes)) for group in path)
            if key not in seen:
                seen.add(key)
                paths.append(path)
        return G2PReading(paths=paths)

    def convert(self, words: list[str]) -> list[G2PWord]:
        text = "".join(words)
        if not text:
            return []
        # Snapshot the surfaces before N-best calls replace MeCab's lattice.
        # Joining also restores okurigana split by the upstream CJK tokenizer:
        # [駆, け, る] reaches MeCab as 駆ける.
        surfaces = [node.surface for node in self._get_tagger()(text)]
        result: list[G2PWord] = []
        for surface in surfaces:
            pronunciations = self._pronunciations(surface)
            if not pronunciations:
                raise G2PConversionError([surface])
            result.append(G2PWord(
                text=surface,
                readings=[self._reading(pron) for pron in pronunciations],
            ))
        return result
