import unittest
from collections.abc import Callable

from g2p.converters.base import Converter, G2PConversionError, G2PWord
from g2p.converters.simple import CharPhonemeConverter, PassthroughConverter
from g2p.pipeline import G2PPipeline
from g2p.preprocessors.base import Preprocessor
from g2p.preprocessors.simple import FilterPunctuation, LowercasePreprocessor
from g2p.tokenizers.simple import WhitespaceTokenizer


class RecordingConverter(Converter):
    def __init__(
        self,
        convert: Callable[[list[str]], list[G2PWord]],
        *,
        claim: Callable[[str], bool] = lambda token: True,
        preprocessors: list[Preprocessor] | None = None,
        language: tuple[str, ...] | None = None,
    ):
        self._convert = convert
        self._claim = claim
        self._preprocessors = preprocessors or []
        self.language = language
        self.calls: list[list[str]] = []

    def claim(self, token: str) -> bool:
        return self._claim(token)

    def preprocessors(self) -> list[Preprocessor]:
        return self._preprocessors

    def convert(self, words: list[str]) -> list[G2PWord]:
        self.calls.append(list(words))
        return self._convert(words)


class MergePreprocessor(Preprocessor):
    def process(self, tokens: list[str]) -> list[str]:
        return ["".join(tokens)]


class PipelineTests(unittest.TestCase):
    def pipeline(self, *converters: Converter) -> G2PPipeline:
        return G2PPipeline(
            tokenizers=[WhitespaceTokenizer()], converters=list(converters),
        )

    def test_converter_can_merge_tokens(self):
        converter = RecordingConverter(lambda words: [G2PWord("".join(words))])
        words = self.pipeline(converter).convert("a b c")
        self.assertEqual([word.text for word in words], ["abc"])
        self.assertEqual(converter.calls, [["a", "b", "c"]])

    def test_converter_can_split_tokens(self):
        converter = RecordingConverter(
            lambda words: [G2PWord(char) for word in words for char in word],
        )
        words = self.pipeline(converter).convert("ab cd")
        self.assertEqual([word.text for word in words], ["a", "b", "c", "d"])

    def test_converter_can_rewrite_text(self):
        converter = RecordingConverter(lambda words: [G2PWord("normalized")])
        words = self.pipeline(converter).convert("original")
        self.assertEqual(words[0].text, "normalized")

    def test_dropped_run_is_handled(self):
        drop = RecordingConverter(lambda words: [])
        fallback = RecordingConverter(lambda words: [G2PWord("unexpected")])
        self.assertEqual(self.pipeline(drop, fallback).convert("a b"), [])
        self.assertEqual(fallback.calls, [])

    def test_preprocessors_can_split_drop_and_rewrite(self):
        converter = RecordingConverter(
            lambda words: [G2PWord(word) for word in words],
            preprocessors=[FilterPunctuation(), LowercasePreprocessor()],
        )
        words = self.pipeline(converter).convert("ONE,TWO ! THREE")
        self.assertEqual(converter.calls, [["one", "two", "three"]])
        self.assertEqual([word.text for word in words], ["one", "two", "three"])

    def test_preprocessor_can_merge_tokens(self):
        converter = RecordingConverter(
            lambda words: [G2PWord(word) for word in words],
            preprocessors=[MergePreprocessor()],
        )
        words = self.pipeline(converter).convert("a b")
        self.assertEqual(converter.calls, [["ab"]])
        self.assertEqual([word.text for word in words], ["ab"])

    def test_preprocessor_can_remove_entire_run(self):
        converter = RecordingConverter(
            lambda words: [G2PWord(word) for word in words],
            preprocessors=[FilterPunctuation()],
        )
        self.assertEqual(self.pipeline(converter).convert("! ?"), [])
        self.assertEqual(converter.calls, [[]])

    def test_language_is_resolved_for_every_output(self):
        converter = RecordingConverter(
            lambda words: [G2PWord(char, language="old") for char in words[0]],
            language=("ja", "jp"),
        )
        words = self.pipeline(converter).convert("ab", languages=["jp"])
        self.assertEqual([word.language for word in words], ["jp", "jp"])
        words = self.pipeline(converter).convert("ab")
        self.assertEqual([word.language for word in words], ["ja", "ja"])

    def test_language_filter_and_neutral_converter(self):
        skipped = RecordingConverter(
            lambda words: [G2PWord("unexpected")], language=("ja",),
        )
        neutral = RecordingConverter(lambda words: [G2PWord("ok", language="old")])
        words = self.pipeline(skipped, neutral).convert("a", languages=["en"])
        self.assertEqual(skipped.calls, [])
        self.assertEqual([(word.text, word.language) for word in words], [("ok", None)])
        with self.assertRaisesRegex(ValueError, "No converter matches"):
            self.pipeline(skipped).convert("a", languages=["en"])

    def test_converted_run_preserves_order_and_separates_later_runs(self):
        for middle_output in ([], ["middle"], ["first", "second"]):
            with self.subTest(middle_output=middle_output):
                middle = RecordingConverter(
                    lambda words: [G2PWord(text) for text in middle_output],
                    claim=lambda token: token == "|",
                )
                rest = RecordingConverter(
                    lambda words: [G2PWord("".join(words))],
                )
                words = self.pipeline(middle, rest).convert("a b | c d")
                self.assertEqual(rest.calls, [["a", "b"], ["c", "d"]])
                self.assertEqual(
                    [word.text for word in words], ["ab", *middle_output, "cd"],
                )

    def test_unclaimed_tokens_separate_runs(self):
        merge = RecordingConverter(
            lambda words: [G2PWord("".join(words))],
            claim=lambda token: token != "|",
        )
        words = self.pipeline(merge, PassthroughConverter()).convert("a b | c d")
        self.assertEqual(merge.calls, [["a", "b"], ["c", "d"]])
        self.assertEqual([word.text for word in words], ["ab", "|", "cd"])

    def test_unresolved_error_contains_only_original_unclaimed_tokens(self):
        converter = RecordingConverter(
            lambda words: [G2PWord("merged")],
            claim=lambda token: token in {"a", "b"},
        )
        with self.assertRaises(G2PConversionError) as caught:
            self.pipeline(converter).convert("left a b right")
        self.assertEqual(caught.exception.unconverted_tokens, ["left", "right"])

    def test_existing_character_conversion(self):
        converter = CharPhonemeConverter({"a": ["AH"], "b": ["B"]})
        words = self.pipeline(converter, PassthroughConverter()).convert("ab unknown a")
        self.assertEqual([word.text for word in words], ["ab", "unknown", "a"])
        self.assertEqual(words[0].readings[0].paths[0][0].phonemes, ["AH", "B"])
        self.assertEqual(words[1].readings[0].paths[0][0].phonemes, ["unknown"])


if __name__ == "__main__":
    unittest.main()
