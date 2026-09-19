import importlib.util
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import yaml

from g2p.api import build_pipeline_from_config
from g2p.converters.base import G2PConversionError
from g2p.converters.japanese import JapaneseKanaConverter
from g2p.converters.japanese_mecab import JapaneseMecabConverter
from g2p.encoding import encode_paths
from g2p.pipeline import G2PPipeline
from g2p.tokenizers.cjk import CJKTokenizer
from lib.config.schema import G2PPipelineConfig
from lib.vocabulary import VocabularyBuilder


ROOT = Path(__file__).resolve().parents[1]
DICT = str(ROOT / "dictionaries/japanese_dict_full.txt")


def phones(word):
    return {
        tuple(phone for group in path for phone in group.phonemes)
        for reading in word.readings for path in reading.paths
    }


def node(surface, pron):
    return SimpleNamespace(surface=surface, feature=SimpleNamespace(pron=pron))


class JapaneseMecabUnitTests(unittest.TestCase):
    def test_claim_leaves_romaji_and_numbers_for_other_converters(self):
        converter = JapaneseMecabConverter(DICT)
        for text in ("", "ka", "cl", "g a cl k o", "10", "８", "A学校", "学校8"):
            with self.subTest(text=text):
                self.assertFalse(converter.claim(text))
        for text in ("学校", "駆ける", "スーパー", "時々", "きゃ"):
            with self.subTest(text=text):
                self.assertTrue(converter.claim(text))
        self.assertIsNone(converter._tagger)
        self.assertEqual(converter.convert([]), [])

    def test_nbest_filters_partial_paths_and_deduplicates_pron(self):
        converter = JapaneseMecabConverter(DICT)
        tagger = Mock(return_value=[node("学校", "ガッコー")])
        tagger.nbestToNodeList.return_value = [
            [node("学校", "ガッコー")],
            [node("学", "ガク"), node("校", "コー")],
            [node("学校", "ガッコー")],
            [node("学校", "ガクコー")],
            [node("校", "コー")],
            [node("学校", "-")],
            [node("学校", "*")],
            [node("学校", None)],
        ]
        converter._tagger = tagger
        words = converter.convert(["学", "校"])
        tagger.assert_called_once_with("学校")
        tagger.nbestToNodeList.assert_called_once_with("学校", 32)
        self.assertEqual([word.text for word in words], ["学校"])
        self.assertEqual(len(words[0].readings), 2)
        self.assertEqual(phones(words[0]), {
            ("g", "a", "cl", "k", "o"), ("g", "a", "k", "u", "k", "o"),
        })

    def test_unknown_kana_uses_existing_kana_logic(self):
        converter = JapaneseMecabConverter(DICT)
        converter._tagger = Mock(return_value=[node("キャー", "-")])
        converter._tagger.nbestToNodeList.return_value = [[node("キャー", "-")]]
        words = converter.convert(["キャ", "ー"])
        self.assertEqual(phones(words[0]), {("ky", "a")})

    def test_unknown_kanji_raises_with_its_actual_surface(self):
        converter = JapaneseMecabConverter(DICT)
        converter._tagger = Mock(return_value=[node("謎", "-")])
        converter._tagger.nbestToNodeList.return_value = [[node("謎", "-")]]
        with self.assertRaises(G2PConversionError) as caught:
            converter.convert(["謎"])
        self.assertEqual(caught.exception.unconverted_tokens, ["謎"])

    def test_dictionary_alternatives_stay_in_complete_word_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kana.txt"
            path.write_text("ka\tk a\nka\tk A\nki\tk i\nki\tk I\n", encoding="utf-8")
            converter = JapaneseMecabConverter(str(path))
            converter._tagger = Mock(return_value=[node("柿", "カキ")])
            converter._tagger.nbestToNodeList.return_value = [[node("柿", "カキ")]]
            words = converter.convert(["柿"])
        self.assertEqual(len(words[0].readings), 1)
        self.assertEqual(phones(words[0]), {
            ("k", "a", "k", "i"), ("k", "a", "k", "I"),
            ("k", "A", "k", "i"), ("k", "A", "k", "I"),
        })
        self.assertTrue(all(
            [group.script for group in path] == ["ka", "ki"]
            for path in words[0].readings[0].paths
        ))

    def test_invalid_nbest_is_rejected(self):
        for nbest in (0, -1, 1.5, True):
            with self.subTest(nbest=nbest), self.assertRaises(ValueError):
                JapaneseMecabConverter(DICT, nbest=nbest)

    @unittest.skipUnless(importlib.util.find_spec("fugashi"), "fugashi is not installed")
    def test_missing_dictionary_has_install_instructions(self):
        with tempfile.TemporaryDirectory() as directory:
            converter = JapaneseMecabConverter(DICT, unidic_dir=directory)
            with self.assertRaisesRegex(RuntimeError, "python -m unidic download"):
                converter.convert(["学校"])


class JapaneseMecabIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not importlib.util.find_spec("fugashi") or not importlib.util.find_spec("unidic"):
            raise unittest.SkipTest("Install fugashi/unidic and run python -m unidic download")
        import unidic
        if not (Path(unidic.DICDIR) / "sys.dic").is_file():
            raise unittest.SkipTest("Run python -m unidic download")
        cls.converter = JapaneseMecabConverter(DICT)
        cls.pipeline = G2PPipeline(
            tokenizers=[CJKTokenizer()], converters=[cls.converter],
        )

    def test_school_uses_pron_and_existing_long_vowel_behavior(self):
        words = self.pipeline.convert("学校", languages=["ja"])
        self.assertEqual([word.text for word in words], ["学校"])
        self.assertEqual(phones(words[0]), {("g", "a", "cl", "k", "o")})
        self.assertEqual(words[0].language, "ja")
        kana = JapaneseKanaConverter(DICT)
        self.assertEqual(kana.text_to_scripts(["ー", "゜"]), [[""], [""]])

    def test_okurigana_reaches_mecab_with_its_kanji(self):
        words = self.pipeline.convert("駆ける", languages=["jpn"])
        self.assertEqual([word.text for word in words], ["駆ける"])
        self.assertEqual(words[0].language, "jpn")
        self.assertIn(("k", "a", "k", "e", "r", "u"), phones(words[0]))

    def test_wind_keeps_readings_in_both_contexts(self):
        candidates = []
        for text in ("風に会える", "夏色の風"):
            words = self.pipeline.convert(text, languages=["ja"])
            wind = next(word for word in words if word.text == "風")
            candidates.append(phones(wind))
            self.assertIn(("f", "u"), candidates[-1])
            self.assertIn(("k", "a", "z", "e"), candidates[-1])
            self.assertEqual(len(wind.readings), len(candidates[-1]))
        self.assertEqual(candidates[0], candidates[1])

    def test_sokuon_option_uses_existing_gemination(self):
        converter = JapaneseMecabConverter(DICT, double_written_sokuon=True)
        self.assertEqual(phones(converter.convert(["学校"])[0]), {("g", "a", "k", "k", "o")})

    def test_known_numeric_blind_spots_are_not_invented(self):
        for text in ("８", "10"):
            with self.subTest(text=text):
                self.assertEqual(self.converter._pronunciations(text), [])
                with self.assertRaises(G2PConversionError):
                    self.pipeline.convert(text, languages=["ja"])

    def test_reference_config_routes_japanese_and_preserves_romaji(self):
        config = yaml.safe_load((ROOT / "configs/g2p.yaml").read_text(encoding="utf-8"))["binarizer"]["g2p"]
        # The English ONNX assets are independent of Japanese G2P and may
        # be installed separately; exercise the other reference converters.
        config["converters"] = [c for c in config["converters"] if c["id"] != "lstm"]
        for converter in config["converters"]:
            converter["kwargs"]["dict_path"] = str(ROOT / converter["kwargs"]["dict_path"])
        pipeline = build_pipeline_from_config(G2PPipelineConfig.model_validate(config))
        for language in ("ja", "jpn"):
            words = pipeline.convert("学校 ka cl", languages=[language])
            self.assertEqual([word.text for word in words], ["学校", "ka", "cl"])
            self.assertEqual([word.language for word in words], [language] * 3)
            self.assertEqual(phones(words[0]), {("g", "a", "cl", "k", "o")})
            self.assertEqual(phones(words[1]), {("k", "a")})
            self.assertEqual(phones(words[2]), {("cl",)})
        converter = next(c for c in pipeline._converters if isinstance(c, JapaneseMecabConverter))
        for token in converter._kana._script_dict:
            if token.isascii():
                self.assertFalse(converter.claim(token))
        words = pipeline.convert("你好", languages=["zh"])
        self.assertEqual([word.text for word in words], ["你", "好"])
        self.assertTrue(all(word.language == "zh" for word in words))

    def test_word_candidates_survive_downstream_encoding(self):
        words = self.pipeline.convert("学校の風", languages=["ja"])
        builder = VocabularyBuilder()
        for word in words:
            for reading in word.readings:
                for path in reading.paths:
                    builder.add([phone for group in path for phone in group.phonemes], word.language)
        data, lexicon, texts = encode_paths(words, builder.build())
        self.assertEqual(texts, ["学校", "の", "風"])
        self.assertEqual(data["candidates"].sum(axis=1).tolist(), [len(entry) for entry in lexicon])
        self.assertEqual(len(lexicon[-1]), len(words[-1].readings))
        self.assertIn(["ja/k", "ja/a", "ja/z", "ja/e"], [entry["phonemes"] for entry in lexicon[-1]])

    def test_initialized_converter_can_be_pickled_for_windows_workers(self):
        original = self.converter.convert(["学校"])
        restored = pickle.loads(pickle.dumps(self.converter))
        self.assertIsNone(restored._tagger)
        self.assertEqual(restored.convert(["学校"]), original)


if __name__ == "__main__":
    unittest.main()
