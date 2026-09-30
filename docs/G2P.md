# G2P in TIFA

TIFA uses [g2pflow](https://github.com/openvpi/g2pflow) to convert text into
multilingual pronunciation candidates. The package provides the pipeline,
preprocessors, converters, configuration models and UniDic lifecycle.
TIFA supplies pronunciation dictionaries and model assets, then encodes the
results against its model vocabulary for alignment.

## Resources

`pip install -r requirements.txt` installs `g2pflow`. The default pipeline in
[configs/g2p.yaml](../configs/g2p.yaml) uses the four pronunciation dictionaries
included under `dictionaries/`:

| Language | Dictionary | Converter |
| --- | --- | --- |
| Mandarin | `ds-zh-pinyin-lite.txt` | `chinese-pinyin`, `dictionary` |
| Cantonese | `jyutping_dict.txt` | `yue-jyutping`, `dictionary` |
| Japanese | `japanese_dict_full.txt` | `japanese-mecab`, `dictionary` |
| English | `ds_cmudict-07b.txt` | `lstm` |

Install the optional backends needed by your pipeline:

```bash
pip install "g2pflow[ja]>=0.3.0,<0.4.0"    # Japanese MeCab and UniDic
pip install "g2pflow[lstm]>=0.3.0,<0.4.0"  # English ONNX inference
```

Use `pip install "g2pflow[all]>=0.3.0,<0.4.0"` to install both backends for the
default configuration. Language filtering does not prevent configured converters
from being constructed, so their dictionary and model metadata files must exist.

### Japanese

g2pflow initializes MeCab on the first Japanese conversion. If the default full
UniDic dictionary is missing, it downloads the dictionary under a process lock.
For offline use, prepare it in the same Python environment in advance:

```bash
python -m unidic download
```

To use a preinstalled dictionary, set `kwargs.unidic_dir` on `japanese-mecab` to
the directory containing `sys.dic` and `mecabrc`. An explicit directory is loaded
directly. The other settings in the default configuration, including `nbest` and
`double_written_sokuon`, are passed to g2pflow unchanged.

### English

Download the model package from
[LstmG2p v1.0.0](https://github.com/wolfgitpr/LstmG2p/releases/tag/v1.0.0) and
extract `encoder.onnx`, `decoder.onnx`, `char.json` and `phonemes.json` into
`assets/LstmG2p-Eng`, or set the converter's `model_path` to their directory.
These assets are not included in Git. ONNX inference runs for words absent from
the dictionary; use the `dictionary` converter for dictionary-only conversion.

Converter options, pronunciation dictionary formats and the full output
structure are documented in
[g2pflow's usage guide](https://github.com/openvpi/g2pflow/blob/main/docs/usage.md).

## Configuration

The existing YAML format is unchanged. Training and binarization use
`binarizer.g2p`; training saves this pipeline as `inference.g2p` in the model's
`config.yaml`. The G2P section is validated by `g2pflow.G2PPipelineConfig`.
TIFA still handles the surrounding training configuration's `bases` inheritance
and overrides.

Converters run in priority order. Use `-l` for the default inference language and
`-L` for additional tags, for example `-l zh -L en`. The default language prefix
is omitted from output labels. For Japanese text, `-l ja` prevents the Mandarin
converter from claiming kanji first in the default pipeline.

### Custom inference configuration

The file passed to `--g2p` contains the pipeline directly, without the training
configuration's `binarizer.g2p` wrapper or `bases` inheritance:

```yaml
preprocessors:
  - id: filter-punctuation
  - id: strip-whitespace
  - id: lowercase
converters:
  - id: dictionary
    language: en
    kwargs:
      dict_path: "path/to/dictionary.txt"
```

```bash
python infer.py [path-or-directory] -m [model-path] \
  --g2p [g2p-config-path] -l en
```

This overrides the model's embedded pipeline. Its emitted phonemes must match
the model vocabulary; supplying another dictionary does not extend that vocabulary.

### Resource paths

Ordinary relative resource paths are resolved from the working directory.
Strings beginning with `@` inside `kwargs`, including nested lists and mappings,
are resolved relative to the pipeline's `root_path`:

- Binarization uses the working directory.
- Inference's embedded pipeline uses the model directory.
- A custom `--g2p` file inside the model directory tree uses its own parent
  directory; a custom file outside that tree uses the working directory.
- Python callers may supply `root_path` explicitly.

For example, `dict_path: "@dictionaries/en.txt"` in an embedded pipeline refers
to `dictionaries/en.txt` under the model directory.

## PFML inference

Inference selects the first existing same-basename transcript in this order:
`.pfml`, `.txt`, `.lab`. Only `.pfml` is parsed as PFML;
the other two formats remain plain text. An empty or invalid selected file
skips the sample with a diagnostic instead of falling back to another file.
Preprocessing and binarization do not interpret PFML or scan for `.pfml` files.

The selected `-l`/`-L` tags are passed to g2pflow as its language filter.

For PFML syntax, semantics and examples, see the
[PFML 1.0 reference](https://github.com/openvpi/g2pflow/blob/v0.3.0/docs/pfml.md)
in the g2pflow documentation.

## Local G2P plugins

Place custom preprocessors and converters in [`plugins/g2p/`](../plugins/g2p/).
TIFA registers this directory with g2pflow when its G2P integration is imported,
including in spawned binarization and inference workers. The directory is resolved
relative to the TIFA checkout, independently of the working directory.

The scan imports direct `.py` files and subpackages with `__init__.py`. Define
components using g2pflow's `@preprocessor` and `@converter` decorators, then select
their IDs in the existing YAML configuration. Packages manage their own internal
imports; relative imports are supported. Duplicate IDs and import errors propagate.

For example, save this as `plugins/g2p/replace_underscores.py`:

```python
from g2pflow import Preprocessor, preprocessor


@preprocessor(id="replace-underscores")
class ReplaceUnderscores(Preprocessor):
    def process(self, tokens: list[str]) -> list[str]:
        return [token.replace("_", " ") for token in tokens]
```

Add `- id: replace-underscores` to the pipeline's `preprocessors` list to use it.
Modules execute once per process. Restart after modifying loaded plugin code;
`g2pflow.discover_plugins()` can discover newly added modules in a running process.

## Vocabulary encoding and alignment

g2pflow returns `G2PWord` objects containing readings, complete pronunciation
paths and groups with scripts and phonemes. TIFA's
[`lib/g2p_encoding.py`](../lib/g2p_encoding.py) converts those paths into numeric
grids and retains candidate order, word ownership, group boundaries and labels.
g2pflow removes silent branches from conversion results and reports incomplete
candidate trees as conversion failures.

`Language.ANY`, used by `passthrough`, defers language selection to vocabulary
encoding. TIFA tries the supplied languages in order, with existing bare and
explicitly prefixed symbols taking precedence. The token ID and matched symbol
are kept together even when several symbols share an ID. Binarization uses the
same resolver when collecting vocabulary symbols, applying its configured global
and stop symbols. Without a prebuilt vocabulary, unresolved symbols receive the
first input language prefix.

Unclaimed text or converter failures cause inference to skip the sample. During
vocabulary encoding, `--oov-handling discard` skips samples with unknown phonemes,
`raise` reports an error, and `force` drops affected pronunciation paths. Samples
without a valid token sequence are still skipped.

## Python usage

```python
from g2pflow import G2PPipelineConfig

from lib.g2p import build_pipeline_from_config
from lib.g2p_encoding import encode_paths
from lib.vocabulary import Vocabulary

config = G2PPipelineConfig.model_validate({
    "converters": [{"id": "passthrough"}],
})
pipeline = build_pipeline_from_config(config)
words = pipeline.convert("aa bb", languages=["en"])
vocabulary = Vocabulary(symbol_to_id={"en/aa": 3, "en/bb": 4})
data, lexicon, texts = encode_paths(words, vocabulary, languages=["en"])
```

Preprocessing uses these grids for text-only training data. Inference uses them
to score pronunciation candidates against audio and produce TextGrid labels.
