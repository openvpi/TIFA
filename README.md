# TIFA: Token-Imputing Forced Aligner

## Overview

An accurate forced aligner to align speech or singing recordings with their transcripts.

### Highlights

1. Pronunciation scoring: choose the pronunciation candidates that best match the audio.
2. Semantic while phonetic: align both the written forms and phonetic symbols.
3. Robust model: work on label errors, noise, reverberation or even accompaniments.
4. Multilingual: able to align multiple languages in a single transcript through flexible configuration.
5. Diagnostic metrics: identify low-quality alignments without reference annotations.

### Downstream applications

1. Grab phoneme-level labels for speech and singing voice synthesis tasks.
2. Inspect and correct misaligned or noisy annotations in existing datasets.
3. Distill pronunciation knowledge to improve G2P and ASR models.

## Installation

### Environment setup

Step 1: Start with a separate Python environment, such as a Conda environment.

Step 2: Install the latest version of PyTorch following its [official website](https://pytorch.org/get-started/locally/).

Step 3: Install the project dependencies:

```bash
pip install -r requirements.txt
```

Run the commands below from the repository root. Dataset, dictionary and asset paths in the example configurations are relative to this directory.

### G2P resources

The default pipeline in [configs/g2p.yaml](configs/g2p.yaml) includes Chinese (Mandarin & Yue), Japanese and English converters. Prepare the dictionaries, optional dependencies and model assets for your configured converters as described in [G2P.md](docs/G2P.md#resources).

### Pretrained models

Full list: [releases](https://github.com/openvpi/TIFA/releases)

| Version     | Description                                 | Link                                                           |
|-------------|---------------------------------------------|----------------------------------------------------------------|
| TIFA-1.0-ST | 1.0 model with 4 languages and special tags | [PyTorch](https://github.com/openvpi/TIFA/releases/tag/v1.0.0) |

## Inference

### Prepare audio and text

Each audio file must have a UTF-8 transcript with the same basename. The script looks for `.txt` first, then `.lab`:

```text
path/to/audio/
├── sample1.wav
├── sample1.txt
├── sample2.flac
└── sample2.lab
```

Put the spoken text or lyrics in each transcript. Samples with missing or empty transcripts are skipped.

The model directory must contain its matching `config.yaml` and `vocabulary.json`:

```text
path/to/model/
├── model.pt
├── config.yaml
└── vocabulary.json
```

Keep any dictionary and G2P assets referenced by the model configuration available as well.

### Align a single file

```bash
python infer.py [audio-path] -m [model-path] -l [language] --stat
```

We recommend enabling `--stat`, as shown above, to save self-check metrics for inspecting alignment quality. See [Statistics](#statistics) for the generated files and metric descriptions.

By default, the TextGrid is saved beside the audio file with the same basename, such as `sample1.TextGrid`. It contains `texts`, `words` and `phones` tiers, described below.

### Process a directory

```bash
python infer.py path/to/audio/ -m path/to/model.pt -l zh -o path/to/output/ --batch-size 8 --stat
```

Directory processing is recursive and preserves the relative subdirectory structure in the output. The default audio extensions are `wav,flac,opus,mp3,aac,ogg`; decoding support depends on the installed audio backend. Use `--input-formats wav,flac` to restrict the scan.

### Language and G2P configuration

Use `-l` to select the default language, for example `zh`, `ja` or `en`. Use `-L` to enable additional languages as a comma-separated list. For example, to align Chinese text containing English:

```bash
python infer.py [path-or-directory] -m [model-path] -l zh -L en --stat
```

The G2P configuration must include converters for the selected languages. The default language prefix is omitted from phoneme labels.

Inference uses `inference.g2p` from the model's `config.yaml`. To override it, or if the model has no G2P configuration:

```bash
python infer.py [path-or-directory] -m [model-path] --g2p [g2p-config-path] --stat
```

For example, a custom configuration for dictionary-based English conversion:

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

Save this pipeline directly as a YAML file, without a `binarizer.g2p` wrapper. Its phonemes must match the model vocabulary. See [G2P.md](docs/G2P.md) for dictionary formats, multilingual configuration, advanced usage and developer documentation.

For all inference options:

```bash
python infer.py --help
```

### Output

Each TextGrid contains three tiers in this order:

| Tier     | Labels                                                                                 | Boundaries                                         |
|----------|----------------------------------------------------------------------------------------|----------------------------------------------------|
| `texts`  | Word text returned by G2P, after preprocessing and conversion.                         | Span all phonemes belonging to each semantic word. |
| `words`  | Pronunciation scripts, such as pinyin or romaji, from the selected pronunciation path. | Span the phonemes in each pronunciation group.     |
| `phones` | Individual phoneme symbols from the selected pronunciation path.                       | Predicted phoneme boundaries.                      |

A semantic word may contain several pronunciation groups, so `texts` and `words` can have different interval counts. Their segmentation follows the G2P converter.

For example, Japanese input `猫` can be converted into the pronunciation groups `ne` and `ko`, with phonemes `n e k o`. With `-l ja`, an illustrative alignment would look like this:

```text
texts      |      |                猫                 |      |
words      |      |        ne        |       ko       |      |
phones     |      | n |      e       |  k  |    o     |      |
```

All three tiers share the same timeline. The word `猫` spans both `ne` and `ko`, and each pronunciation group spans its two phonemes. Unlabeled gaps, shown as blank cells above, are filled with `""`.

The `-l ja` option removes `ja/` from phoneme labels in this example. Other language prefixes remain in mixed-language output. Zero-width phoneme intervals are omitted by default; use `--skip-handling` to change this behavior.

### Statistics

With `--stat`, inference saves `scores.json` and `diagnosis.json` together under `statistics/` in the output directory, along with metric histograms and scatter plots against sample length. No statistics are written if there are no diagnostic records.

#### Scoring

When G2P provides alternative pronunciations, inference scores them before alignment and jointly selects a complete pronunciation for each word. This runs by default. The `scores.json` report includes only words with multiple candidates and samples containing such words.

For example, the following report shows two pronunciations of English `read`, with illustrative scores:

```json
[
  {
    "identifier": "sample1",
    "words": [
      {
        "index": 0,
        "text": "read",
        "chosen": 1,
        "alternatives": [
          {"index": 0, "scripts": ["read"], "phones": ["en/r", "en/iy", "en/d"], "score": 1.42},
          {"index": 1, "scripts": ["read"], "phones": ["en/r", "en/eh", "en/d"], "score": 2.08}
        ]
      }
    ]
  }
]
```

`identifier` locates the sample, and each word's `index` is its position in the G2P output. `chosen` identifies the selected entry in `alternatives`; both word and candidate indices start at 0. Each candidate contains pronunciation-group `scripts`, its `phones`, and a `score`. Phone labels retain their language prefixes in this report.

Higher scores favor a candidate, but the values are not probabilities. Each score describes the best whole-sample pronunciation combination containing that candidate. Compare alternatives for the same word and use `chosen` for the actual joint selection. In this example, candidate 1 supplies the phonemes used for alignment and TextGrid output.

If there are no words with multiple candidates, or scoring is disabled with `--score-unit none`, the report is an empty list. Disabling scoring selects the first available candidate for each word.

#### Diagnosis

The `diagnosis.json` report contains per-sample self-check metrics. These metrics require no reference annotations and help identify alignments that need inspection.

| Metric         | Range   | What it measures                                                                                                                  | How to interpret a low value                                                                                          |
|----------------|---------|-----------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| `agreement`    | 0 to 1  | The mean probability that the model's phoneme classifier assigns to each phoneme in the selected sequence.                        | The model has weak support for the supplied text or selected pronunciation. Check the transcript and G2P result.      |
| `confidence`   | -1 to 1 | The mean audio-phoneme cosine similarity inside each predicted span, averaged over phonemes. Zero-width spans contribute 0.       | The assigned audio regions match their phonemes weakly. Listen to the audio and inspect the boundaries.               |
| `determinacy`  | 0 to 1  | How much positive similarity evidence is concentrated on the decoded alignment, compared with competing nearby phoneme positions. | Several positions compete for the same audio, making the alignment ambiguous. Repeated sounds are one possible cause. |
| `monotonicity` | 0 to 1  | How much positive similarity evidence points to the currently aligned phoneme or later positions, averaged over aligned frames.   | The evidence is drawn toward earlier phonemes, suggesting that the alignment may have advanced too far.               |

Higher values indicate stronger self-consistency for all four metrics. They are not calibrated probabilities that an alignment is correct.

For inference, `diagnosis.json` is sorted to put suspicious samples first. Samples are ranked independently from lowest to highest score for each metric, with rank 0 being the worst. Each sample's smallest rank is the primary sort key, in ascending order. Ties are broken by its ranks for `agreement`, `confidence`, `determinacy` and `monotonicity`, in that order. A sample can therefore appear near the beginning because it performs poorly on just one metric, even if its other scores are strong.

This report can be used to inspect or filter low-quality data and incorrect annotations, including poor recordings, transcript-audio mismatches, pronunciation errors and unreliable alignments. Start with the first records and use each sample's `identifier` to locate its audio and TextGrid. Review them alongside `--plot` similarity plots to decide which annotations to correct or samples to exclude. For automatic filtering, choose metric thresholds from a manually reviewed subset using the same model.

Add `--plot` to save per-sample similarity plots next to the TextGrid files for closer inspection.

## Training

### Data preparation

Training supports heterogeneous datasets: a main dataset with phoneme duration labels, and an auxiliary dataset with transcripts or phoneme sequences without timing annotations. **The main dataset is required**, including when auxiliary data is used: it supplies supervised training targets and defines the vocabulary and validation split. Adding an auxiliary dataset is optional and can improve alignment quality by expanding data coverage through semi-supervised learning.

Keep the two annotation types in separate data roots. Both use the following layout, with an `index.csv` and a `waveforms` directory in each subset:

```text
path/to/datasets/
├── dataset1/
│   ├── index.csv
│   └── waveforms/
│       ├── item1.wav
│       └── item2.wav
└── dataset2/
    ├── index.csv
    └── waveforms/
        └── item3.flac
```

#### Main dataset: phoneme timing

Each main dataset CSV must contain these columns:

| Column      | Description                                                                           |
|-------------|---------------------------------------------------------------------------------------|
| `name`      | Audio filename without its extension.                                                 |
| `language`  | Language tag, such as `zh`, `ja` or `en`.                                             |
| `phones`    | Space-separated phoneme symbols, including silence or breath labels where applicable. |
| `durations` | Space-separated durations in seconds, one per phoneme.                                |

For example:

```csv
name,language,phones,durations
item1,zh,SP n i h ao SP,0.20 0.08 0.22 0.08 0.32 0.10
```

The phoneme and duration sequences must have equal lengths. Durations should cover the audio, including silence. Both dataset types accept WAV, FLAC and Opus audio.

#### Auxiliary dataset: audio without timing labels

Auxiliary data still requires audio, but does not require `durations`. Its `index.csv` contains `name`, `language`, and either `text` or `phones`. For transcripts:

```csv
name,language,text
item1,zh,你好
item2,en,hello world
item3,zh+en,你好 hello world
item4,ja+en,君と hello world
```

The pipeline in `binarizer.g2p` converts the text to pronunciation candidates. Prepare its dictionaries and dependencies as described in [G2P.md](docs/G2P.md). If phoneme sequences are already available, provide them directly:

```csv
name,language,phones
item1,zh,n i h ao
```

A nonempty `phones` field takes precedence over `text` and bypasses G2P for that item. Auxiliary data uses the main dataset's vocabulary. Preprocessing filters pronunciation candidates containing unsupported phonemes and skips samples that cannot be fully encoded; auxiliary-only phonemes do not expand the vocabulary.

For multilingual fixed phones, use language prefixes such as `zh/n zh/i en/m en/iy`. The first language tag supplies the default for unqualified phones and converters without a language label.

### Configuration

Configurations use YAML inheritance through the `bases` key. [configs/supervised.yaml](configs/supervised.yaml) inherits training and feature defaults from [configs/base.yaml](configs/base.yaml), and G2P and vocabulary settings from [configs/g2p.yaml](configs/g2p.yaml).

For heterogeneous training, set both dataset roots, enable semi-supervised learning and its EMA teacher, and configure optional augmentation resources:

```yaml
binarizer:
  phoneme_timing_data_dir: "path/to/phoneme-timing-data"
  text_only_data_dir: "path/to/text-only-data"

training:
  semisupervised:
    enabled: true
    aux_loss_weight: 0.25
  weight_averaging:
    ema_enabled: true
  dataloader:
    aux_ratio: 1.0
    aux_warmup_epochs: 10
  augmentation:
    natural_noise:
      enabled: false
      noise_path_glob: "path/to/noise/**/*.wav"
    rir_reverb:
      enabled: false
      kernel_path_glob: "path/to/reverb/**/*.wav"
```

The base configuration enables both natural noise and RIR augmentation. Supply the corresponding files or disable these augmentations. Noise recordings should not contain clear speech or singing; RIR files should contain room impulse responses.

Review the validation split, batch frame limits, precision and training duration for your dataset and hardware. The default validation split reserves 1,000 main-dataset items, so adjust `binarizer.split.count` for smaller datasets.

Use repeated `--override key.path=value` options to change settings from the command line. Apply preprocessing-related overrides consistently when binarizing and training.

### Preprocessing

```bash
python binarize.py --config [config-path]
```

When both data roots are configured, this single command processes both datasets. The main dataset is split into training and validation sets; retained auxiliary samples all go into training under the `aux` prefix, with no separate validation split. Each root receives its own binary data, metadata, `feature.yaml` and the shared `vocabulary.json`.

Keep the source audio available: waveform loading and augmentation take place during training. When moving training to another machine, copy both complete datasets and any augmentation resources.

Re-binarize after changing feature or vocabulary settings. Training checks that the saved feature settings match the active configuration.

### Start or resume training

```bash
python train.py --config [config-path] --exp-name [experiment-name]
```

Checkpoints and logs are saved under `experiments/[experiment-name]/` by default. Training automatically resumes when it finds an unambiguous latest checkpoint. To select one explicitly:

```bash
python train.py --config [config-path] --exp-name [experiment-name] \
  --resume-from [checkpoint-path]
```

For example, to override the training duration:

```bash
python train.py --config [config-path] --exp-name [experiment-name] \
  --override training.trainer.max_steps=10000
```

View training metrics and validation plots with TensorBoard:

```bash
tensorboard --logdir [experiment-dir]
```

For all startup options, run `python train.py --help`.

### Post-training

After the initial training stage, use [configs/post.yaml](configs/post.yaml) to fine-tune a selected checkpoint on longer sequences. The recipe concatenates up to 8 samples per item with a 12,000-frame budget, lowers the optimizer learning rate to `0.00005`, and uses smaller batches with gradient accumulation. Validation runs every 1,000 steps.

The configuration inherits [configs/supervised.yaml](configs/supervised.yaml). If you customized the first-stage configuration, carry over the same model, dataset, feature, vocabulary and semi-supervised learning settings. Concatenation happens during data loading, so the existing binarized dataset can be reused without re-binarization when its feature and vocabulary settings are unchanged.

Replace `BEST_PRETRAINING_CHECKPOINT` in `training.finetuning.pretraining_from` with the selected first-stage checkpoint, or override it when starting the new stage:

```bash
python train.py --config configs/post.yaml --exp-name [post-experiment-name] \
  --override training.finetuning.pretraining_from=[checkpoint-path]
```

The post-training recipe sets `aux_warmup_epochs: 0`, so enabled auxiliary training contributes immediately. The teacher labels and filters individual auxiliary samples before accepted samples are concatenated for the student.

Use a new experiment name. Post-training loads the pretrained weights and starts fresh optimizer and scheduler states, with the step count reset. Set `training.trainer.max_steps` to the desired duration of this stage. To continue an interrupted post-training run, use the same configuration and `--resume-from` with a checkpoint saved during this stage.

### Reduce a checkpoint

Reduce a training checkpoint to an inference model by dropping optimizer states and keeping only inference weights:

```bash
python reduce.py [input-ckpt-path] [output-model-path]
```

Copy the matching `config.yaml` and `vocabulary.json` beside the reduced model, and retain its required G2P resources. Reduced models are intended for inference rather than resuming training.

## Evaluation

### Evaluate a model

Evaluate the validation split of a binarized dataset:

```bash
python evaluate.py -d [dataset-dir] -m [model-path] -o [save-dir] --plot
```

The dataset must use the same feature settings and vocabulary as the model. Evaluation uses the `valid` prefix by default; select another split with `--prefix`.

For a separate test dataset, use the training feature configuration and reuse the model's vocabulary when binarizing:

```bash
python binarize.py --config path/to/config.yaml --eval \
  --override binarizer.phoneme_timing_data_dir=path/to/test-datasets \
  --override binarizer.text_only_data_dir=null \
  --override binarizer.vocabulary.prebuilt_vocab_file=path/to/model/vocabulary.json

python evaluate.py -d path/to/test-datasets -m path/to/model.pt \
  -o path/to/evaluation --plot
```

The `--eval` preprocessing mode puts the entire dataset into the validation split. Use a separate test-data root to keep the training split intact.

Results include `summary.json` and a `statistics/` directory. Add `--plot` to save per-sample similarity and alignment plots under `plots/`.

Online boundary errors and tolerances are measured in frames. Convert to time using `hop_size / audio_sample_rate`; the default feature configuration uses 10 ms per frame. BER and overlap values are ratios. Lower BER, boundary MAE and conjunction MAE indicate smaller alignment errors; higher overlap indicates better agreement.

### Compare TextGrid annotations

Compare predicted TextGrids with reference TextGrids without loading a model:

```bash
python evaluate.py offline --pred [pred-dir] --gt [gt-dir] \
  -o [save-dir]
```

Files are paired by relative path. The default tier is `phones`; use `--tier-name` to select another tier. Phoneme labels must match after stop-symbol filtering. Label mismatches raise an error by default; `--mismatch-handling skip` skips those pairs.

Offline boundary errors and tolerances are measured in milliseconds. Results are saved to `summary.json` and `statistics/` in the output directory.

For more options:

```bash
python evaluate.py --help
python evaluate.py offline --help
```

## Deployment

Models can be exported to ONNX format for further deployment.

### Export ONNX models

Run the following command to export a supervised model:

```bash
python deploy.py -m [model-path] -o [save-dir]
```

Keep the matching `config.yaml` and `vocabulary.json` beside the model. By default, ONNX models are exported using the _trace_ exporter and opset version 18. Use `--opset-version` to select an opset from 18 through 20.

### Inference with ONNX models

We don't provide a standalone ONNX inference pipeline in this repository. See the [documentation](ONNX.md) for the workflow, tensor interfaces and steps to implement in the host application.

### Package for the dataset-tools application

[dataset-tools](https://github.com/openvpi/dataset-tools) contains the **Tifa** application, a forced aligner with the same pipeline as this repository and a graphical interface. Build a package it can load with:

```bash
python deploy_dataset_tools.py -m [model-path] -o [save-dir]
```

The command exports one model as before and copies the pronunciation dictionaries, and the English out-of-vocabulary model when `assets/LstmG2p-Eng` is available, into the package. See the [documentation](ONNX.md#tifa-application-of-dataset-tools) for the layout and the installation steps.

## Integration

The repository exposes APIs for downstream applications:

- Preprocessing: [preprocessing/api.py](preprocessing/api.py)
- Training: [training/api.py](training/api.py)
- Inference and evaluation: [inference/api.py](inference/api.py)
- Deployment: [deployment/api.py](deployment/api.py)

## License

TIFA is licensed under the [MIT License](LICENSE).
