# Small Language Models for Alzheimer's Speech Screening

[![tests](https://github.com/Nabdalnabi/small-language-models-ad-screening/actions/workflows/tests.yml/badge.svg)](https://github.com/Nabdalnabi/small-language-models-ad-screening/actions/workflows/tests.yml)

Research code samples on local classification of speech transcripts using small language models. This public code-only repository demonstrates full fine-tuning, validation-based checkpoint selection,
and separate evaluation with DeepSeek, GPT-2, or SmolLM2.

## Research

**Evaluating Open-Source Small Language Models for On-Premise Alzheimer's Disease
Speech Screening**

Venkatanand Ram Addepalli, Nader Abdalnabi, Erich Kummerfeld, Guy Hembroff,
Andrew Kiselica, Praveen Rao, and Knoo Lee.

[Paper link](https://openreview.net/pdf?id=W4i1HQSd2K)

Related preprint: **Small-Sized Reasoning Language Models for Linguistic Screening
of Alzheimer's Disease**,
[DOI: 10.64898/2025.12.24.25342972](https://doi.org/10.64898/2025.12.24.25342972).

The study reports 91.67% accuracy
and 91.71% F1 for full fine-tuning of DeepSeek-R1 (1.5B), compared with 70.0%
accuracy for its Phi-3 baseline. These are manuscript results, not measurements
from this refactored code. The research uses restricted-access Cookie Theft picture-description
transcripts from the DementiaBank Pitt Corpus. These are not freely redistributable
open data. Obtain access through [DementiaBank](https://talkbank.org/dementia/access.html)
and follow the [TalkBank ground rules](https://talkbank.org/0share/rules.html).
No transcripts, audio, private CSVs, or paper figures are redistributed here.

## Contents

- `slm_classifier.py`: training and evaluation command-line entry point.
- `data_utils.py`: CSV validation, shared labels, duplicate and participant checks.
- `test_data_utils.py`: synthetic tests requiring only Python's standard library.
- `requirements.txt`: bounded dependencies for the Transformers 4.x API.
- `.gitignore`: excludes datasets, notebooks, credentials, and generated weights.

## Setup

Python 3.10 or later is recommended. Full fine-tuning of the 1.5B model requires
substantial memory; the small batch size and gradient checkpointing reduce
activation memory but do not eliminate optimizer and model memory requirements.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v test_data_utils
python slm_classifier.py --help
```

## Data Contract

Supply local CSV files with nonempty `input` (transcript) and `output` (class)
columns. The `instruction` column in the original exports is intentionally unused
for supervised classification. Supply already-preprocessed participant responses;
the code does not implement CLAN parsing or automatic transcription. Preserve
fillers and repetitions if following the research preprocessing protocol.

The training set must contain exactly two classes. Labels are learned once from
training, saved with the checkpoint, and reused unchanged for evaluation. Set
`--positive-label` to the exact AD label in your data to obtain sensitivity,
specificity, and positive-class ROC-AUC; the code never guesses which class is AD.

Normalized duplicate text across supplied splits is rejected. Where a participant
ID column exists, pass `--group-col participant_id` and construct participant-disjoint
splits before running. The supplied CSV schemas contain no participant ID, so their
participant separation cannot be verified. Different recordings from the same
person may otherwise leak information between splits. Text overlap checks alone
cannot resolve this limitation.

## Train

Use local, authorized datasets and a new output directory for each run:

```bash
python slm_classifier.py train \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --train-csv data/train_data.csv \
  --val-csv data/val_data.csv \
  --output-dir runs/deepseek-01 \
  --epochs 10 --batch-size 2 --gradient-accumulation 8
```

Append `--positive-label YOUR_EXACT_AD_LABEL` when the class semantics are known.
Use `--model gpt2` for the original GPT-2 notebook checkpoint, or
`--model HuggingFaceTB/SmolLM2-135M-Instruct` for the SmolLM2 alternative.
The original `gpt2` identifier selects GPT-2 small, not the 1.5B GPT-2 variant
described in the papers. Use `gpt2-xl` explicitly when that variant is intended.

For controlled comparisons, hold splits and seed fixed, create a new run
directory for every trial, and rank trials using validation macro-F1. Early
stopping and best-checkpoint selection use the same metric. This compact sample
does not run the original Bayesian sweep or enable remote experiment logging.
Pin `--revision` to a model commit for repeatability and retain your installed
package versions with local experiment records.

## Evaluate Without Retraining

After freezing model choice, run held-out evaluation. Include train and validation
paths so the test set is also checked for overlap:

```bash
python slm_classifier.py evaluate \
  --checkpoint runs/deepseek-01/best_model \
  --test-csv data/test_data.csv \
  --train-csv data/train_data.csv \
  --val-csv data/val_data.csv \
  --output-dir runs/deepseek-test --cpu --offline
```

Evaluation uses the saved maximum sequence length and class mapping. Results are
written to local JSON: accuracy, balanced accuracy, micro/macro/weighted F1,
per-class precision/recall/F1, and a confusion matrix with explicit class order.
ROC-AUC requires a configured positive label and both classes in the test set.
Sensitivity or specificity is null when its denominator has no observations.
No participant-level predictions or transcripts are exported.

## Privacy and Scope

Models execute locally. Initial model downloads contact Hugging Face; use cached
models with `--offline` for offline operation. Hub uploads and experiment-reporting
integrations are disabled. Checkpoints can retain information about training data:
keep all run folders and metrics on approved storage, regardless of `.gitignore`.

This implementation retains the notebooks' native sequence-classification heads.
The newer manuscript instead describes mean pooling; it also reports a 70/15/15
split, whereas the earlier preprint describes 80/10/10. Existing split files are
accepted as supplied, and this repository does not claim to reproduce either
paper's exact protocol. It includes full fine-tuning only, not the papers' LoRA
or instruction-fine-tuning experiments.

The DeepSeek checkpoint is the dense Qwen-based distilled 1.5B model, not the
full DeepSeek-R1 mixture-of-experts model. See the
[official model card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B).

Generated reasoning traces discussed in the paper are illustrative and do not
establish clinical explainability. This classifier does not generate such traces.
The code is for research use only, not clinical diagnosis.

## Validation Status

Four synthetic unit tests passed, Python syntax checks passed, and CLI help was
verified. A local check of the train/validation/test CSVs found no
normalized-text overlap. Participant separation remains unverified. The full training
and evaluation runs reported in the papers were performed on GPU hardware with the
restricted dataset.
The dependency ranges are not a tested environment lockfile. Run the commands
above in a configured environment before relying on experimental results.
