# PALetter

PALetter is a research repository for evaluating **large language models (LLMs) in political text classification**, with a focus on how model selection, prompting strategies, and prompt formatting affect classification performance and stability.

The repository contains two experimental examples.

## Repository Overview

| | Example 1: `ours/` | Example 2: `manifesto/` |
|---|---|---|
| **Task** | Biodiversity relevance classification | Political ideology classification |
| **Language** | German | English-translated / original manifesto texts |
| **Labels** | 0 = Not relevant, 1 = Relevant | 0 = Left, 1 = Right |
| **Models** | GPT-4o, Llama 3.1, Apertus, Qwen3 | GPT-4o and other configured models |
| **Prompt experiments** | Three modes × three formatting versions | Seven prompt variants |
| **Evaluation** | Classification performance | Classification performance and prompt stability |

---

# Example 1: Biodiversity Classification (`ours/`)

This experiment classifies German political text sentences according to their relevance to biodiversity.

- **0:** Not related to biodiversity
- **1:** Related to biodiversity

## 1. Installation

```bash
pip install pandas numpy scikit-learn tqdm openai torch transformers accelerate
```

## 2. Dataset

The default input file is `input_new.csv`, containing the following columns:

| Column | Description |
|---|---|
| `sentence` | German text to classify |
| `label_human` | Human-annotated binary label (0 or 1) |

The dataset configuration can be modified in each script:

```python
CSV_FILE = "input_new.csv"
TEXT_COLUMN = "sentence"
LABEL_COLUMN = "label_human"
```

## 3. Configuration

Before running an experiment, **modify the configuration parameters directly in the corresponding Python script**.

### Prompt version and classification mode

```python
prompt_version = 1
mode = 'd'
```

Three prompt formatting versions are available:

| Version | Description |
|---|---|
| `1` | Original prompt formatting |
| `2` | Additional indentation |
| `3` | Compact formatting |

Three classification modes are supported:

| Mode | Description |
|---|---|
| `z` | Zero-shot classification |
| `d` | Domain-specific classification using German biodiversity keywords |
| `o` | Example-based classification using one labeled example from each class |

### Model selection

| Script | Default model |
|---|---|
| `gpt-improved.py` | `gpt-4o` |
| `llama-predict.py` | `meta-llama/Llama-3.1-8B-Instruct` |
| `apertus-predict.py` | `swiss-ai/Apertus-8B-Instruct-2509` |
| `qwen-predict.py` | `Qwen/Qwen3-32B/Qwen3-8B/...` |

Change `MODEL_NAME` in the relevant script to select a model.

For Qwen, the model size is configured using:

```python
size = '32B'
MODEL_NAME = f"Qwen/Qwen3-{size}"
```

For GPT-4o, **configure your OpenAI API key in the code**. We recommend using an environment variable rather than hardcoding credentials:

```python
import os
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
```


### Other settings

The default generation settings are:

```python
temperature = 0.2
do_sample = True  # Local models
```

The scripts perform experiments using random seeds from **421 to 430**.

## 4. Run

From the `ours/` directory, run the script for your selected model:

```bash
python gpt-improved.py
# or
python llama-predict.py
# or
python apertus-predict.py
# or
python qwen-predict.py
```

Run one script at a time. Local models require sufficient computational resources, particularly Qwen3-32B.

Ensure that the output directory exists:

```bash
mkdir -p results-dec
```

## 5. Results

Results are saved in `results-dec/`.

Each experiment produces:

- **Predictions CSV:** Predicted labels and classification correctness for individual sentences.
- **Summary JSON:** Accuracy, precision, recall, F1-score, and confusion matrix.

Output filenames include experiment settings such as model, classification mode, temperature, random seed, and prompt version.

These outputs support comparisons across different models and prompt configurations.

---

# Example 2: Political Manifesto Classification (`manifesto/`)

This experiment classifies political manifestos into **Left (0)** and **Right (1)** categories and examines the stability of model predictions across different prompt formats.

## 1. Installation and Environment

Install the required Python packages for the GPT-based experiment:

```bash
pip install openai pandas numpy scikit-learn tqdm
```

For the GPT implementation (`gpt-predict.py`), configure your OpenAI API key before running. Prefer loading it from an environment variable rather than embedding it in the script.

**Apertus environment requirement:** Use `transformers==4.56.0` when running the Apertus implementation.

```bash
pip install transformers==4.56.0
```

Additional dependencies, such as PyTorch, are required for locally hosted models.

## 2. Dataset

Select the input CSV file in the corresponding Python script:

```python
CSV_FILE = "manifesto_with_text_en_complete_updated.csv"

# Alternative datasets:
# CSV_FILE = "sampled_4to1.csv"
# CSV_FILE = "manifesto_prompt_stability_500.csv"
```

The datasets contain political manifesto texts and their corresponding RILE ideology scores.

### Required columns

| Column | Description |
|---|---|
| `text` | Original manifesto text |
| `text_en` | English-translated manifesto text, where available |
| `rile` | Original Left–Right ideological score |

**Important:** Set `TEXT_COLUMN` according to the selected dataset. Use `text_en` for the English-translated dataset and `text` for datasets containing original texts.

Binary classification labels are derived from RILE scores:

- **0 = Left:** RILE < 0
- **1 = Right:** RILE ≥ 0

## 3. Configuration

Adjust the following parameters in the experiment script:

```python
MODEL_NAME = "gpt-4o"
TEXT_COLUMN = "text_en"
RILE_COLUMN = "rile"

temperature = 0.0
max_text_chars = 6000
```

The model and text column can be changed depending on the experiment and selected dataset.

### Prompt variations

The prompt-stability experiment evaluates seven variants:

| Prompt variant | Description |
|---|---|
| `normal` | Standard prompt formatting |
| `compact` | Compact formatting |
| `extra_spaces` | Additional spaces |
| `tabs` | Tab-based formatting |
| `label_order_reversed` | Reversed label order |
| `minimal` | Minimal instructions |
| `question_style` | Question-style instructions |

These variants are designed to test whether changes in prompt wording or formatting influence classification results.

## 4. Run

From the `manifesto/` directory, run the relevant Python experiment script.

For example:

```bash
python gpt-predict.py
```

Select the script corresponding to the model and experiment you want to run.

The prompt-stability experiment performs classification, calculates evaluation metrics, and compares predictions across prompt variants.

## 5. Results

For the English prompt-stability experiment, results are saved in:

```text
results_manifesto_prompt_stability_en/
```

Three types of output files are generated:

| Output | Description |
|---|---|
| **Predictions CSV** | Individual predictions across prompt variants |
| **Summary JSON** | Accuracy, precision, recall, F1-score, and confusion matrices |
| **Bucket Instability CSV** | Prediction instability grouped by absolute RILE score |

### Prompt stability analysis

Prediction instability is measured by whether a manifesto receives different classifications across prompt variants.

The experiment also groups manifestos by the absolute value of their RILE scores to investigate whether texts closer to the ideological center exhibit greater prompt sensitivity.

---

## Evaluation

Both examples evaluate classification performance using:

- Accuracy
- Precision and recall
- F1-score
- Confusion matrix

The biodiversity experiment focuses on comparisons across models, prompting modes, and formatting versions, while the manifesto experiment additionally investigates **prediction stability and its relationship with ideological positioning**.

## Reproducibility

For reproducible comparisons, record the model name, dataset, prompt version or variant, classification mode (where applicable), temperature, and random seed used for each experiment.
