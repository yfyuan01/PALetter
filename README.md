# Biodiversity Text Classification

This project uses GPT-4o, Llama 3.1, Apertus, and Qwen3 to classify German political texts according to their relevance to biodiversity.

- **0** = Not related to biodiversity
- **1** = Related to biodiversity

## 1. Installation

```bash
pip install pandas numpy scikit-learn tqdm openai torch transformers accelerate
```

## 2. Dataset

The default input dataset is `input_new.csv`, containing:

- `sentence`: German text to classify
- `label_human`: Human-annotated binary label (0 or 1)

## 3. Configuration

Before running, modify the corresponding parameters in each Python script.

**Prompt version and classification mode:**

```python
prompt_version = 1
mode = 'd'
```

Prompt versions:

- `1`: Original formatting
- `2`: Additional indentation
- `3`: Compact formatting

Classification modes:

- `z`: Zero-shot
- `d`: Domain-specific (biodiversity keywords)
- `o`: Example-based (one example per class)

**Model selection:**

| Script | Model |
|---|---|
| `gpt-improved.py` | GPT-4o / GPT-4o-mini |
| `llama-predict.py` | Llama-3.1-8B-Instruct |
| `apertus-predict.py` | Apertus-8B-Instruct-2509 |
| `qwen-predict.py` | Qwen3-32B |

For Qwen, the model size can be configured using:

```python
size = '32B'
MODEL_NAME = f"Qwen/Qwen3-{size}"
```

For GPT, configure your OpenAI API key securely:

```python
import os
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
```

## 4. Run

Run the corresponding script:

```bash
python gpt-improved.py
python llama-predict.py
python apertus-predict.py
python qwen-predict.py
```

Run only the script for the selected model. Local models require sufficient hardware resources, especially Qwen3-32B.

The scripts run experiments with seeds 421–430.

## 5. Results

Results are saved in the `results-dec/` directory.

Each experiment generates:

- **Predictions CSV:** Predicted labels and classification correctness
- **Summary JSON:** Accuracy, precision, recall, F1-score, and confusion matrix

Results can be compared across models, prompt versions, classification modes, and random seeds.

# PALetter Example 2

PALetter example 2 uses different models to classify political manifestos into Left (0) and Right (1) and evaluate prediction stability across different prompt formats.

## 1. Installation and Environments

Install the required packages:

```bash
pip install openai pandas numpy scikit-learn tqdm
```

In gpt-predict.py, line 43, input your own OPENAI API KEY.

For running apertus, it's necessary that you use transformers==4.56.0!

## 2. Dataset

Select the input CSV file in the Python script:

```python
CSV_FILE = "manifesto_with_text_en_complete_updated.csv"
# CSV_FILE = "sampled_4to1.csv"
# CSV_FILE = "manifesto_prompt_stability_500.csv"
```

The default dataset contains English-translated political manifestos.

Required columns:

- `text`: the manifesto text (**when running the code, change TEXT_COLUMN = "text"**)
- `text_en`: English translation of the manifesto text (**only for manifesto_with_text_en_complete_updated.csv!**)
- `rile`: Original Left–Right ideological score

Binary labels are generated from RILE scores:

- `0` = Left (RILE < 0)
- `1` = Right (RILE ≥ 0)

## 3. Configuration

```python
MODEL_NAME = "gpt-4o" (or change to other models)
TEXT_COLUMN = "text_en" #(change to TEXT_COLUMN = "text" when running sampled_4to1.csv/manifesto_with_text_en_complete_updated.csv)
RILE_COLUMN = "rile"

temperature = 0.0
max_text_chars = 6000
```

The experiment evaluates seven prompt variants: `normal`, `compact`, `extra_spaces`, `tabs`, `label_order_reversed`, `minimal`, and `question_style`.

## 4. Run

Run the Python experiment script:

```bash
python your_script_name.py
```

Replace `your_script_name.py` with the actual filename.

The script automatically performs classification, calculates evaluation metrics, and analyzes prediction instability across prompt variants.

## 5. Results

Results are saved in:

```text
results_manifesto_prompt_stability_en/
```

Three output files are generated:

- **Predictions CSV:** Individual predictions for each prompt variant.
- **Summary JSON:** Accuracy, precision, recall, F1-score, and confusion matrices.
- **Bucket Instability CSV:** Prediction instability grouped by absolute RILE score.

The instability analysis compares predictions across prompt variants to identify manifestos whose classifications change with prompt formatting or wording.
