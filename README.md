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

temperature = 0.2
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
