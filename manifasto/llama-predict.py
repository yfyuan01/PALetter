import os
import re
import json
import random
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_SIZE = "8B"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# CSV_FILE = 'sampled_4to1.csv'
CSV_FILE = "manifesto_with_text_en_complete_updated.csv"
# CSV_FILE = "manifesto_prompt_stability_500.csv"  # your final_df csv
TEXT_COLUMN = "text"
RILE_COLUMN = "rile"

OUTPUT_DIR = "results_manifesto_prompt_stability_en"
os.makedirs(OUTPUT_DIR, exist_ok=True)

temperature = 0.0
do_sample = False
max_new_tokens = 30
max_input_tokens = 8192
max_text_chars = 6000

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# LOAD MODEL
# ============================================================================

print("\nLoading model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
)

if device == "cpu":
    model = model.to(device)

model.eval()

print(f"Model loaded: {MODEL_NAME}")
print(f"Device: {device}")

# ============================================================================
# LOAD DATA
# ============================================================================

print(f"\nLoading data from {CSV_FILE}...")

df = pd.read_csv(CSV_FILE)
df = df.dropna(subset=[TEXT_COLUMN, RILE_COLUMN]).copy()

# Binary ideology label:
# 0 = Left, 1 = Right
# You can change this if you want Center removed or 5-class labels later.
df["label"] = df[RILE_COLUMN].apply(lambda x: 0 if float(x) < 0 else 1)

texts = df[TEXT_COLUMN].astype(str).tolist()
true_labels = df["label"].astype(int).tolist()

print(f"Loaded {len(df)} documents")
print(f"Label distribution: Left/0={true_labels.count(0)}, Right/1={true_labels.count(1)}")

# ============================================================================
# PROMPT FORMAT VARIANTS
# ============================================================================

PROMPT_VARIANTS = {
    "normal": """You are a political ideology classifier.

Classify the following political manifesto as:

0 = Left
1 = Right

Text:
{}

Return only 0 or 1.""",

    "compact": """You are a political ideology classifier.
Classify the following political manifesto as:
0 = Left
1 = Right
Text:
{}
Return only 0 or 1.""",

    "extra_spaces": """You are a political ideology classifier.

Classify   the   following   political   manifesto   as:

0  =  Left

1  =  Right

Text:

{}

Return only 0 or 1.""",

    "tabs": """You are a political ideology classifier.

\tClassify the following political manifesto as:

\t0 = Left
\t1 = Right

\tText:
\t{}

\tReturn only 0 or 1.""",

    "label_order_reversed": """You are a political ideology classifier.

Classify the following political manifesto as:

1 = Right
0 = Left

Text:
{}

Return only 0 or 1.""",

    "minimal": """Classify this political manifesto.

0 = Left
1 = Right

{}

Only output 0 or 1.""",

    "question_style": """Based on the following manifesto, is the political ideology closer to the Left or the Right?

Use:
0 for Left
1 for Right

Manifesto:
{}

Answer with only one digit."""
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_binary_label(response: str) -> int:
    """
    Extract 0/1 from model response.
    Defaults to 0 if unclear.
    """
    response = response.strip()

    match = re.search(r"\b[01]\b", response[:50])
    if match:
        return int(match.group())

    if "1" in response[:20]:
        return 1
    if "0" in response[:20]:
        return 0

    print(f"Unclear response: {response[:100]}")
    return 0


def classify_one_text(text: str, prompt_template: str) -> int:
    text = text[:max_text_chars]
    prompt = prompt_template.format(text)
    messages = [
    {"role": "user", "content": prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)
    # print(type(inputs))
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2,
            do_sample=False,      # deterministic
            temperature=0.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    return extract_binary_label(response)


def evaluate_predictions(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)

    precision_bin, recall_bin, f1_bin, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0
    )

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "precision_binary": precision_bin,
        "recall_binary": recall_bin,
        "f1_binary": f1_bin,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "confusion_matrix": cm.tolist()
    }


# ============================================================================
# RUN EXPERIMENT
# ============================================================================

all_summaries = []

for seed in range(421, 422):
    print(f"\n================ Seed {seed} ================")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    for prompt_name, prompt_template in PROMPT_VARIANTS.items():
        print(f"\nRunning prompt variant: {prompt_name}")

        predictions = []

        for text in tqdm(texts):
            pred = classify_one_text(text, prompt_template)
            predictions.append(pred)

        pred_col = f"pred_{prompt_name}_seed{seed}"
        df[pred_col] = predictions

        metrics = evaluate_predictions(true_labels, predictions)
        metrics.update({
            "seed": seed,
            "prompt_variant": prompt_name,
            "model": MODEL_NAME,
            "temperature": temperature,
            "do_sample": do_sample,
            "num_samples": len(df)
        })

        all_summaries.append(metrics)

        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision_binary']:.4f}")
        print(f"Recall:    {metrics['recall_binary']:.4f}")
        print(f"F1:        {metrics['f1_binary']:.4f}")
        print(f"Macro-F1:  {metrics['f1_macro']:.4f}")
        print("Confusion Matrix:")
        print(metrics["confusion_matrix"])

# ============================================================================
# PROMPT FORMAT INSTABILITY ANALYSIS
# ============================================================================

pred_cols = [c for c in df.columns if c.startswith("pred_")]

df["num_unique_predictions"] = df[pred_cols].nunique(axis=1)
df["format_unstable"] = (df["num_unique_predictions"] > 1).astype(int)

format_instability_rate = df["format_unstable"].mean()

print("\n================ FORMAT INSTABILITY ================")
print(f"Format instability rate: {format_instability_rate:.4f}")
print(f"Unstable examples: {df['format_unstable'].sum()} / {len(df)}")

# Instability by ideological ambiguity
df["abs_rile"] = df[RILE_COLUMN].abs()

def rile_bucket(abs_rile):
    if abs_rile < 5:
        return "center_abs_rile_lt_5"
    elif abs_rile < 15:
        return "moderate_abs_rile_5_15"
    else:
        return "clear_abs_rile_ge_15"

df["rile_bucket"] = df["abs_rile"].apply(rile_bucket)

bucket_instability = (
    df.groupby("rile_bucket")["format_unstable"]
    .agg(["mean", "sum", "count"])
    .reset_index()
)

print("\nInstability by RILE bucket:")
print(bucket_instability)

# ============================================================================
# SAVE RESULTS
# ============================================================================

prediction_file = os.path.join(
    OUTPUT_DIR,
    f"manifesto_llama_predictions_{MODEL_SIZE}_temp{temperature}_sample{do_sample}.csv"
)

summary_file = os.path.join(
    OUTPUT_DIR,
    f"manifesto_llama_summary_{MODEL_SIZE}_temp{temperature}_sample{do_sample}.json"
)

bucket_file = os.path.join(
    OUTPUT_DIR,
    f"manifesto_llama_bucket_instability_{MODEL_SIZE}.csv"
)

summary_df = pd.DataFrame(all_summaries)

summary_df.to_json(
    summary_file,
    orient="records",
    indent=2,
    force_ascii=False
)

df.to_csv(prediction_file, index=False)
bucket_instability.to_csv(bucket_file, index=False)

print("\nSaved files:")
print(prediction_file)
print(summary_file)
print(bucket_file)