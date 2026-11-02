# ============================================================
# Manifesto Prompt Stability Experiment (Qwen API Version)
# Only the model inference backend has been changed from GPT-4o
# to the SDSC vLLM Qwen API.
# ============================================================

import os
import re
import json
import time
import random
import requests
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_SIZE = "Qwen3-32B-AWQ"
MODEL_NAME = "Qwen/Qwen3-32B-AWQ"

URL = "https://vllm-gateway-runai-sharedllm-ralf.inference.compute.datascience.ch/v1/chat/completions"

API_KEY = "eyJhbGciOiJSUzI1NiIsImtpZCI6IjFkYzYwZDBlNTY3M2IzZjc4ZjU4NmUxMzJmYjZlY2U1IiwidHlwIjoiSldUIn0.eyJpc3MiOiJodHRwczovL2F1dGhlbnRpay1zZXJ2ZXItcnVuYWktc2hhcmVkbGxtLXJhbGYuaW5mZXJlbmNlLmNvbXB1dGUuZGF0YXNjaWVuY2UuY2gvYXBwbGljYXRpb24vby92bGxtLyIsInN1YiI6ImUzZmM4MzI0ZTkxNWY4NDU1ZmYyYmVmMzVjOTcwMDFmMTNmMjRhNTliMTQ4MTk5YWVhZGUzZjNmNTMzZGFiMmIiLCJhdWQiOiJQOGRXMnZyTlBEYThkNDNxZDRCSzQ5ZUVEWUpGdHZZayIsImV4cCI6MTc3NTM4Mzc1OCwiaWF0IjoxNzcyNzkxNzU4LCJhdXRoX3RpbWUiOjE3NzI3OTE3NTgsImFjciI6ImdvYXV0aGVudGlrLmlvL3Byb3ZpZGVycy9vYXV0aDIvZGVmYXVsdCIsImVtYWlsIjoieWlmZWkueXVhbkBzZHNjLmV0aHouY2giLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwibmFtZSI6InlpZmVpIHl1YW4iLCJnaXZlbl9uYW1lIjoieWlmZWkgeXVhbiIsInByZWZlcnJlZF91c2VybmFtZSI6InlpZmVpIiwibmlja25hbWUiOiJ5aWZlaSIsImdyb3VwcyI6WyJkZW1vY3Jhc2NpIiwiZGVtb2NyYXNjaS9kZW1vY3Jhc2NpLXByaXZhdGUiLCJkZW1vY3Jhc2NpL2RlbW9jcmFzY2ktcHJpdmF0ZS9kZW1vY3Jhc2NpX3dlYnBhZ2UiLCJkZW1vY3Jhc2NpL2RlbW9jcmFzY2kta2ctcGlwZWxpbmUiLCJnaXRsYWIvZGVtb2NyYXNjaSIsImdpdGxhYi9kZW1vY3Jhc2NpL2RlbW9jcmFzY2ktcHJpdmF0ZSIsImdpdGxhYi9kZW1vY3Jhc2NpL2RlbW9jcmFzY2ktcHJpdmF0ZS9kZW1vY3Jhc2NpX3dlYnBhZ2UiLCJnaXRsYWIvZGVtb2NyYXNjaS9kZW1vY3Jhc2NpLWtnLXBpcGVsaW5lIl0sImF6cCI6IlA4ZFcydnJOUERhOGQ0M3FkNEJLNDllRURZSkZ0dllrIiwidWlkIjoiZXlXWGdXc0NOU3paNGJ5TWl6QmlLOWFoZmpJVGxTZGVid2l6b295NSJ9.ZphE-0KLppIOuS2ERiA5_lgdldw3H7HcLtZ9PtUJgFgyo6Ur53sqkoGPzR38IYu_VXGFz6w5pyIubYEzofjLhTqWq71bM0TRiCRi5mia914A4oWZvtIa-APTP0W0Ahjblr-8EOEDxiMBNArV06HZyvSdrrv2KVVfC8LFO_-fKOQAIGT3Jw2HhKbKD9dW3OPCmxy1HjKap8lFTbYSQzCpIRucjtDm1u4bqHLMsSM2OXn0O05Oeft2A0Vos5jXcu3AI_3VN4iUmY51luH5f6mevo1uR7JWQ1StsPoPwR0r2PNQp6QW0cgP6oaCCm84jLwlx8ng5MlYcgYjahjNnsL_R7L20emGVbzYan-6Dhg8ndKcmT3sUCYOWKK9e93u4y0gAcEuFfyOf6Xh5QGBJSYFRg9gdM4NJkgJhJEVLzEM1OyJr-O9mE_EiUFwjvH8WxCuc-Bk25vlqNtyJt0_BO3hgEuV8-DUph9vTN2oFY_vySjgHrMMHE8FugXWqhqtKD1GVoEOmqKaSr8Zhy23xK0MlJnrerx4N7rMdO1h8UKRN_ElIUdfSZbxHU5LObJ2hkccPRigbq98kmDuf8Y-eeHJ5YsF_lkUxrJzfyN8gQ2p7-Fj-oTnfO-ptOZ5o9LGHwiW_nTrbb5ltAD2HfdmhdMxlZu40jLihR3QZjVcDpqCvWM"

CSV_FILE = "manifesto_prompt_stability_500.csv"
TEXT_COLUMN = "text"
RILE_COLUMN = "rile"

OUTPUT_DIR = "results_manifesto_prompt_stability_qwen"
os.makedirs(OUTPUT_DIR, exist_ok=True)

temperature = 0.0
do_sample = False
max_text_chars = 6000

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

print("Using model:", MODEL_NAME)

# ============================================================
# DATA
# ============================================================

df = pd.read_csv(CSV_FILE)
df = df.dropna(subset=[TEXT_COLUMN, RILE_COLUMN]).copy()

df["label"] = df[RILE_COLUMN].apply(lambda x: 0 if float(x) < 0 else 1)

texts = df[TEXT_COLUMN].astype(str).tolist()
true_labels = df["label"].astype(int).tolist()

print("Loaded", len(df), "documents")

# ============================================================
# PROMPTS
# ============================================================

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

# ============================================================
# HELPERS
# ============================================================

def extract_binary_label(response: str) -> int:
    response = response.strip()

    if "</think>" in response:
        response = response.split("</think>")[-1].strip()

    m = re.search(r"\b[01]\b", response[:50])
    if m:
        return int(m.group())

    if "1" in response[:20]:
        return 1
    if "0" in response[:20]:
        return 0

    print("Unclear:", response[:100])
    return 0


def query_qwen(prompt, retries=3):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 2,
    }

    for i in range(retries):
        try:
            r = requests.post(
                URL,
                headers=HEADERS,
                json=payload,
                timeout=120,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print("Retry", i + 1, e)
            time.sleep(2)

    return ""


def classify_one_text(text, prompt_template):
    text = text[:max_text_chars]
    prompt = prompt_template.format(text)
    output = query_qwen(prompt)
    return extract_binary_label(output)


def evaluate_predictions(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)

    p_bin, r_bin, f_bin, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "precision_binary": p_bin,
        "recall_binary": r_bin,
        "f1_binary": f_bin,
        "precision_macro": p_mac,
        "recall_macro": r_mac,
        "f1_macro": f_mac,
        "confusion_matrix": cm.tolist(),
    }

# ============================================================
# RUN
# ============================================================

all_summaries = []

# API models are deterministic at temperature=0,
# so only one seed is needed.
for seed in [421]:

    random.seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print("Seed", seed)

    for prompt_name, prompt_template in PROMPT_VARIANTS.items():

        print("Prompt:", prompt_name)

        predictions = []

        for text in tqdm(texts):
            predictions.append(classify_one_text(text, prompt_template))

        col = f"pred_{prompt_name}_seed{seed}"
        df[col] = predictions

        metrics = evaluate_predictions(true_labels, predictions)
        metrics.update(
            {
                "seed": seed,
                "prompt_variant": prompt_name,
                "model": MODEL_NAME,
                "temperature": temperature,
                "num_samples": len(df),
            }
        )

        all_summaries.append(metrics)

        print(metrics)

# ============================================================
# INSTABILITY
# ============================================================

pred_cols = [c for c in df.columns if c.startswith("pred_")]

df["num_unique_predictions"] = df[pred_cols].nunique(axis=1)
df["format_unstable"] = (df["num_unique_predictions"] > 1).astype(int)

df["abs_rile"] = df[RILE_COLUMN].abs()

def rile_bucket(x):
    if x < 5:
        return "center_abs_rile_lt_5"
    elif x < 15:
        return "moderate_abs_rile_5_15"
    return "clear_abs_rile_ge_15"

df["rile_bucket"] = df["abs_rile"].apply(rile_bucket)

bucket_instability = (
    df.groupby("rile_bucket")["format_unstable"]
    .agg(["mean", "sum", "count"])
    .reset_index()
)

summary_df = pd.DataFrame(all_summaries)

summary_df.to_json(
    os.path.join(OUTPUT_DIR, "summary.json"),
    orient="records",
    indent=2,
    force_ascii=False,
)

df.to_csv(
    os.path.join(OUTPUT_DIR, "predictions.csv"),
    index=False,
)

bucket_instability.to_csv(
    os.path.join(OUTPUT_DIR, "bucket_instability.csv"),
    index=False,
)

print("Finished.")


