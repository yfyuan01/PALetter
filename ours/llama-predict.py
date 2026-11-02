import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
import random
import numpy as np
import os
# torch.manual_seed(421)
# ============================================================================
# CONFIGURATION - CHANGE THESE
# ============================================================================
CSV_FILE = "input_new.csv"  # Change this to your CSV file path
TEXT_COLUMN = "sentence"
LABEL_COLUMN = "label_human"
# size = '0.6B'
MODEL_NAME = f"meta-llama/Llama-3.1-8B-Instruct"
temperature = 0.2
do_sample = True
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
    trust_remote_code=True
)
if device == "cpu":
    model = model.to(device)
model.eval()
print(f"Model loaded on {device}")
# Load data
print(f"\nLoading data from {CSV_FILE}...")
df = pd.read_csv(CSV_FILE)
df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
sentences = df[TEXT_COLUMN].tolist()
true_labels = df[LABEL_COLUMN].astype(int).tolist()
print(f"Loaded {len(sentences)} sentences")
print(f"Label distribution: 0={true_labels.count(0)}, 1={true_labels.count(1)}")

# ⭐ CHANGE THIS: Set your mode ('z', 'd', or 'o')
# mode = 'z'  # 'z' = zero-shot, 'd' = domain keywords, 'o' = one-shot examples

# ⭐ KEYWORDS FOR MODE 'd' - German biodiversity keywords
KEYWORDS = [
    "Biodiversität",
    "Biodiversitätsprogramm",
    "biodiversitätsfreundliche",
    "Waldbiodiversität",
    "biologisch",
    "Vielfalt",
    "Tierpopulation",
    "Pflanzenpopulation",
    "Tierart",
    "Pflanzenart",
    "Ökosystem",
    "Biotop",
    "Lebensraum",
    "Lebensräume",
    "Ökosystemleistung"
]
# ============================================================================

# Prompt templates for different modes
for seed in range(421,431):
    for t in range(1):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        print(f"\n{t}-th time: === Running classification with seed {seed} ===")
        prompt_version = 3
        # ⭐ CHANGE THIS: Set your mode ('z', 'd', or 'o')
        mode = 'o'  # 'z' = zero-shot, 'd' = domain keywords, 'o' = one-shot examples

        # ⭐ KEYWORDS FOR MODE 'd' - German biodiversity keywords
        KEYWORDS = [
            "Biodiversität",
            "Biodiversitätsprogramm",
            "biodiversitätsfreundliche",
            "Waldbiodiversität",
            "biologisch",
            "Vielfalt",
            "Tierpopulation",
            "Pflanzenpopulation",
            "Tierart",
            "Pflanzenart",
            "Ökosystem",
            "Biotop",
            "Lebensraum",
            "Lebensräume",
            "Ökosystemleistung"
        ]
        # ============================================================================

        # Prompt templates for different modes
        if mode == 'z':
            if prompt_version ==2:
                PROMPT_TEMPLATE = """Du bist ein KI-Modell, das bestimmt, ob ein gegebener Text sich mit Biodiversität befasst.

    Anweisung:
    Klassifiziere den Text wie folgt:

    1, wenn der Text Themen im Zusammenhang mit Biodiversität behandelt.

    0, wenn dies nicht der Fall ist.

    Klassifikationsaufgabe:
    Text: "{}"

    Gib nur 0 oder 1 aus, ohne zusätzlichen Kommentar.
"""
            elif prompt_version ==1: 
                PROMPT_TEMPLATE = """Du bist ein KI-Modell, das bestimmt, ob ein gegebener Text sich mit Biodiversität befasst.

Anweisung:
Klassifiziere den Text wie folgt:

1, wenn der Text Themen im Zusammenhang mit Biodiversität behandelt.

0, wenn dies nicht der Fall ist.

Klassifikationsaufgabe:
Text: "{}"

Gib nur 0 oder 1 aus, ohne zusätzlichen Kommentar.
"""
            elif prompt_version == 3:
                PROMPT_TEMPLATE = """Du bist ein KI-Modell, das bestimmt, ob ein gegebener Text sich mit Biodiversität befasst.
Anweisung:
Klassifiziere den Text wie folgt:
1, wenn der Text Themen im Zusammenhang mit Biodiversität behandelt.
0, wenn dies nicht der Fall ist.
Klassifikationsaufgabe:
Text: "{}"
Gib nur 0 oder 1 aus, ohne zusätzlichen Kommentar.
"""
            # Zero-shot: Direct classification
#             PROMPT_TEMPLATE = """Du bist ein KI-Modell, das bestimmt, ob ein gegebener Text sich mit Biodiversität befasst.
# Anweisung:
# Klassifiziere den Text wie folgt:
# 1, wenn der Text Themen im Zusammenhang mit Biodiversität behandelt.
# 0, wenn dies nicht der Fall ist.
# Klassifikationsaufgabe:
# Text: "{}"
# Gib nur 0 oder 1 aus, ohne zusätzlichen Kommentar.
#         """

        elif mode == 'd':
            # Domain-specific: Use keywords
            keywords_str = ", ".join(KEYWORDS)
            if prompt_version == 1:
                PROMPT_TEMPLATE = f"""Du bist ein KI-Modell, das bewertet, ob ein gegebener Text für eine Menge von Schlüsselwörtern relevant ist.
Anweisungen:
Du erhältst eine Menge von Schlüsselwörtern und einen Text. 
Antworte mit 1, wenn der Text für die Schlüsselwörter relevant ist. 
Antworte mit 0, wenn der Text nicht relevant ist.
Gib nur 0 oder 1 aus, nichts anderes.

Schlüsselwörter: [{keywords_str}]

Text: "{{}}"

Schreibe nur 0 oder 1, kommentiere die Antwort nicht.
        """
            elif prompt_version == 2:
                PROMPT_TEMPLATE = f"""Du bist ein KI-Modell, das bewertet, ob ein gegebener Text für eine Menge von Schlüsselwörtern relevant ist.
    Anweisungen:
    Du erhältst eine Menge von Schlüsselwörtern und einen Text. 
    Antworte mit 1, wenn der Text für die Schlüsselwörter relevant ist. 
    Antworte mit 0, wenn der Text nicht relevant ist.
    Gib nur 0 oder 1 aus, nichts anderes.

    Schlüsselwörter: [{keywords_str}]

    Text: "{{}}"

    Schreibe nur 0 oder 1, kommentiere die Antwort nicht.
        """
            elif prompt_version == 3:
                PROMPT_TEMPLATE = f"""Du bist ein KI-Modell, das bewertet, ob ein gegebener Text für eine Menge von Schlüsselwörtern relevant ist.
Anweisungen:
Du erhältst eine Menge von Schlüsselwörtern und einen Text. 
Antworte mit 1, wenn der Text für die Schlüsselwörter relevant ist. 
Antworte mit 0, wenn der Text nicht relevant ist.
Gib nur 0 oder 1 aus, nichts anderes.
Schlüsselwörter: [{keywords_str}]
Text: "{{}}"
Schreibe nur 0 oder 1, kommentiere die Antwort nicht.
        """


        elif mode == 'o':
            # One-shot: Will be dynamically filled with examples
            # Template structure (will be filled in the loop):
            # - example_text_0, example_label_0
            # - example_text_1, example_label_1  
            # - test_text
            if prompt_version == 1:
                PROMPT_TEMPLATE = """Du bist ein KI-Modell, das klassifiziert, ob ein gegebener Text basierend auf bereitgestellten Beispielen zur Kategorie 1 oder 0 gehört.

Beispiele:
- Text: "{example_0}"
- Label: {label_0}

- Text: "{example_1}"
- Label: {label_1}

Klassifizierungsaufgabe:
Text: "{test_text}"

Schreibe nur 0 oder 1, kommentiere die Antwort nicht."""
            elif prompt_version ==2:
                PROMPT_TEMPLATE = """Du bist ein KI-Modell, das klassifiziert, ob ein gegebener Text basierend auf bereitgestellten Beispielen zur Kategorie 1 oder 0 gehört.

    Beispiele:
    - Text: "{example_0}"
    - Label: {label_0}

    - Text: "{example_1}"
    - Label: {label_1}

    Klassifizierungsaufgabe:
    Text: "{test_text}"

    Schreibe nur 0 oder 1, kommentiere die Antwort nicht."""
            elif prompt_version ==3:
                PROMPT_TEMPLATE = """Du bist ein KI-Modell, das klassifiziert, ob ein gegebener Text basierend auf bereitgestellten Beispielen zur Kategorie 1 oder 0 gehört.
Beispiele:
- Text: "{example_0}"
- Label: {label_0}
- Text: "{example_1}"
- Label: {label_1}
Klassifizierungsaufgabe:
Text: "{test_text}"
Schreibe nur 0 oder 1, kommentiere die Antwort nicht."""


        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'z', 'd', or 'o'")

        print(f"Mode: {mode}")
        print(f"Model: {MODEL_NAME}")

        # Load model
       
        # For mode 'o', prepare example pools
        if mode == 'o':
            # Separate examples by class
            class_0_examples = [(s, l) for s, l in zip(sentences, true_labels) if l == 0]
            class_1_examples = [(s, l) for s, l in zip(sentences, true_labels) if l == 1]
            
            print(f"Example pool: {len(class_0_examples)} class 0, {len(class_1_examples)} class 1")
            
            if len(class_0_examples) < 1 or len(class_1_examples) < 1:
                raise ValueError("Not enough examples for one-shot learning. Need at least 1 example of each class.")

        # Classify sentences
        print("\nClassifying sentences...")
        predictions = []

        for idx, sentence in enumerate(tqdm(sentences)):
            # Create prompt based on mode
            if mode == 'z' or mode == 'd':
                # Simple formatting
                prompt = PROMPT_TEMPLATE.format(sentence)
            
            elif mode == 'o':
                # Randomly sample one example from each class
                # Make sure we don't sample the current sentence
                available_class_0 = [ex for ex in class_0_examples if ex[0] != sentence]
                available_class_1 = [ex for ex in class_1_examples if ex[0] != sentence]
                
                # If no examples available (shouldn't happen), use original pools
                if not available_class_0:
                    available_class_0 = class_0_examples
                if not available_class_1:
                    available_class_1 = class_1_examples
                
                # Sample one from each class
                example_0_text, example_0_label = random.choice(available_class_0)
                example_1_text, example_1_label = random.choice(available_class_1)
                
                # Format prompt with examples
                prompt = PROMPT_TEMPLATE.format(
                    example_0=example_0_text,
                    label_0=example_0_label,
                    example_1=example_1_text,
                    label_1=example_1_label,
                    test_text=sentence
                )
            
            # Tokenize and generate
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            # Extract label
            if '1' in response[:20]:
                predictions.append(1)
            elif '0' in response[:20]:
                predictions.append(0)
            else:
                print(response)
            #     # If response is unclear, print warning and default to 0
            #     if idx < 5:  # Only print first few warnings to avoid spam
            #         print(f"\nWarning: Unclear response '{response}' for sentence {idx}, defaulting to 0")
                predictions.append(0)
        # breakpoint()
        # Calculate metrics
        print("\n" + "="*60)
        print(f"RESULTS - Mode: {mode}, Model: llama")
        print("="*60)

        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary', zero_division=0
        )

        print(f"\nBinary Metrics (treating class 1 as positive):")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")

        # Per-class metrics
        precision_class, recall_class, f1_class, support = precision_recall_fscore_support(
            true_labels, predictions, average=None, zero_division=0
        )

        print(f"\nPer-Class Metrics:")
        print(f"  Class 0: P={precision_class[0]:.4f}, R={recall_class[0]:.4f}, F1={f1_class[0]:.4f}, Support={support[0]}")
        print(f"  Class 1: P={precision_class[1]:.4f}, R={recall_class[1]:.4f}, F1={f1_class[1]:.4f}, Support={support[1]}")

        # Macro-averaged metrics
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            true_labels, predictions, average='macro', zero_division=0
        )

        print(f"\nMacro-Averaged Metrics:")
        print(f"  Precision: {precision_macro:.4f}")
        print(f"  Recall:    {recall_macro:.4f}")
        print(f"  F1-Score:  {f1_macro:.4f}")

        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        print(f"\nConfusion Matrix:")
        print(f"              Predicted 0  Predicted 1")
        print(f"  Actual 0    {cm[0][0]:>11d}  {cm[0][1]:>11d}")
        print(f"  Actual 1    {cm[1][0]:>11d}  {cm[1][1]:>11d}")

        # Calculate accuracy
        accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
        print(f"\nAccuracy: {accuracy:.4f}")

        print("="*60)

        # Save predictions
        if os.path.exists(f'results-dec/predictions_llama_{mode}_{temperature}_{do_sample}_{seed}_{t}_{prompt_version}.csv'):
            output_file = f'results-dec/predictions_llama_{mode}_{temperature}_{do_sample}_{seed}_{t+1}_{prompt_version}.csv'
        else:
            output_file = f'results-dec/predictions_llama_{mode}_{temperature}_{do_sample}_{seed}_{t}_{prompt_version}.csv'
        df['predicted'] = predictions
        df['correct'] = [int(t == p) for t, p in zip(true_labels, predictions)]
        df.to_csv(output_file, index=False)
        print(f"\nPredictions saved to: {output_file}")

        # Save summary
        summary = {
            'mode': mode,
            'total_samples': len(sentences),
            'accuracy': accuracy,
            'precision_binary': precision,
            'recall_binary': recall,
            'f1_binary': f1,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'confusion_matrix': cm.tolist()
        }

        import json
        if os.path.exists(f'results-dec/summary_llama_{mode}_{temperature}_{do_sample}_{seed}_{t}_{prompt_version}.json'):
            summary_file = f'results-dec/summary_llama_{mode}_{temperature}_{do_sample}_{seed}_{t+1}_{prompt_version}.json'
        else:
            summary_file = f'results-dec/sduiummary_llama_{mode}_{temperature}_{do_sample}_{seed}_{t}_{prompt_version}.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Summary saved to: {summary_file}")
