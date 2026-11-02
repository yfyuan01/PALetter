import pandas as pd
from openai import OpenAI
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
import os
import random
import json
import numpy as np
# ============================================================================
# CONFIGURATION - CHANGE THESE
# ============================================================================
CSV_FILE = "input_new.csv"  # Change this to your CSV file path
TEXT_COLUMN = "sentence"
LABEL_COLUMN = "label_human"
MODEL_NAME = "gpt-4o"  # or "gpt-4o-mini" for cheaper option

# ⭐ CHANGE THIS: Set your mode ('z', 'd', or 'o')
mode = 'o'  # 'z' = zero-shot, 'd' = domain keywords, 'o' = one-shot examples

# ⭐ SET YOUR API KEY
OPENAI_API_KEY = ""
temperature = 0.2
prompt_version = 1
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
for seed in range(421,431):
    for t in range(1):
        random.seed(seed)
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        print(f"\n{t}-th time: === Running classification with seed {seed} ===")
# Prompt templates for different modes
        if mode == 'z':
    # Zero-shot: Direct classification
            if prompt_version ==1: 
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
            elif prompt_version ==2:
                PROMPT_TEMPLATE = """Du bist ein KI-Modell, das bestimmt, ob ein gegebener Text sich mit Biodiversität befasst.

    Anweisung:
    Klassifiziere den Text wie folgt:

    1, wenn der Text Themen im Zusammenhang mit Biodiversität behandelt.

    0, wenn dies nicht der Fall ist.

    Klassifikationsaufgabe:
    Text: "{}"

    Gib nur 0 oder 1 aus, ohne zusätzlichen Kommentar.
"""

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

        # Initialize OpenAI client
        print("\nInitializing OpenAI client...")
        if not OPENAI_API_KEY:
            print("ERROR: OPENAI_API_KEY not set!")
            print("Set it in the script or as environment variable")
            exit(1)

        client = OpenAI(api_key=OPENAI_API_KEY)
        print("OpenAI client initialized successfully")

        # Load data
        print(f"\nLoading data from {CSV_FILE}...")
        df = pd.read_csv(CSV_FILE)
        df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
        sentences = df[TEXT_COLUMN].tolist()
        true_labels = df[LABEL_COLUMN].astype(int).tolist()
        print(f"Loaded {len(sentences)} sentences")
        print(f"Label distribution: 0={true_labels.count(0)}, 1={true_labels.count(1)}")

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
        failed_indices = []

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
            
            # Call OpenAI API
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a precise classifier. Respond only with 0 or 1."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=5,
                    seed=seed
                )
                
                answer = response.choices[0].message.content.strip()
                
                # Extract label
                if '1' in answer[:20]:
                    predictions.append(1)
                elif '0' in answer[:20]:
                    predictions.append(0)
                else:
                    # If response is unclear, print warning and default to 0
                    if idx < 5:  # Only print first few warnings
                        print(f"\nWarning: Unclear response '{answer}' for sentence {idx}, defaulting to 0")
                    predictions.append(0)
                    
            except Exception as e:
                if idx < 5:  # Only print first few errors
                    print(f"\nError processing sentence {idx}: {e}")
                predictions.append(0)  # Default to 0 on error
                failed_indices.append(idx)

        # Report failed classifications
        if failed_indices:
            print(f"\nWarning: {len(failed_indices)} sentences failed to classify and defaulted to 0")

        # Calculate metrics
        print("\n" + "="*60)
        print(f"RESULTS - Mode: {mode}, Model: {MODEL_NAME}")
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
        output_file = f'results-dec/predictions_gpt4o_{mode}_{temperature}_{seed}_{t}_{prompt_version}.csv'
        # output_file = f'predictions_gpt4o_{mode}_{temperature}.csv'
        df['predicted'] = predictions
        df['correct'] = [int(t == p) for t, p in zip(true_labels, predictions)]
        df.to_csv(output_file, index=False)
        print(f"\nPredictions saved to: {output_file}")

        # Save summary
        summary = {
            'mode': mode,
            'model': MODEL_NAME,
            'total_samples': len(sentences),
            'failed_classifications': len(failed_indices),
            'accuracy': accuracy,
            'precision_binary': precision,
            'recall_binary': recall,
            'f1_binary': f1,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'confusion_matrix': cm.tolist()
        }

        summary_file = f'results-dec/summary_gpt4o_{mode}_{temperature}_{seed}_{t}_{prompt_version}.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Summary saved to: {summary_file}")

        # Estimate API cost
        print(f"\n💰 Estimated API Cost:")
        avg_prompt_tokens = 150 if mode == 'z' else (200 if mode == 'd' else 300)
        avg_completion_tokens = 2
        total_tokens = len(sentences) * (avg_prompt_tokens + avg_completion_tokens)

        if MODEL_NAME == "gpt-4o":
            cost = (avg_prompt_tokens * len(sentences) / 1_000_000 * 2.50) + \
                (avg_completion_tokens * len(sentences) / 1_000_000 * 10.00)
            print(f"  GPT-4o: ~${cost:.2f}")
        elif MODEL_NAME == "gpt-4o-mini":
            cost = (avg_prompt_tokens * len(sentences) / 1_000_000 * 0.15) + \
                (avg_completion_tokens * len(sentences) / 1_000_000 * 0.60)
            print(f"  GPT-4o-mini: ~${cost:.2f}")
        print(f"  (Estimated for {len(sentences)} sentences)")
