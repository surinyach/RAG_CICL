import mauve
import json
import os
from urllib.request import urlretrieve

"""
This script proves the correct behaviour of the MAUVE metric in a concrete environment.
The test is done comparing the score between human text (WebText) and machine-generated text (GPT-2 outputs).
"""

# Function to download and load sample data
def load_gpt2_dataset(file_path, num_examples=100):
    """
    Load text samples from a JSONL file.
    Args:
        file_path: Path to JSONL file
        num_examples: Number of samples to load
    Returns:
        List of text strings
    """
    texts = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_examples:
                break
            texts.append(json.loads(line)['text'])
    return texts

# Download sample data if not already present
data_dir = './data'
os.makedirs(data_dir, exist_ok=True)

webtext_url = 'https://raw.githubusercontent.com/krishnap25/mauve-experiments/main/data/webtext.valid.jsonl'
gpt2_url = 'https://raw.githubusercontent.com/krishnap25/mauve-experiments/main/data/webtext-xl-1542M.valid.jsonl'
webtext_path = os.path.join(data_dir, 'webtext.valid.jsonl')
gpt2_path = os.path.join(data_dir, 'webtext-xl-1542M.valid.jsonl')

if not os.path.exists(webtext_path):
    print("Downloading WebText sample data...")
    urlretrieve(webtext_url, webtext_path)
if not os.path.exists(gpt2_path):
    print("Downloading GPT-2 sample data...")
    urlretrieve(gpt2_url, gpt2_path)

# Load human (WebText) and machine (GPT-2) texts
num_samples = 1000  # Use 1000 samples for better statistical stability (MAUVE paper uses 5000)
p_text = load_gpt2_dataset(webtext_path, num_examples=num_samples)  # Human text
q_text = load_gpt2_dataset(gpt2_path, num_examples=num_samples)     # GPT-2 generated text

# Compute MAUVE score
print("Computing MAUVE score...")
out = mauve.compute_mauve(
    p_text=p_text,           # Human text
    q_text=q_text,           # Machine text
    device_id=0,             # GPU ID (set to -1 for CPU)
    max_text_length=256,     # Truncate texts to 256 tokens
    featurize_model_name='gpt2-large',  # Use GPT-2 large for featurization
    verbose=True
)

# Print results
print(f"MAUVE score: {out.mauve:.4f}")

# Validate against benchmark
expected_mauve = 0.9
tolerance = 0.1  # Allow some variation due to sampling and implementation
if abs(out.mauve - expected_mauve) <= tolerance:
    print("MAUVE score aligns with benchmark (~0.9). Implementation appears valid.")
else:
    print(f"MAUVE score deviates from benchmark (~0.9). Got {out.mauve:.4f}. Scrutinize implementation or data.")

# Optional: Cross-validation by splitting data
from sklearn.model_selection import KFold
import numpy as np

# Perform 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mauve_scores = []

print("\nPerforming 5-fold cross-validation...")
for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(len(p_text)))):
    p_text_fold = [p_text[i] for i in test_idx]
    q_text_fold = [q_text[i] for i in test_idx]
    
    fold_out = mauve.compute_mauve(
        p_text=p_text_fold,
        q_text=q_text_fold,
        device_id=0,
        max_text_length=256,
        featurize_model_name='gpt2-large',
        verbose=False
    )
    mauve_scores.append(fold_out.mauve)
    print(f"Fold {fold+1} MAUVE score: {fold_out.mauve:.4f}")

# Summarize cross-validation results
mean_mauve = np.mean(mauve_scores)
std_mauve = np.std(mauve_scores)
print(f"\nCross-validation results: Mean MAUVE = {mean_mauve:.4f}, Std = {std_mauve:.4f}")

if abs(mean_mauve - expected_mauve) <= tolerance:
    print("Cross-validated MAUVE score aligns with benchmark (~0.9). Implementation is likely correct.")
else:
    print(f"Cross-validated MAUVE score deviates from benchmark (~0.9). Got {mean_mauve:.4f}. Check data or implementation.")