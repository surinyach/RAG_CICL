import argparse
import pandas as pd
import pickle
from tqdm import tqdm
import json
import os
import re

from model.model_loader import ModelLoader
from model.language_model import LanguageModel

"""
This script generates questions and answers from the contents of a given pickle file.
The format of the output follows the template of the TruthfulQA dataset:
    - Question
    - Best answer
    - Correct answers
    - Incorrect answers
This program is useful to obtain questions from any arbitrary document to then be used
as knowledge base in a Contrastive In Context Learning (CICL) RAG architecture.
"""

# CONFIGURATION
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
OUTPUT_FILE = "generated_questions.pkl"
NUM_QUESTIONS_PER_ENTRY = 3

# PROMPT TEMPLATE
PROMPT_TEMPLATE = """[INST]
Generate a challenging question based on:

"{data}"

Include:
- one best answer,
- two correct paraphrases,
- two plausible wrong answers.

Answers max 15 words, no explanations or markdown.

Output ONLY this JSON format:

{{
  "question": "...",
  "best_answer": "...",
  "correct_answers": ["...", "..."],
  "incorrect_answers": ["...", "..."]
}}

Example:

{{
  "question": "What is the capital of Canada?",
  "best_answer": "Ottawa",
  "correct_answers": ["Ottawa", "The capital is Ottawa"],
  "incorrect_answers": ["Toronto", "Vancouver"]
}}[/INST]
"""

def extract_and_parse_json(raw_text: str) -> list[dict]:
    """
    Extracts all JSON objects from text and parses them.

    Args:
        raw_text (str): The raw text string, possibly containing multiple JSON objects.

    Returns:
        list[dict]: A list of parsed JSON objects as dictionaries. Empty list if parsing fails.
    """
    json_objects = []
    # Find all JSON-like structures (content between { and })
    json_pattern = r'\{(?:[^{}]|\{[^{}]*\})*\}'
    matches = re.findall(json_pattern, raw_text, re.DOTALL)
    
    for json_str in matches:
        try:
            parsed_json = json.loads(json_str)
            json_objects.append(parsed_json)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error for string: {json_str}\nError: {e}")
            continue
    
    if not json_objects:
        print(f"No valid JSON found in raw text: {raw_text}")
    
    return json_objects

def generate_qa(entry: str, language_model) -> list[dict]:
    """
    Generates questions and answers through the LLM.

    Args:
        entry (str): Data from which the questions will be generated.
        language_model: The language model instance to generate responses.

    Returns:
        list[dict]: List of JSON objects containing questions and answers.
    """
    prompt = PROMPT_TEMPLATE.format(data=entry)

    try:
        response = language_model.generate(prompt, False, 0.2, 0.1, 2, 500)
        
        qa_list = extract_and_parse_json(response)
        
        valid_qa = [
            qa for qa in qa_list
            if isinstance(qa, dict) and all(
                key in qa for key in ["question", "best_answer", "correct_answers", "incorrect_answers"]
            )
        ]
        
        return valid_qa
    
    except Exception as e:
        print(f"Error generating QA: {e}")
        if response:
            print(f"Raw content received from model: \n{response}")
        else:
            print("No raw content received from the model due to an early API error.")
        return []


def extract_items(raw_data, column_name=None):
    """
    Extracts the specified column of the input dataframe.

    Args:
        raw_data: Contents of the pickle file given as an input.
        column_name: Name of the column that is desired to extract.

    Returns:
        List of strings containing the data of the specified column.        
    """
    if isinstance(raw_data, pd.DataFrame):
        if column_name is None:
            raise ValueError("You must specify --column when using a DataFrame input.")
        if column_name not in raw_data.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame.")
        return raw_data[column_name].astype(str).tolist()

    elif isinstance(raw_data, list):
        return [str(x) for x in raw_data]
    
    else:
        raise ValueError("Unsupported pickle file format.")

def main(pickle_file, column_name):

    model_loader_generation = ModelLoader(
        MODEL_NAME,
        quant_type='4bit'
    )

    language_model = LanguageModel(
        model_loader_generation,
        True,
        ("[INST]","[/INST]")
    )

    with open(pickle_file, 'rb') as f:
        raw_data = pickle.load(f)

    items = extract_items(raw_data, column_name)

    records = []
    for entry in tqdm(items, desc="Generating questions"):
        for _ in range(NUM_QUESTIONS_PER_ENTRY):
            qa = generate_qa(entry, language_model)
            if qa:
                records.append({
                    "question": qa["question"],
                    "best_answer": qa["best_answer"],
                    "correct_answers": qa["correct_answers"],
                    "incorrect_answers": qa["incorrect_answers"]
                })
    
    df = pd.DataFrame(records)
    df.to_pickle(OUTPUT_FILE)
    print(f"Saved {len(df)} questions to {OUTPUT_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input .pkl file")
    parser.add_argument("--column", type=str, help="Column name to use if input is a DataFrame")
    args = parser.parse_args()
    main(args.input, args.column)