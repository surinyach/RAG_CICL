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
PROMPT_TEMPLATE = """
You are a question generation system. Given the following input:

"{data}"

Generate a challenging, fact-based question suitable for a QA dataset. Then provide, concisely:
- one best answer,
- 2 correct paraphrased answers,
- 2 incorrect but plausible-sounding answers.

Keep all answers under 15 words. Avoid explanations or markdown formatting.

Follow this example format from the TruthfulQA dataset:

Example:
{{
  "question": "What is the capital of Canada?",
  "best_answer": "Ottawa",
  "correct_answers": ["Ottawa", "The capital is Ottawa"],
  "incorrect_answers": ["Toronto", "Vancouver"]
}}

Now generate one based on the input:

"{data}"

Respond in the same JSON format.
"""


def extract_and_parse_json(raw_text: str) -> dict | None:
    """
    Extracts a JSON string from text (potentially wrapped in markdown) and parses it.

    Args:
        raw_text (str): The raw text string, possibly containing a JSON markdown block.

    Returns:
        dict | None: The parsed JSON as a dictionary, or None if parsing fails.
    """

    # Try to find the first JSON object in the text
    try:
        start = raw_text.index('{')
        end = raw_text.rindex('}') + 1
        json_str = raw_text[start:end]
        return json.loads(json_str)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"JSON parsing error: {e}")
        print(f"Raw text: {raw_text}")
        return None

def generate_qa(entry: str, language_model) -> dict:
    """
    Generates the question and answers through the LLM.

    Args:
        entry(str): Data from where the questions will be generated.
    
    Returns:
        data(json): Json with the responses of the LLM, containing the q&a. 
    """

    prompt = PROMPT_TEMPLATE.format(data = entry)

    try:
        response = language_model.generate(prompt, False, 0.2, 0.1, 2, 500)

        # Remove the prompt from the start of the response
        if response.startswith(prompt):
            json_text = response[len(prompt):].strip()
        else:
            json_text = response.strip()

        # Now parse the cleaned JSON text
        data = extract_and_parse_json(json_text)
        return data
    
    except Exception as e:
        if response:
            print(f"Raw content received from model (before JSON parsing attempt): \n{response}")
        else:
            print("No raw content received from the model due to an early API error.")
        return None


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