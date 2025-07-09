import argparse
import pandas as pd
import pickle
import openai
from tqdm import tqdm
import json

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
openai.api_key = ""
MODEL_NAME = "gpt-4"
OUTPUT_FILE = "generated_questions.pkl"
NUM_QUESTIONS_PER_ENTRY = 5

# PROMPT TEMPLATE
PROMPT_TEMPLATE = """
You are a question generation system. Given the following input:

"{data}"

Generate a challenging, fact-based question suitable for a QA dataset. Then provide:
- one best answer,
- 2 correct paraphrased answers,
- 2 incorrect but plausible-sounding answers.

Format your output as JSON with the following fields:
{
  "question": "...",
  "best_answer": "...",
  "correct_answers": ["...", "..."],
  "incorrect_answers": ["...", "..."]
}
"""

def generate_qa(entry: str) -> dict:
    """
    Generates the question and answers through the LLM.

    Args:
        entry(str): Data from where the questions will be generated.
    
    Returns:
        data(json): Json with the responses of the LLM, containing the q&a. 
    """

    prompt = PROMPT_TEMPLATE.format(data = entry)
    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature = 0.7
    )

    try:
        content = response.choices[0].message["content"]
        data = json.loads(content)
        return data
    except Exception as e:
        print("Failed to parse LLM response: ", e)
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
    with open(pickle_file, 'rb') as f:
        raw_data = pickle.load(f)

    items = extract_items(raw_data, column_name)

    records = []
    for entry in tqdm(items, desc="Generating questions"):
        for _ in range(NUM_QUESTIONS_PER_ENTRY):
            qa = generate_qa(entry)
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
