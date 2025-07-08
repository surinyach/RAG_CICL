import pickle
import pandas as pd
import sys
import os

def print_pickle_contents(pickle_path, column=None, max_rows=5):
    if not os.path.exists(pickle_path):
        print(f"❌ File not found: {pickle_path}")
        return

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    print(f"\n📂 Loaded file: {pickle_path}")
    print(f"📦 Type of object: {type(data)}\n")

    if isinstance(data, pd.DataFrame):
        print(f"🧾 DataFrame shape: {data.shape}")
        print(f"📋 Available columns: {list(data.columns)}\n")

        if column:
            if column in data.columns:
                print(f"📄 Full contents of column '{column}':\n")
                for i, val in enumerate(data[column]):
                    print(f"[{i}]: {val}\n")
                    if max_rows and i + 1 >= max_rows:
                        break
            else:
                print(f"⚠️ Column '{column}' not found in DataFrame.")
        else:
            print("📑 Preview of full DataFrame:")
            print(data.head(max_rows))
    elif isinstance(data, list):
        print(f"📜 List of length {len(data)}:")
        for i, item in enumerate(data[:max_rows]):
            print(f"[{i}]: {item}\n")
    elif isinstance(data, dict):
        print(f"📘 Dictionary with {len(data)} keys:")
        for i, (k, v) in enumerate(list(data.items())[:max_rows]):
            print(f"{k}: {v}\n")
    else:
        print(f"🤷 Unhandled type. Raw output:")
        print(data)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print contents of a pickle file, with optional full column view for DataFrames.")
    parser.add_argument("pickle_path", help="Path to the .pkl file")
    parser.add_argument("--column", type=str, help="Name of the column to display fully (if DataFrame)")
    parser.add_argument("--max_rows", type=int, default=5, help="Maximum number of rows to display")

    args = parser.parse_args()

    print_pickle_contents(args.pickle_path, column=args.column, max_rows=args.max_rows)
