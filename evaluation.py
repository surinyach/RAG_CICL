import gc
import json
import os
import argparse
from datetime import datetime
import random

import pandas as pd
import numpy as np
import torch
from datasets import load_dataset

from model.index_builder import IndexBuilder
from model.language_model import LanguageModel
from model.model_loader import ModelLoader
from model.rag import RAG
from model.retriever import Retriever

from config import cicl_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the RAG model")
    parser.add_argument('--dataset', default='truthfulqa', type=str, help='Dataset to evaluate on')
    parser.add_argument('--output-dir', default='outputs', type=str, help='Output directory')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def initialize_index_builder(knowledge_base, config):
    index_builder = IndexBuilder(
        knowledge_base,
        config['embedding_model_name'],
        **config['index_builder']
    )
    return index_builder.initialize_components()


def initialize_rag(knowledge_base, config, model_loader_generation, index_pre, same_index, first_run):
    build_index = not same_index or first_run

    if build_index:
        index, doc_info = initialize_index_builder(knowledge_base, config)
    else:
        index, doc_info = index_pre[0], index_pre[1]

    retriever = Retriever(index, doc_info, config['embedding_model_name'])
    language_model = LanguageModel(
        model_loader_generation,
        config['is_chat_model'],
        config['instruct_tokens']
    )

    if not same_index:
        del index, doc_info
        gc.collect()
        index_pre = None
    else:
        index_pre = (index, doc_info)

    return RAG(retriever, language_model, **config['ralm']), index_pre


def mean_metrics_item(evaluation):
    metrics = ['r1f1', 'r2f1', 'rLf1', 'similarity']
    return {metric: float(evaluation[metric].mean()) for metric in metrics}


if __name__ == "__main__":
    args = parse_args()
    set_random_seed(args.seed)

    # Load and preprocess dataset
    if args.dataset == 'truthfulqa':
        truthful_qa = load_dataset("truthful_qa", "generation", split='validation').to_pandas()
        test_data = truthful_qa[['question', 'best_answer', 'correct_answers', 'incorrect_answers']]
        test_data.loc[:, 'correct_answers'] = test_data['correct_answers'].apply(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else [x]
        )
        test_data.loc[:, 'correct_answers'] = test_data['correct_answers'].apply(lambda x: [i for i in x if i])
        test_data.loc[:, 'incorrect_answers'] = test_data['incorrect_answers'].apply(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else [x]
        )
        test_data.loc[:, 'incorrect_answers'] = test_data['incorrect_answers'].apply(lambda x: [i for i in x if i])
        test_data.loc[:, 'best_answer'] = test_data['best_answer'].apply(lambda x: [x] if x else [])
        test_data = test_data[
            (test_data['correct_answers'].apply(len) > 1) &
            (test_data['incorrect_answers'].apply(len) > 1)
        ].reset_index(drop=True)

    elif args.dataset == 'mmlu':
        mmlu = load_dataset("cais/mmlu", "all")
        test_data = mmlu['test'].to_pandas().groupby('subject').head(32).drop(columns='subject').reset_index(drop=True)
        test_data['aswer'] = test_data['answer'].astype(int)
        test_data.loc[:, 'choices'] = test_data['choices'].apply(lambda x: x.tolist())

        def extract_answers(row):
            correct_answers = [choice for i, choice in enumerate(row['choices']) if i == row['answer']]
            assert len(correct_answers) > 0
            incorrect_answers = [choice for i, choice in enumerate(row['choices']) if i != row['answer']]
            return pd.Series([correct_answers, incorrect_answers], index=['correct_answers', 'incorrect_answers'])

        test_data[['correct_answers', 'incorrect_answers']] = test_data.apply(extract_answers, axis=1)
        test_data['best_answer'] = [[] for _ in range(len(test_data))]
        test_data = test_data[['question', 'best_answer', 'correct_answers', 'incorrect_answers']].reset_index(drop=True)
        print(f"Loaded {len(test_data)} questions from MMLU dataset")

    # Load knowledge base
    knowledge_base = pd.read_pickle('resources/articles_l3.pkl')
    all_results = {}

    # Output setup
    time = datetime.now().strftime("%m-%d_%H-%M")
    results_dir = f'{args.output_dir}/{args.dataset}/run_{time}'
    os.makedirs(results_dir, exist_ok=True)

    # Setup indexing configuration
    index_configs = [cicl_config['index_builder']]
    same_index = True
    index_pre = None
    first_run = True

    evaluations = {}
    name = "CICL"

    # Evaluation loop
    model_loader_generation = ModelLoader(
        cicl_config['generation_model_name'],
        'causal',
        quant_type='4bit'
    )

    ralm, index_pre = initialize_rag(
        knowledge_base,
        cicl_config,
        model_loader_generation,
        index_pre,
        same_index,
        first_run
    )

    print(f"Evaluating model: {name}")
    evaluations[name], mauve_score = ralm.evaluate(test_data)

    del ralm
    del model_loader_generation
    gc.collect()
    torch.cuda.empty_cache()
    first_run = False

    # Save evaluation results
    evaluations[name].to_pickle(os.path.join(results_dir, f'evaluation_{name}.pkl'))

    with open(os.path.join(results_dir, f'config_{name}.json'), 'w') as f:
        json.dump(cicl_config, f, indent=4)

    results = mean_metrics_item(evaluations[name])
    results['mauve'] = mauve_score

    with open(f"{results_dir}/eval_results_{name}.json", "w") as outfile:
        json.dump(results, outfile)

    all_results[name] = results

    with open(f"{results_dir}/eval_results_all.json", "w") as outfile:
        json.dump(all_results, outfile)
