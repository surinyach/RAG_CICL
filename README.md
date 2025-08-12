# Generalizing RAG Contrastive In-Context Learning

This repository contains the framework used to evaluate generalization mechanisms for Contrastive In-Context Learning, a RAG enhancement technique described in the paper [Enhancing Retrieval-Augmented Generation: A Study of Best Practices](https://arxiv.org/pdf/2501.07391).

## Overview

This project presents two novel generalization proposals for the Contrastive In-Context Learning mechanism:
- **Document-Based Content Extraction**
- **Automatic Q&A Generation**

This work was developed as part of a Master's seminar in Informatics at the Technical University of Munich (TUM).

## Framework Description

This codebase is an adapted version of the original framework from the [RAG Best Practices repository](https://github.com/ali-bahrainian/RAG_best_practices). We have streamlined the implementation by removing non-relevant functions and modifying the configuration file to support our new mechanisms.

All configuration changes can be made in the `config.py` file.

## Key Components

### Question Generator
The `question_generator.py` file contains the core logic for generating questions and answers from generic data sources, which is essential for the Automatic Q&A Generation method. It utilizes the same Mixtral 7B parameter model used for response generation in the general RAG system.

**Usage:**
```bash
python question_generator.py --input "path/to/pickle/file/containing/document/data"
```

## Installation

Follow these steps to set up the environment:

1. **Clone the Mixtral-offloading repository:**
   ```bash
   git clone https://github.com/dvmazur/mixtral-offloading.git
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download knowledge sources:**
   - Download the necessary knowledge base files from the provided [Google Drive link](https://drive.google.com/drive/folders/1_-2PHI0-Wz1VjnW5Yvy5Ne9C7mMWk1nf?usp=drive_link)
   - Extract the downloaded files into the `resources/` directory
   - Generate the `generated_questions.pkl` file by running the `question_generator.py` script. This file will contain questions and answers formatted according to the TruthfulQA dataset structure

## Project Structure

```
RAG_best_practices/
│
├── mixtral-offloading/          # Mixtral model offloading library
│
├── model/                       # Core RAG implementation
│   ├── index_builder.py         # Document index builder
│   ├── language_model.py        # LLM text generation setup
│   ├── model_loader.py          # Mixtral LLM loader
│   ├── rag.py                   # Main RAG pipeline
│   ├── retriever.py             # Document retrieval system
│   ├── config.py                # Configuration setup
│   ├── evaluation.py            # Full RAG pipeline execution
│   └── requirements.txt         # Python dependencies
│
├── resources/                   # Knowledge base
│   ├── articles_l3.pkl          # Knowledge base file (level 3)
│   └── generated_questions.pkl  # Generated Q&A pairs
│
├── utils/                       # Utility scripts (see Utils section)
│
└── README.md
```

## Utils

The `utils/` directory contains several helpful scripts:

### MAUVE Metric Testing
- **`download_gpt2_dataset.py`**: Downloads the dataset required for MAUVE metric testing
- **`mauve_checker.py`**: Tests the MAUVE metric functionality

These utilities are necessary because MAUVE metric results can vary depending on hardware architecture. It's recommended to test the system's capability to process this metric without execution issues.

### Result Analysis
- **`print_pickle.py`**: Displays pickle file contents in the terminal, particularly useful for viewing evaluation results

**Usage:**
```bash
python print_pickle.py [-h] [--column COLUMN] [--max_rows MAX_ROWS] pickle_path
```

**Options:**
- `--column`: Print a specific column
- `--max_rows`: Limit the number of results displayed
- `pickle_path`: Path to the pickle file to analyze

## Running the RAG System

Execute the following command to run the complete evaluation:

```bash
python evaluation.py
```

This command will:
- Run all Document-Based Content Extraction tests with various K values
- Test the Automatic Q&A Generation method using automatically generated questions and answers

### Command Line Options

- `--dataset`: Choose dataset (`truthfulqa` or `mmlu`)
- `--outputdir`: Specify output directory for results
- `--seed`: Set random seed for reproducible execution

**Example:**
```bash
python evaluation.py --dataset truthfulqa --outputdir ./results --seed 42
```

## Research Context

This implementation explores generalization techniques for Contrastive In-Context Learning in RAG systems, contributing to the broader research on improving retrieval-augmented generation performance through enhanced context utilization.

## License

This project builds upon the original RAG Best Practices framework. Please refer to the original repository for licensing information.