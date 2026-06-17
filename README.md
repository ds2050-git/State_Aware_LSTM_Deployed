# Next Word Prediction with LSTM Models

## Overview

This project implements and compares **two LSTM architectures** for next-word prediction using *Alice's Adventures in Wonderland*. 

It features:
- A **Base LSTM** using PyTorch's built-in `nn.LSTM`
- A **Modified LSTM** with custom implementation including **working memory connections** (additional tanh connections from cell state to gates) for better context modeling.

The project demonstrates full training pipeline, evaluation, text generation, and detailed performance comparison.

Working memory connections refer to additional learnable transformations from the cell state to gate computations, allowing gates to incorporate richer internal memory signals.

## Features

- Text preprocessing and vocabulary building
- Sequence generation for language modeling
- Base LSTM vs Modified LSTM comparison
- Training with learning rate scheduling and gradient clipping
- Next-word prediction with temperature sampling
- Text generation
- Comprehensive evaluation on 1000+ test sentences
- Visualization of training loss and model comparison

## Project Structure

```
next-word-lstm/
├── data/
│   ├── raw/alice.txt                 # Original book text
│   ├── processed/tokenizer.pkl       # Saved vocabulary
│   └── test/alice_test.txt           # Test sentences
├── src/
│   ├── base_lstm.py                  # Standard LSTM model
│   ├── modified_lstm.py              # Custom LSTM with working memory
│   ├── data_preprocessing.py         # Text processing
│   ├── predict.py                    # Prediction & generation
│   ├── train_base.py                 # Training logic (Base)
│   └── train_modified.py             # Training logic (Modified)
├── utils/
│   └── helpers.py                    # Dataset, DataLoader, metrics
├── models/                           # Trained models & plots
├── config.py                         # Hyperparameters
├── main.py                           # Full training script
├── val.py                            # Evaluation & comparison
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
# 1. Create virtual environment
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

```

## Usage

### 1. Train Both Models

```bash
python main.py
```

This will:
- Preprocess the data
- Train Base LSTM and Modified LSTM
- Save models and generate loss plots

### 2. Run Full Evaluation

```bash
python val.py
```

Generates detailed comparison (accuracy, speed, win rates, graphs) and saves results in `models/`.

## Model Architecture

### Base LSTM

- Standard `nn.LSTM` from PyTorch
- Embedding layer → LSTM → Dropout → Linear output

### Modified LSTM

- Fully custom LSTM cell
- Enhanced input, forget, cell, and output gates
- Working memory connections: Additional tanh(W_ic · c) and similar terms for better gate regulation

## Results

After running `val.py`, you will find:

- `models/full_evaluation_results.png` — Comprehensive comparison dashboard
- `models/model_comparison.png` — Loss and perplexity curves
- `models/full_evaluation_results.txt` — Summary metrics

The project evaluates whether the Modified LSTM can improve coherence and prediction accuracy through enhanced memory connections.