# Next Word Prediction with LSTM Models

## Overview

This project implements and compares **two LSTM architectures** for next-word prediction using *Alice's Adventures in Wonderland*. 

It features:
- A **Base LSTM** using PyTorch's built-in `nn.LSTM`
- A **Modified LSTM** with custom implementation including **working memory connections** (additional tanh connections from cell state to gates) for better context modeling.

The project demonstrates full training pipeline, evaluation, text generation, and detailed performance comparison.

Working memory connections refer to additional learnable transformations from the cell state to gate computations, allowing gates to incorporate richer internal memory signals.