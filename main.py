import torch
import numpy as np
from config import *
from src.data_preprocessing import DataPreprocessor
from src.base_lstm import BaseLSTM
from src.modified_lstm import ModifiedLSTM
from src.train_base import train_base_model,plot_training_history
from src.train_modified import train_modified_model
from src.predict import NextWordPredictor
from utils.helpers import create_dataloaders,compare_models
import os

def main():
    os.makedirs('models',exist_ok=True)
    os.makedirs('data/processed',exist_ok=True)
    
    print("="*50)
    print("Next Word Prediction with LSTM Models")
    print("="*50)
    
    preprocessor=DataPreprocessor(vocab_size=VOCAB_SIZE,seq_length=SEQ_LENGTH)
    
    if not os.path.exists(DATA_PATH):
        print("Downloading sample text data...")
        import urllib.request
        url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url,DATA_PATH)
    
    text=preprocessor.load_text(DATA_PATH)
    X,y,vocab_size=preprocessor.create_sequences(text)
    preprocessor.save_tokenizer(PROCESSED_DATA_PATH+'tokenizer.pkl')
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Number of sequences: {len(X)}")
    
    train_loader=create_dataloaders(X,y,batch_size=BATCH_SIZE)
    
    print("\n2. Training Base LSTM Model...")
    base_model=BaseLSTM(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    
    base_losses=train_base_model(
        model=base_model,
        train_loader=train_loader,
        vocab_size=vocab_size,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE
    )
    
    torch.save(base_model.state_dict(),'models/base_lstm_model.pth')
    plot_training_history(base_losses,"Base LSTM Training Loss")
    
    print("\n3. Training Modified LSTM Model...")
    modified_model=ModifiedLSTM(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    
    modified_losses=train_modified_model(
        model=modified_model,
        train_loader=train_loader,
        vocab_size=vocab_size,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE
    )
    
    torch.save(modified_model.state_dict(),'models/modified_lstm_model.pth')
    plot_training_history(modified_losses,"Modified LSTM Training Loss")
    
    print("\n4. Comparing Model Performance...")
    compare_models(base_losses,modified_losses)
    
    print("\n5. Prediction Demo")
    print("-"*40)
    
    preprocessor.load_tokenizer(PROCESSED_DATA_PATH+'tokenizer.pkl')
    
    base_predictor=NextWordPredictor(base_model,preprocessor.word_to_idx,preprocessor.idx_to_word,DEVICE)
    modified_predictor=NextWordPredictor(modified_model,preprocessor.word_to_idx,preprocessor.idx_to_word,DEVICE)
    
    test_sentences=[
        "to be or not to",
        "the sun rises in the",
        "in the beginning was the",
        "all that glitters is not"
    ]
    
    for sentence in test_sentences:
        print(f"\nInput: '{sentence}'")
        
        print("Base LSTM Predictions:")
        predictions=base_predictor.predict_next_word(sentence,num_predictions=5)
        for i,(word,prob) in enumerate(predictions,1):
            print(f"  {i}. {word} ({prob:.3f})")
        
        print("Modified LSTM Predictions:")
        predictions=modified_predictor.predict_next_word(sentence,num_predictions=5)
        for i,(word,prob) in enumerate(predictions,1):
            print(f"  {i}. {word} ({prob:.3f})")
        
        print("\nText Generation (Base):")
        print(f"  {base_predictor.generate_text(sentence,num_words=10)}")
        
        print("Text Generation (Modified):")
        print(f"  {modified_predictor.generate_text(sentence,num_words=10)}")
    
    print("\n"+"="*50)
    print("Training Complete! Models saved in 'models/' directory")
    print("="*50)

if __name__=="__main__":
    main()