import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader

class TextDataset(Dataset):
    def __init__(self,X,y):
        self.X=torch.LongTensor(X)
        self.y=torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        return self.X[idx],self.y[idx]

def create_dataloaders(X,y,batch_size=32):
    dataset=TextDataset(X,y)
    return DataLoader(dataset,batch_size=batch_size,shuffle=True)

def calculate_perplexity(loss):
    return np.exp(loss)

def compare_models(base_losses,modified_losses):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12,5))
    
    plt.subplot(1,2,1)
    plt.plot(base_losses,label='Base LSTM',linewidth=2)
    plt.plot(modified_losses,label='Modified LSTM',linewidth=2)
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1,2,2)
    base_perp=[calculate_perplexity(l)for l in base_losses]
    mod_perp=[calculate_perplexity(l)for l in modified_losses]
    plt.plot(base_perp,label='Base LSTM',linewidth=2)
    plt.plot(mod_perp,label='Modified LSTM',linewidth=2)
    plt.title('Perplexity Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/model_comparison.png')
    plt.show()