import torch
import torch.nn as nn

class BaseLSTM(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,num_layers,dropout):
        super(BaseLSTM,self).__init__()
        
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        
        self.embedding=nn.Embedding(vocab_size,embedding_dim,padding_idx=0)
        self.lstm=nn.LSTM(embedding_dim,hidden_dim,num_layers,batch_first=True,dropout=dropout)
        self.fc=nn.Linear(hidden_dim,vocab_size)
        self.dropout=nn.Dropout(dropout)
        
    def forward(self,x,hidden):
        x=self.embedding(x)
        x=self.dropout(x)
        x,hidden=self.lstm(x,hidden)
        x=x[:,-1,:]
        x=self.fc(x)
        return x,hidden
    
    def init_hidden(self,batch_size,device):
        h=torch.zeros(self.num_layers,batch_size,self.hidden_dim).to(device)
        c=torch.zeros(self.num_layers,batch_size,self.hidden_dim).to(device)
        return (h,c)