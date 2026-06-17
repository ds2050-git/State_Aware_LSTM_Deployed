import torch
import torch.nn as nn
import torch.nn.functional as F

class ModifiedLSTM(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,num_layers,dropout):
        super(ModifiedLSTM,self).__init__()
        
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        
        self.embedding=nn.Embedding(vocab_size,embedding_dim,padding_idx=0)
        
        self.W_ix=nn.Linear(embedding_dim,hidden_dim)
        self.W_ih=nn.Linear(hidden_dim,hidden_dim)
        self.W_ic=nn.Linear(hidden_dim,hidden_dim)
        self.b_i=nn.Parameter(torch.zeros(hidden_dim))
        
        self.W_fx=nn.Linear(embedding_dim,hidden_dim)
        self.W_fh=nn.Linear(hidden_dim,hidden_dim)
        self.W_fc=nn.Linear(hidden_dim,hidden_dim)
        self.b_f=nn.Parameter(torch.zeros(hidden_dim))
        
        self.W_cx=nn.Linear(embedding_dim,hidden_dim)
        self.W_ch=nn.Linear(hidden_dim,hidden_dim)
        self.b_c=nn.Parameter(torch.zeros(hidden_dim))
        
        self.W_ox=nn.Linear(embedding_dim,hidden_dim)
        self.W_oh=nn.Linear(hidden_dim,hidden_dim)
        self.W_oc=nn.Linear(hidden_dim,hidden_dim)
        self.b_o=nn.Parameter(torch.zeros(hidden_dim))
        
        self.fc=nn.Linear(hidden_dim,vocab_size)
        self.dropout=nn.Dropout(dropout)
        
    def forward(self,x,hidden):
        batch_size,seq_len=x.shape
        x=self.embedding(x)
        
        h,c=hidden
        
        outputs=[]
        for t in range(seq_len):
            xt=x[:,t,:]
            
            i_t=torch.sigmoid(
                self.W_ix(xt)+self.W_ih(h[-1])+torch.tanh(self.W_ic(c[-1]))+self.b_i
            )
            
            f_t=torch.sigmoid(
                self.W_fx(xt)+self.W_fh(h[-1])+torch.tanh(self.W_fc(c[-1]))+self.b_f
            )
            
            c_tilde=torch.tanh(
                self.W_cx(xt)+self.W_ch(h[-1])+self.b_c
            )
            
            c_t=f_t*c[-1]+i_t*c_tilde
            
            o_t=torch.sigmoid(
                self.W_ox(xt)+self.W_oh(h[-1])+torch.tanh(self.W_oc(c_t))+self.b_o
            )
            
            h_t=o_t*torch.tanh(c_t)
            
            c=torch.cat([c[1:],c_t.unsqueeze(0)],dim=0)
            h=torch.cat([h[1:],h_t.unsqueeze(0)],dim=0)
            outputs.append(h_t)
        
        last_output=outputs[-1]
        output=self.fc(self.dropout(last_output))
        
        return output,(h,c)
    
    def init_hidden(self,batch_size,device):
        h=torch.zeros(self.num_layers,batch_size,self.hidden_dim).to(device)
        c=torch.zeros(self.num_layers,batch_size,self.hidden_dim).to(device)
        return (h,c)