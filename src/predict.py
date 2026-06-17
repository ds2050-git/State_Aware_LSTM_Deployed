import torch
import torch.nn.functional as F
import numpy as np

class NextWordPredictor:
    def __init__(self,model,word_to_idx,idx_to_word,device='cpu'):
        self.model=model
        self.word_to_idx=word_to_idx
        self.idx_to_word=idx_to_word
        self.device=device
        self.model.to(device)
        self.model.eval()
        
    def predict_next_word(self,text,num_predictions=5,temperature=1.0):
        words=text.lower().split()
        if len(words)>10:
            words=words[-10:]
        
        x_seq=[]
        for w in words:
            if w in self.word_to_idx:
                x_seq.append(self.word_to_idx[w])
            else:
                x_seq.append(self.word_to_idx['<UNK>'])
        
        while len(x_seq)<10:
            x_seq.insert(0,self.word_to_idx['<PAD>'])
        
        input_tensor=torch.LongTensor([x_seq]).to(self.device)
        
        with torch.no_grad():
            hidden=self.model.init_hidden(1,self.device)
            output,_=self.model(input_tensor,hidden)
            output=output/temperature
            probabilities=F.softmax(output,dim=1)
            top_probs,top_indices=torch.topk(probabilities,num_predictions)
        
        predictions=[]
        for i in range(num_predictions):
            word_idx=top_indices[0][i].item()
            word=self.idx_to_word.get(word_idx,'<UNK>')
            prob=top_probs[0][i].item()
            predictions.append((word,prob))
        
        return predictions
    
    def generate_text(self,seed_text,num_words=10,temperature=0.8):
        current_text=seed_text.lower()
        
        for _ in range(num_words):
            predictions=self.predict_next_word(current_text,num_predictions=1,temperature=temperature)
            next_word=predictions[0][0]
            current_text+=' '+next_word
            
            if next_word=='<UNK>':
                break
        
        return current_text