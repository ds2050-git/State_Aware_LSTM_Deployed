import numpy as np
import pickle
from collections import defaultdict
import random

class DataPreprocessor:
    def __init__(self,vocab_size=10000,seq_length=15):
        self.vocab_size=vocab_size
        self.seq_length=seq_length
        self.word_to_idx={}
        self.idx_to_word={}
        self.vocab=None
        
    def load_text(self,filepath):
        with open(filepath,'r',encoding='utf-8')as f:
            text=f.read()
        return text.lower()
    
    def create_sequences(self,text):
        words=text.split()
        
        word_freq=defaultdict(int)
        for w in words:
            word_freq[w]+=1
        
        sorted_words=sorted(word_freq.items(),key=lambda x:x[1],reverse=True)
        self.vocab=['<PAD>','<UNK>']+[w for w,f in sorted_words[:self.vocab_size-2]]
        
        self.word_to_idx={w:i for i,w in enumerate(self.vocab)}
        self.idx_to_word={i:w for w,i in self.word_to_idx.items()}
        
        total_words=len(self.vocab)
        
        X=[]
        y=[]
        
        for i in range(self.seq_length,len(words)):
            seq=words[i-self.seq_length:i+1]
            
            x_seq=[]
            for w in seq[:-1]:
                if w in self.word_to_idx:
                    x_seq.append(self.word_to_idx[w])
                else:
                    x_seq.append(self.word_to_idx['<UNK>'])
            
            y_word=seq[-1]
            if y_word in self.word_to_idx:
                y_idx=self.word_to_idx[y_word]
            else:
                y_idx=self.word_to_idx['<UNK>']
            
            X.append(x_seq)
            y.append(y_idx)
        
        X=np.array(X,dtype=np.int32)
        y=np.array(y,dtype=np.int32)
        
        return X,y,total_words
    
    def save_tokenizer(self,path):
        with open(path,'wb')as f:
            pickle.dump({
                'word_to_idx':self.word_to_idx,
                'idx_to_word':self.idx_to_word,
                'vocab':self.vocab
            },f)
    
    def load_tokenizer(self,path):
        with open(path,'rb')as f:
            data=pickle.load(f)
            self.word_to_idx=data['word_to_idx']
            self.idx_to_word=data['idx_to_word']
            self.vocab=data['vocab']