import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_modified_model(model,train_loader,vocab_size,device,num_epochs=50,lr=0.001):
    criterion=nn.CrossEntropyLoss(ignore_index=0,label_smoothing=0.1)
    optimizer=optim.AdamW(model.parameters(),lr=lr,weight_decay=0.01)
    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=num_epochs)
    
    model.to(device)
    train_losses=[]
    
    for epoch in range(num_epochs):
        model.train()
        total_loss=0
        
        progress_bar=tqdm(train_loader,desc=f'Modified Epoch {epoch+1}/{num_epochs}')
        
        for batch_x,batch_y in progress_bar:
            batch_x,batch_y=batch_x.to(device),batch_y.to(device)
            hidden=model.init_hidden(batch_x.size(0),device)
            optimizer.zero_grad()
            output,hidden=model(batch_x,hidden)
            loss=criterion(output,batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1)
            optimizer.step()
            total_loss+=loss.item()
            progress_bar.set_postfix({'loss':f'{loss.item():.4f}'})
        
        avg_loss=total_loss/len(train_loader)
        train_losses.append(avg_loss)
        scheduler.step()
        
        print(f'Modified Epoch {epoch+1}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        if (epoch+1)%10==0:
            torch.save(model.state_dict(),f'models/modified_lstm_epoch_{epoch+1}.pth')
    
    return train_losses