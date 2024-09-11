import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from models.gender_age_model import GenderAgeModel
from preprocessing.data_loader import DialogueDataset
from tqdm import tqdm 
from utils.checkpoint import save_checkpoint
from torch.utils.tensorboard import SummaryWriter

def train_model(model, train_loader, tokenizer, criterion, optimizer, device, epochs=10, accumulation_steps=4, start_epoch=0):
    writer = SummaryWriter(log_dir='./runs')
    model.train()
    for epoch in range(start_epoch, epochs):
        running_loss = 0.0
        total_acc_age = 0.0
        total_acc_gender = 0.0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            gender_labels, age_labels = labels

            # Tokenize the inputs
            inputs = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True, max_length=32).to(device)
            
            gender_labels = gender_labels.to(device)
            age_labels = age_labels.to(device)

            # forward
            age_logits, gender_logits = model(inputs['input_ids'], inputs['attention_mask'])
            
            # 손실 계산
            age_loss = criterion(age_logits, age_labels) / accumulation_steps
            gender_loss = criterion(gender_logits, gender_labels) / accumulation_steps
            loss = age_loss + gender_loss
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            acc_age = (age_logits.argmax(1) == age_labels).float().mean().item()
            acc_gender = (gender_logits.argmax(1) == gender_labels).float().mean().item()

            running_loss += loss.item() * accumulation_steps
            total_acc_age += acc_age
            total_acc_gender += acc_gender
            progress_bar.set_postfix(loss=running_loss / len(progress_bar), acc1=acc_age, acc2=acc_gender)
            
            # tensorboard --logdir ./runs --bind_all --reload_interval 5
            writer.add_scalar('Loss/train', loss.item(), epoch * len(progress_bar) + batch_idx)
            writer.add_scalar('Accuracy/Age_train', acc_age, epoch * len(progress_bar) + batch_idx)
            writer.add_scalar('Accuracy/Gender_train', acc_gender, epoch * len(progress_bar) + batch_idx)
            writer.flush()
        
            
        save_checkpoint(model, optimizer, epoch + 1)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(progress_bar)}, Age Acc : {acc_age}, Gender Acc : {acc_gender}')
