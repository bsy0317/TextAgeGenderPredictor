import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from models.gender_age_model import GenderAgeModel
from preprocessing.data_loader import DialogueDataset
from tqdm import tqdm 
from utils.checkpoint import save_checkpoint

def train_model(model, train_loader, tokenizer, criterion, optimizer, device, epochs=10, start_epoch=0):
    model.train()
    for epoch in range(start_epoch, epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            gender_labels, age_labels = labels
            
            # 배치 크기 출력
            # print(f"Batch {batch_idx+1}:")
            # print(f"  Inputs batch size: {len(inputs)}")
            # print(f"  Gender labels batch size: {len(gender_labels)}")
            # print(f"  Age labels batch size: {len(age_labels)}")
            
            # 데이터 크기 확인
            #print(f"Batch size - Inputs: {len(inputs)}, Gender Labels: {len(gender_labels)}, Age Labels: {len(age_labels)}")

            # Tokenize the inputs
            inputs = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
            
            gender_labels = gender_labels.to(device)
            age_labels = age_labels.to(device)

            # forward
            optimizer.zero_grad()
            age_logits, gender_logits = model(inputs['input_ids'], inputs['attention_mask'])
            
            # 손실 계산
            age_loss = criterion(age_logits, age_labels)
            gender_loss = criterion(gender_logits, gender_labels)
            loss = age_loss + gender_loss
            loss.backward()
            optimizer.step()
            
            acc_age = (age_logits.argmax(1) == age_labels).float().mean()
            acc_gender = (gender_logits.argmax(1) == gender_labels).float().mean()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / len(progress_bar), acc1=acc_age, acc2=acc_gender)
            
        save_checkpoint(model, optimizer, epoch + 1)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(progress_bar)}, Age Acc : {acc_age}, Gender Accuracy : {acc_gender}')
