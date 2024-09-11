import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from models.gender_age_model import GenderAgeModel
from preprocessing.data_loader import DialogueDataset
from tqdm import tqdm

def validate_model(model, validation_loader, tokenizer, criterion, device):
    model.eval()
    total_loss = 0.0
    progress_bar = tqdm(validation_loader, desc="Validating", leave=False)
    with torch.no_grad():
        for batch in progress_bar:
            inputs, labels = batch
            inputs = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True, max_length=32).to(device)
            age_labels = torch.tensor([label[1] for label in labels]).to(device)
            gender_labels = torch.tensor([label[0] for label in labels]).to(device)

            age_logits, gender_logits = model(inputs['input_ids'], inputs['attention_mask'])
            age_loss = criterion(age_logits, age_labels)
            gender_loss = criterion(gender_logits, gender_labels)
            loss = age_loss + gender_loss

            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss / len(validation_loader))

    print(f'Validation Loss: {total_loss / len(validation_loader)}')
