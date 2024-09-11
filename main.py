import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from models.gender_age_model import GenderAgeModel
from preprocessing.data_loader import DialogueDataset
from transformers import AutoTokenizer
from training.train import train_model
from training.validation import validate_model
from utils.collate import custom_collate_fn
from utils.checkpoint import save_checkpoint, load_checkpoint
import numpy as np

def main():
    device_name = "cpu"
    if torch.cuda.is_available():
        device_name = "cuda"
    elif torch.backends.mps.is_available():
        device_name = "mps"
    device = torch.device(device_name)
    
    # 데이터 경로 설정
    train_data_path = './data/Training'
    validation_data_path = './data/Validation'

    # 데이터셋 로드 및 DataLoader 설정
    print("="*10+"Loading Train Dataset"+"="*10)
    train_dataset = DialogueDataset(train_data_path)
    
    # 불균형 데이터 처리
    labels = [label for _, label in train_dataset]
    gender_labels, age_labels = zip(*labels)
    
    gender_class_weights = 1. / np.bincount(gender_labels)
    age_class_weights = 1. / np.bincount(age_labels)

    # 각 샘플별 가중치 계산
    sample_weights = []
    for g_label, a_label in zip(gender_labels, age_labels):
        gender_weight = gender_class_weights[g_label]
        age_weight = age_class_weights[a_label]
        sample_weights.append(gender_weight + age_weight)

    # 데이터 로더
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    train_loader = DataLoader(
        train_dataset, 
        batch_size=128, 
        sampler=sampler, 
        collate_fn=custom_collate_fn,
        num_workers=8,
        pin_memory=True
    )

    print("="*10+"Loading Validation Dataset"+"="*10)
    validation_dataset = DialogueDataset(validation_data_path)
    validation_loader = DataLoader(
        validation_dataset, 
        batch_size=128, 
        shuffle=False, 
        num_workers=8,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    # KcELECTRA 모델 및 토크나이저 로드
    model_name = "beomi/KcELECTRA-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GenderAgeModel().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 가중치 불러오기
    start_epoch = 0
    checkpoint_path = "checkpoint.pth"
    
    # 체크포인트가 존재하는 경우 불러오기
    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

    # 모델 학습
    train_model(model, train_loader, tokenizer, criterion, optimizer, device, epochs=5, start_epoch=start_epoch)

    # 모델 검증
    validate_model(model, validation_loader, tokenizer, criterion, device)
    
    # 모델 저장
    torch.save(model.state_dict(), 'model.pth')
    
    # 모델 로드
    loaded_model = GenderAgeModel().to(device)
    loaded_model.load_state_dict(torch.load('model.pth'))
    loaded_model.eval()
    
    # 모델 추론
    input_text = "안녕하세요~~"
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    age_logits = loaded_model(inputs['input_ids'], inputs['attention_mask'])[0]
    gender_logits = loaded_model(inputs['input_ids'], inputs['attention_mask'])[1]
    print(f"Age logits: {age_logits} / Gender logits: {gender_logits}")
if __name__ == "__main__":
    main()
