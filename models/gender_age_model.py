import torch
import torch.nn as nn
from transformers import ElectraModel

class GenderAgeModel(nn.Module):
    def __init__(self, num_age_classes=6, num_gender_classes=2):
        super(GenderAgeModel, self).__init__()
        # KcELECTRA 모델을 로드합니다
        self.electra = ElectraModel.from_pretrained("beomi/KcELECTRA-base")
        
        # 드롭아웃과 분류 레이어를 정의합니다
        self.dropout = nn.Dropout(0.3)
        
        # 성별 분류기 (2개의 클래스: 남성/여성)
        self.gender_classifier = nn.Linear(768, num_gender_classes)
        
        # 나이 분류기 (6개의 클래스: 20대, 30대, 40대, 50대, 60대, 70대)
        self.age_classifier = nn.Linear(768, num_age_classes)

    def forward(self, input_ids, attention_mask):
        # KcELECTRA를 통해 인코딩된 출력값을 얻습니다
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] 토큰에 해당하는 출력
        
        # 드롭아웃 적용
        pooled_output = self.dropout(pooled_output)
        
        # 성별 및 나이 예측
        gender_logits = self.gender_classifier(pooled_output)
        age_logits = self.age_classifier(pooled_output)
        
        return age_logits, gender_logits
