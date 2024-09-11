import json
import os
import torch
from torch.utils.data import Dataset, random_split
from tqdm import tqdm

class DialogueDataset(Dataset):
    def __init__(self, data_dir, split_ratio=0.5):
        self.data = []  # 발화(utterance) 목록
        self.labels = []  # 성별, 나이 레이블 목록
        self._load_data(data_dir)

        # 전체 데이터셋 크기
        dataset_size = len(self.data)
        
        # nn% 데이터를 학습용으로 샘플링
        train_size = int(split_ratio * dataset_size)
        val_size = dataset_size - train_size
        
        self.train_data, self.val_data = random_split(list(zip(self.data, self.labels)), [train_size, val_size])

    def _load_data(self, data_dir):
        for file_name in tqdm(os.listdir(data_dir), desc="Loading Data"):
            if file_name.endswith(".json"):
                file_path = os.path.join(data_dir, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    for dialogue in json_data['data']:
                        participants_info = dialogue['header']['participantsInfo']  # 참여자 정보
                        
                        for utterance in dialogue['body']:
                            # 발화(utterance) 텍스트
                            self.data.append(utterance['utterance'])

                            # 발화자(participantID) 정보
                            participant_id = utterance['participantID']
                            participant_info = next(p for p in participants_info if p['participantID'] == participant_id)

                            # 남성(0), 여성(1)으로 매핑
                            gender = 0 if participant_info['gender'] == '남성' else 1
                            age_str = participant_info['age']
                            age_map = {"20대": 0, "30대": 1, "40대": 2, "50대": 3, "60대": 4, "70대": 5}
                            if age_str in age_map:
                                age = age_map[age_str]

                            self.labels.append((gender, age))

    def __len__(self, split='train'):
        if split == 'train':
            return len(self.train_data)
        else:
            return len(self.val_data)

    def __getitem__(self, idx, split='train'):
        if split == 'train':
            utterance, (gender, age) = self.train_data[idx]  # train_data에서 튜플 반환
        else:
            utterance, (gender, age) = self.val_data[idx]  # val_data에서 튜플 반환
        return utterance, (gender, age)
