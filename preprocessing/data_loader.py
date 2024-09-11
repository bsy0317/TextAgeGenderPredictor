import json
import os
from torch.utils.data import Dataset
from collections import defaultdict
from tqdm import tqdm

class DialogueDataset(Dataset):
    def __init__(self, data_dir):
        self.data = []  # 발화(utterance) 목록
        self.labels = []  # 성별, 나이 레이블 목록
        self.label_counts = defaultdict(int)
        self._load_data(data_dir)
        self._print_label_counts()

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
                            text = utterance['utterance']

                            # 발화자(participantID) 정보
                            participant_id = utterance['participantID']
                            participant_info = next(p for p in participants_info if p['participantID'] == participant_id)

                            # 남성(0), 여성(1)으로 매핑
                            gender = 0 if participant_info['gender'] == '남성' else 1
                            age_str = participant_info['age']
                            age_map = {"20대": 0, "30대": 1, "40대": 2, "50대": 3, "60대": 4, "70대": 5}
                            if age_str in age_map:
                                age = age_map[age_str]

                            # 데이터 및 레이블 추가
                            self.data.append(text)
                            self.labels.append((gender, age))
                            self.label_counts[(gender, age)] += 1

    def _print_label_counts(self):
        for label, count in self.label_counts.items():
            gender_label = "Male" if label[0] == 0 else "Female"
            age_label = f"{label[1]*10 + 20}s"
            print(f"Gender: {gender_label}, Age: {age_label} => {count} Counts")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        utterance, (gender, age) = self.data[idx], self.labels[idx]
        return utterance, (gender, age)
