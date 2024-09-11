import torch

def custom_collate_fn(batch):
    utterances, labels = zip(*batch)
    genders, ages = zip(*labels)
    
    return list(utterances), (torch.tensor(genders), torch.tensor(ages))