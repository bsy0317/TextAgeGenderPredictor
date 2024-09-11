import torch

def accuracy(predictions, labels):
    _, preds = torch.max(predictions, dim=1)
    correct = torch.sum(preds == labels).item()
    return correct / len(labels)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model
