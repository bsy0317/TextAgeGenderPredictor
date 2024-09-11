import torch

def save_checkpoint(model, optimizer, epoch, file_name="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, file_name)
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(model, optimizer, file_name="checkpoint.pth"):
    checkpoint = torch.load(file_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from epoch {epoch}")
    return epoch
