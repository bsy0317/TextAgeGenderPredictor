import torch

def save_checkpoint(model, optimizer, epoch, file_name="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, file_name)
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(model, optimizer, file_name="checkpoint.pth", map_location=None):
    map_location = map_location or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(file_name, map_location=torch.device(map_location))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)  # Use 0 as default if 'epoch' is missing
    print(f"Checkpoint loaded from epoch {epoch}")
    
    return epoch

