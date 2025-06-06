from pathlib import Path
import torch
import os 
from dotenv import load_dotenv

load_dotenv()
ROOT_DIR_PATH = os.environ.get('ROOT_PATH')

def save_model(model_state, model_name, epoch, val_acc):
    """
    model_state : model.state_dict()
    model_name : name of the model by which you wanna save the model
    epoch : epoch used in training
    val_acc : best val accuracy after training completion
    """
    save_dir = f'{ROOT_DIR_PATH}/checkpoints'
    save_dir = Path(save_dir / model_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = f"epoch{epoch}_valacc{val_acc:.2f}.pth"
    torch.save(model_state, save_dir / filename)
