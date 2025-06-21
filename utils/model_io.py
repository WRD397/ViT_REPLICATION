from pathlib import Path
import torch
import os 
from dotenv import load_dotenv
import time

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

def measure_throughput(model, device, image_size):
    BATCH_SIZE = 128
    IMAGE_SIZE = image_size
    CHANNELS = 3
    NUM_STEPS = 100
    NUM_WARMUP = 10

    model.to(device)
    model.eval()

    dummy_input = torch.randn(BATCH_SIZE, CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)
    #warmup
    for _ in range(NUM_WARMUP):
        _ = model(dummy_input)

    #benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(NUM_STEPS):
        _ = model(dummy_input)
        torch.cuda.synchronize()
    end = time.time()
    total_images = BATCH_SIZE * NUM_STEPS
    elapsed_time = end - start
    throughput = round(total_images / elapsed_time, 2)
    return throughput