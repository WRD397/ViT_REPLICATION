import sys
import os
from dotenv import load_dotenv
load_dotenv()
ROOT_DIR_PATH = os.environ.get('ROOT_PATH')
sys.path.append(os.path.abspath(ROOT_DIR_PATH))  # Adds root directory to sys.path

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.vit import VisionTransformerSmall
from utils.model_io import save_model
from utils.config_loader import load_config
from utils.data_loader import DatasetLoader
from pynvml import (
    nvmlInit, nvmlDeviceGetName, nvmlShutdown,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates
)
from torch.optim.lr_scheduler import CosineAnnealingLR
from timm.data import Mixup
import numpy as np
from transformers import get_cosine_schedule_with_warmup

def train_one_epoch(model, loader, criterion, optimizer, device, 
                    mixup_fn=None, scheduler_warmup_enabled=False, scheduler_warmup=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc="Training", leave=True)
    for  inputs, targets in progress_bar:
        #print(f'input shape : {inputs.shape}, taget_shape : {targets.shape}, target dim : {targets.ndim}')
        inputs, targets = inputs.to(device), targets.to(device)
        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)

        if targets.ndim == 2:
            targets = targets.type_as(inputs)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if scheduler_warmup_enabled:
            if scheduler_warmup is None : raise Exception(f'scheduler warmup is enabled, but no scheduler object has been passed in train_one_epoch function')
            scheduler_warmup.step()

        running_loss += loss.item() * inputs.size(0)

        if targets.ndim == 2:
            # MixUp with soft labels
            _, predicted = outputs.max(1)
            _, true_classes = targets.max(1)  # Take argmax of soft labels as true class
            correct += predicted.eq(true_classes).sum().item()
            total += targets.size(0)
        else :
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

        # Update progress bar with metrics
        if total > 0:
            avg_loss = running_loss / total
            accuracy = 100. * correct / total
            progress_bar.set_postfix({
                "Loss": f"{avg_loss:.4f}",
                "Acc": f"{accuracy:.2f}%"
            })

        else : raise Exception(f'Expected non-zero batch size, but got 0 targets. Check if the dataset is empty or DataLoader is misconfigured.')

    
    return avg_loss, accuracy

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(loader, desc="Validation", leave=True)
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # If labels are soft (e.g., using BCEWithLogitsLoss), convert to float
            if labels.ndim == 2:
                labels = labels.type_as(inputs)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            # Compute accuracy
            _, predicted = outputs.max(1)

            if labels.ndim == 2:
                # Soft labels → convert to class index
                _, true_classes = labels.max(1)
                correct += predicted.eq(true_classes).sum().item()

            else:
                # Hard labels
                correct += predicted.eq(labels).sum().item()


            total += labels.size(0)

            # Avoid division by zero on first step
            if total > 0:
                avg_loss = running_loss / total
                accuracy = 100. * correct / total

                progress_bar.set_postfix({
                    "Loss": f"{avg_loss:.4f}",
                    "Acc": f"{accuracy:.2f}%"
                })
                
    return avg_loss, accuracy

def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config
    config = load_config(f"{ROOT_DIR_PATH}/config/vit_config.yaml")
    # loading cifar100
    #cifar100_config = config["data"]['CIFAR100']
    dataset_config = config["data"]['CIFAR100']

    DATASET = dataset_config["dataset"]
    DATA_DIR = dataset_config["data_path"]
    BATCH = dataset_config["batch_size"]
    NUM_WORKERS = dataset_config["num_workers"]
    IMAGE = dataset_config["img_size"]
    NUM_CLASSES = dataset_config["num_classes"]
    CHANNELS = dataset_config["channels"]
    
    # Model
    modelConfig = config["model"]
    specific_config = modelConfig['VIT_SMALL']
    MODEL_NAME = specific_config["name"]
    modelConfigDict = {
        'CHANNEL' : CHANNELS,
        'PATCH' : specific_config['patch_size'],
        'EMBEDDING' : specific_config['emb_size'],
        'IMAGE' : IMAGE,
        'NUM_HEADS' : specific_config['num_heads'],
        'MLP_RATIO' : specific_config['mlp_ratio'],
        'DROPOUT' : specific_config['dropout'],
        'NUM_CLASSES' : NUM_CLASSES,
        'DEPTH' : specific_config['depth']
    }    
    CHECKPOINT_DIR = f'{ROOT_DIR_PATH}/checkpoints/{MODEL_NAME}'

    # training config
    trainingConfig = config['training']
    LEARNING_RATE = trainingConfig['lr']
    EPOCHS = trainingConfig['epochs']
    WEIGHT_DECAY = trainingConfig['weight_decay']
    USE_SCHEDULER = trainingConfig['scheduler']
    USE_SCHEDULER_WARMUP = trainingConfig['scheduler_warmup']
    WARMUP_STEPS = trainingConfig['warmup_steps']
    USE_LABEL_SMOOTHENING = trainingConfig["label_smoothing_enabled"]
    LABEL_SMOOTHENING = trainingConfig["label_smoothing"]
    # mixup config
    mixupConfig = trainingConfig['mixup']
    MIXUP_ALPHA = mixupConfig["mixup_alpha"]
    CUTMIX_ALPHA = mixupConfig["cutmix_alpha"]
    LABEL_SMOOTHENING_MIXUP = mixupConfig["label_smoothing_mixup"]
    USE_MIXUP = mixupConfig["enabled"]
   
    # logging switches
    print('-----------')
    print('Label Smoothening is Enabled') if USE_LABEL_SMOOTHENING else print('Label Smoothening is Disabled')
    print('LR Scheduler is Enabled') if USE_SCHEDULER else print('LR Scheduler is Disabled')
    print('LR SchedulerWarmup is Enabled') if USE_SCHEDULER_WARMUP else print('LR SchedulerWarmup is Disabled')
    print('MixUp is Enabled') if USE_MIXUP else print('MixUp is Disabled')
    print('-----------')
    # === Mixup Setup ===
    mixup_fn = None
    if USE_MIXUP:
        mixup_fn = Mixup(
            mixup_alpha=MIXUP_ALPHA,
            cutmix_alpha=CUTMIX_ALPHA,
            label_smoothing=LABEL_SMOOTHENING_MIXUP,
            num_classes=NUM_CLASSES
        )
        train_criterion = nn.BCEWithLogitsLoss()
        val_criterion = nn.CrossEntropyLoss()
    else:        
        mixup_fn = None
        train_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHENING if USE_LABEL_SMOOTHENING else 0.0)
        val_criterion = train_criterion

    # loading data
    print(f'loading dataset : {DATASET}')
    loader = DatasetLoader(dataset_name=DATASET,
                            data_dir=DATA_DIR,
                            batch_size=BATCH,
                            num_workers=NUM_WORKERS,
                            img_size=IMAGE)
    train_loader, val_loader = loader.get_loaders()
    print(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    print('data sanity check')
    for images, labels in train_loader:
        print(f'image shape and labels shape in training data - one batch : {images.shape}, {labels.shape}')
        break
    
    # loading model
    model = VisionTransformerSmall(**modelConfigDict).to(device)
    # logging model parameters and config
    print('Data Configuration:')
    for k, v in dataset_config.items():
        print(f"  {k}: {v}")
    print("Model Configuration:")
    for k, v in specific_config.items():
        print(f"  {k}: {v}")
    print("Training Configuration:")
    for k, v in trainingConfig.items():
        print(f"  {k}: {v}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f'printing a few of the model weights - should be random and unique in every run.')
    print(model.patch_embed.projection.weight[0][0][:5]) 

    # optimizer & scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler_warmup_obj = None
    scheduler_warmup_enabled_flag = False
    if USE_SCHEDULER : 
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    elif USE_SCHEDULER_WARMUP:
        scheduler_warmup_enabled_flag = True
        num_training_steps = EPOCHS * len(train_loader)
        num_warmup_steps = WARMUP_STEPS * len(train_loader)

        scheduler_warmup = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        scheduler_warmup_obj = scheduler_warmup

    # monitors initialization
    best_val_acc = 0.0
    best_val_loss = np.inf
    best_model_state = None
    # gpu utilization
    max_mem_used = 0
    max_gpu_util = 0
    max_mem_util = 0

    # Training loop
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    gpu_name = nvmlDeviceGetName(handle)

    startTime = time.time()
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, train_criterion, optimizer, device, 
                                                mixup_fn=mixup_fn,
                                                scheduler_warmup_enabled=scheduler_warmup_enabled_flag,
                                                scheduler_warmup=scheduler_warmup_obj)
        val_loss, val_acc = validate(model, val_loader, val_criterion, device)
        if USE_SCHEDULER : scheduler.step()

        # Monitor GPU usage
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        util_info = nvmlDeviceGetUtilizationRates(handle)
        mem_used_mb = mem_info.used / (1024 ** 2)
        max_mem_used = max(max_mem_used, mem_used_mb)
        max_gpu_util = max(max_gpu_util, util_info.gpu)
        max_mem_util = max(max_mem_util, util_info.memory)
        import copy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            corresponding_train_acc = train_acc
            corresponding_train_loss = train_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())

        
        endTime = time.time()
        elapsedTime = endTime - startTime
        hours = int(elapsedTime // 3600)
        minutes = int((elapsedTime % 3600) // 60)
        seconds = int(elapsedTime % 60)
        print(f"Elapsed Time : {hours}h : {minutes}m : {seconds}s")
  
    # gpu monitoring shutdown 
    nvmlShutdown()

    # saving the best state
    best_model_name_save = f'{MODEL_NAME}_data{DATASET}_valacc{round(best_val_acc)}.pth'
    try :
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': best_optimizer_state,
                    'val_acc': best_val_acc,
                    'val_loss': best_val_loss,
                    'train_acc': corresponding_train_acc,
                    'train_loss': corresponding_train_loss,
                },f'{CHECKPOINT_DIR}/{best_model_name_save}')
        print(f"Saved best model from epoch {best_epoch} with val_acc={best_val_acc:.4f}")
    except Exception as e:
        raise Exception(f'\n****nmodel save failed due to error\n {e}\n')

    print('\n====== Model Performance =======')
    print(f"Train     Loss: {corresponding_train_loss:.4f},    Accuracy: {corresponding_train_acc:.2f}%")
    print(f"Val(Best) Loss: {best_val_loss:.4f}, Accuracy: {best_val_acc:.2f}%")
    print('====== Hardware Performance =======')
    print(f"GPU Used : {gpu_name}")
    print(f"Peak GPU Memory: {max_mem_used:.2f} MB")
    print(f"Peak GPU Utilization: {max_gpu_util}%")
    print(f"Peak Memory Bandwidth Utilization: {max_mem_util}%")
    print(f"\nTraining completed in: {hours}h : {minutes}m : {seconds}s\n\n")
    

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
    
