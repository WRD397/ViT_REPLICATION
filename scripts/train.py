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
from tqdm import tqdm
from model.vit import VisionTransformerSmall
from model.vit_custom import VisionTransformerTiny
from utils.model_io import measure_throughput
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
import wandb
from utils.checkpoints_manager import CheckpointManager
from datetime import datetime as dt
from torch.amp import GradScaler
from torch.amp import autocast
import gc
import subprocess


def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
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
        with autocast(device_type='cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()  
        scaler.step(optimizer)         
        scaler.update()                

        if scheduler_warmup_enabled:
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
    top5_correct = 0
    total = 0
    progress_bar = tqdm(loader, desc="Validation", leave=True)
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            with autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            # Compute accuracy
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            # Top-5 Accuracy
            _, top5_preds = outputs.topk(5, dim=1, largest=True, sorted=True)
            top5_correct += sum([labels[i] in top5_preds[i] for i in range(len(labels))])

            total += labels.size(0)

            avg_loss = running_loss / total
            accuracy = 100. * correct / total
            top5_accuracy = 100. * top5_correct / total

            progress_bar.set_postfix({
                "Loss": f"{avg_loss:.4f}",
                "Acc": f"{accuracy:.2f}%"
            })
                
    return avg_loss, accuracy, top5_accuracy

def main():
    
    print('loading config')
    # Load config
    config = load_config(f"{ROOT_DIR_PATH}/config/vit_config.yaml")
    PROJECT_NAME = config['project']
    RUN_NAME = config['run']
    RUN_NOTES = config['run_notes']
    WANDB_TAGS = config['wandb_tags']

    # *************  choosing the DATASET & MODEL *************
    
    dataset_config = config["data"]['TINYIMAGENET200']
    specific_config = config["model"]['VIT_TINYV3']
    trainingConfig = config['training']

    # **********************************************************
    
    # data
    DATASET = dataset_config["dataset"]
    DATA_DIR =f'{ROOT_DIR_PATH}/data/{DATASET}/'
    BATCH = dataset_config["batch_size"]
    NUM_WORKERS = dataset_config["num_workers"]
    IMAGE = dataset_config["img_size"]
    NUM_CLASSES = dataset_config["num_classes"]
    CHANNELS = dataset_config["channels"]
    if DATASET == 'TINYIMAGENET200':
        SUBSET_ENABLED = dataset_config['subset_enabled']
        SUBSET_SIZE = dataset_config['subset_size']
    
    # Model
    MODEL_NAME = specific_config['name']
    modelConfigDict = {
        'CHANNEL' : CHANNELS,
        'PATCH' : specific_config['patch_size'],
        'EMBEDDING' : specific_config['emb_size'],
        'IMAGE' : IMAGE,
        'NUM_HEADS' : specific_config['num_heads'],
        'MLP_RATIO' : specific_config['mlp_ratio'],
        'DROPOUT' : specific_config['dropout'],
        'NUM_CLASSES' : NUM_CLASSES,
        'DEPTH' : specific_config['depth'],
        'QKV_BIAS':specific_config['qkv_bias'],
        'ATTN_DROP_RATE': specific_config['attn_drop_rate'],
        'DROP_PATH_RATE': specific_config['drop_path_rate']
    }    

    # training config
    LEARNING_RATE = trainingConfig['lr']
    EPOCHS = trainingConfig['epochs']
    WEIGHT_DECAY = trainingConfig['weight_decay']
    USE_SCHEDULER = trainingConfig['scheduler']
    USE_SCHEDULER_WARMUP = trainingConfig['scheduler_warmup']
    WARMUP_STEPS = trainingConfig['warmup_steps']
    USE_LABEL_SMOOTHENING = trainingConfig["label_smoothing_enabled"]
    LABEL_SMOOTHENING = trainingConfig["label_smoothing"]
    EARLY_STOPPING_PATIENCE = trainingConfig["es_patience"]
    EARLY_STOPPING_IMPROVEMENT_DELTA = trainingConfig["es_improv_delta"]
    AUG_ENABLED = trainingConfig["augmentation_enabled"]

    # mixup config
    mixupConfig = trainingConfig['mixup']
    MIXUP_ALPHA = mixupConfig["mixup_alpha"]
    CUTMIX_ALPHA = mixupConfig["cutmix_alpha"]
    LABEL_SMOOTHENING_MIXUP = mixupConfig["label_smoothing_mixup"]
    USE_MIXUP = mixupConfig["enabled"]
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
   
    # logging switches
    print('-----------')
    print('Label Smoothening is Enabled') if USE_LABEL_SMOOTHENING else print('Label Smoothening is Disabled')
    print('LR Scheduler is Enabled') if USE_SCHEDULER else print('LR Scheduler is Disabled')
    print('LR SchedulerWarmup is Enabled') if USE_SCHEDULER_WARMUP else print('LR SchedulerWarmup is Disabled')
    print('MixUp is Enabled') if USE_MIXUP else print('MixUp is Disabled')
    print('Data Augmentation is Enabled') if AUG_ENABLED else print('Data Augmentation is Disabled')
    if DATASET == 'TINYIMAGENET': print(f'Subset is Enabled - {SUBSET_SIZE}') if SUBSET_ENABLED else print('Subset is Disabled.')
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
        val_criterion = nn.CrossEntropyLoss(label_smoothing=0.0) #NO LABEL SMOOTHENING during validation
    else:        
        mixup_fn = None
        train_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHENING if USE_LABEL_SMOOTHENING else 0.0)
        val_criterion =  nn.CrossEntropyLoss(label_smoothing=0.0) #NO LABEL SMOOTHENING during validation

    # loading data
    print(f'loading dataset : {DATASET}')
    loader = DatasetLoader(training_config=trainingConfig,
                            dataset_name=DATASET,
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
    #model = VisionTransformerSmall(**modelConfigDict).to(device)
    model = VisionTransformerTiny(**modelConfigDict).to(device)
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

    # scaler
    scaler = GradScaler() 

    loggable_config = {
        "dataset": DATASET,
        "train_sample":len(train_loader),
        "val_sample": len(val_loader),
        "subset_size": SUBSET_SIZE if DATASET == 'TINYIMAGENET200' else np.nan,
        "batch_size": BATCH,
        "img_size": IMAGE,
        
        "model_name": MODEL_NAME,
        "model_param_m": round((total_params / 1e6),2),
        "patch_size": specific_config['patch_size'],
        "embed_dim": specific_config["emb_size"],
        "depth":specific_config["depth"],
        "heads": specific_config["num_heads"],
        "mlp_ratio": specific_config["mlp_ratio"],
        "dropout":specific_config["dropout"],
        "drop_path_rate":specific_config["drop_path_rate"],
        "attn_drop_rate": specific_config['attn_drop_rate'],

        "epochs": config["training"]["epochs"],
        "lr": config["training"]["lr"],
        "mixup_alpha": config["training"]["mixup"]["mixup_alpha"] if config["training"]["mixup"]["enabled"] else np.nan,
        "cutmix_alpha": config["training"]["mixup"]["cutmix_alpha"] if config["training"]["mixup"]["enabled"] else np.nan,
        "label_smooth_mixup": config["training"]["mixup"]["label_smoothing_mixup"] if config["training"]["mixup"]["enabled"] else np.nan,
        "label_smooth": config["training"]["label_smoothing"] if config["training"]["label_smoothing_enabled"] else np.nan,
        "weight_decay": config["training"]["weight_decay"],
        "augmentation":config["training"]["augmentation_enabled"]
    }

    print('initializing wandb')
    
    # creating and storing runids for future reference.
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{RUN_NAME}_{timestamp}"
    runid_file_path = f"{ROOT_DIR_PATH}/wandb_runids/wandb_run_id.txt"
    dir_path = os.path.dirname(runid_file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(runid_file_path, "w") as f:
        f.write(run_id)

    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
    wandb.login(key=WANDB_API_KEY, relogin=True)
    wandb.init(
        project=PROJECT_NAME, 
        id=run_id,
        name=RUN_NAME,
        notes = RUN_NOTES,
        resume="never",
        config=loggable_config,
        allow_val_change=False,
        tags=WANDB_TAGS
        )
    
    wandb.define_metric("*", summary="none")
    checkpoint_manager = CheckpointManager()

    # monitors initialization
    best_val_acc = 0.0
    best_val_loss = np.inf
    best_model_state = None
    # gpu utilization
    max_mem_used = 0
    max_gpu_util = 0
    max_mem_util = 0
    # earlystopping
    epochs_without_improvement = 0

    # Training loop
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    gpu_name = nvmlDeviceGetName(handle)

    startTime = time.time()
    for epoch in range(1,EPOCHS+1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, train_criterion, optimizer, scaler, device,
                                                mixup_fn=mixup_fn,
                                                scheduler_warmup_enabled=scheduler_warmup_enabled_flag,
                                                scheduler_warmup=scheduler_warmup_obj)
        val_loss, val_acc, val_acc_top5 = validate(model, val_loader, val_criterion, device)
        if USE_SCHEDULER : scheduler.step()

        # monitoring learning rate variation
        current_lr = optimizer.param_groups[0]['lr']

        # Monitor GPU usage
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        util_info = nvmlDeviceGetUtilizationRates(handle)
        mem_used_mb = mem_info.used / (1024 ** 2)
        max_mem_used = max(max_mem_used, mem_used_mb)
        max_gpu_util = max(max_gpu_util, util_info.gpu)
        max_mem_util = max(max_mem_util, util_info.memory)
        
        import copy
        if val_acc > best_val_acc + EARLY_STOPPING_IMPROVEMENT_DELTA:
            best_val_acc = val_acc
            best_val_acc_top5 = val_acc_top5
            best_val_loss = val_loss
            corresponding_train_acc = train_acc
            corresponding_train_loss = train_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            if USE_SCHEDULER: best_scheduler_state = copy.deepcopy(scheduler.state_dict())
            elif USE_SCHEDULER_WARMUP : best_scheduler_state = copy.deepcopy(scheduler_warmup_obj.state_dict())
            else : best_scheduler_state = None
            best_scaler_state = copy.deepcopy(scaler.state_dict())

            epochs_without_improvement = 0
        else:
            epochs_without_improvement+=1

        # Early stopping condition
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered. No improvement(delta={EARLY_STOPPING_IMPROVEMENT_DELTA}) in val_acc for {EARLY_STOPPING_PATIENCE} consecutive epochs.")
            break

        endTime = time.time()
        elapsedTime = endTime - startTime
        hours = int(elapsedTime // 3600)
        minutes = int((elapsedTime % 3600) // 60)
        seconds = int(elapsedTime % 60)
        print(f"Elapsed Time : {hours}h : {minutes}m : {seconds}s")

        if epoch%5 == 0:
            wandb.log({
                    "Time/Epoch (min)": int((elapsedTime/epoch)/60)
                })
        wandb.log(
        {
            "Epoch": epoch,
            "Train Loss": train_loss,
            "Train Acc": train_acc,
            "Val Loss": val_loss,
            "Val Acc": val_acc,
            "Val Acc (Top5)": val_acc_top5,
            "LR (Scheduler Warmup)": current_lr
        })

        #saving the latest best model/optimizer dict at 10 epochs interval
        checkpoint_manager.save_and_upload(
            current_epoch=epoch,
            best_epoch=best_epoch,
            model_state=best_model_state,
            optimizer_state=best_optimizer_state,
            scheduler_state=best_scheduler_state,
            scaler_state = best_scaler_state,
            extra={"val_acc": best_val_acc,
                   "val_acc_top5" : best_val_acc_top5,
                   "val_loss": best_val_loss,
                   "train_acc":corresponding_train_acc,
                   "train_loss":corresponding_train_loss
                    }
            )

    # gpu monitoring shutdown 
    nvmlShutdown()

    # saving the best state
    print('\n====== Model Performance =======')
    print(f"Train     Loss: {corresponding_train_loss:.4f},    Accuracy: {corresponding_train_acc:.2f}%")
    print(f"Val(Best) Loss: {best_val_loss:.4f}, Accuracy: {best_val_acc:.2f}%, Top5 Accuracy: {best_val_acc_top5:.2f}%")
    print('====== Hardware Performance =======')
    print(f"GPU Used : {gpu_name}")
    print(f"Peak GPU Memory: {max_mem_used:.2f} MB")
    print(f"Peak GPU Utilization: {max_gpu_util}%")
    print(f"Peak Memory Bandwidth Utilization: {max_mem_util}%")
    throughput = measure_throughput(model,device,IMAGE)
    print(f"Throughput: {throughput}")
    print(f"average time per epoch: {int((elapsedTime/epoch)/60)}")
    print('\n\n-----------')
    print(f"saving & uploading the best model state as till epoch : {epoch}")
    print(f"Best model Epoch : {best_epoch}")
    checkpoint_manager.save_checkpoint(
        best_epoch=best_epoch,
        model_state=best_model_state,
        optimizer_state=best_optimizer_state,
        scheduler_state=best_scheduler_state,
        scaler_state = best_scaler_state,
        extra={"val_acc": best_val_acc,
               "val_acc_top5" : best_val_acc_top5,
                "val_loss": best_val_loss,
                "train_acc":corresponding_train_acc,
                "train_loss":corresponding_train_loss
                }
    )
    checkpoint_manager.upload_to_wandb()
    time.sleep(5)
    checkpoint_manager.cleanup_old_wandb_artifacts()   
    print('-----------\n')
    print(f"\nTraining completed in: {hours}h : {minutes}m : {seconds}s\n\n")
    
    wandb.run.summary["performance"] = {
    "trainAcc": corresponding_train_acc,
    "trainLoss": corresponding_train_loss,
    "valAcc": best_val_acc,
    "valAccTop5": best_val_acc_top5,
    "valLoss": best_val_loss,
    "param(m)": round((total_params / 1e6),2),
    "throughput": throughput,
    "avgTime/Epoch(min)": int((elapsedTime/epoch)/60),
    "elapsedTime":f"{hours}h : {minutes}m : {seconds}s",

    }

    #wandb syncing - syncing any possible leftovers
    script_path = f"{ROOT_DIR_PATH}/scripts/wandb_sync.sh"

    try:
        result = subprocess.run(["bash", script_path], check=True, capture_output=True, text=True)

        print("Script completed successfully.")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        wandb.finish()

    except subprocess.CalledProcessError as e:
        print("Script failed with return code", e.returncode)
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        wandb.finish()

if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
    main()