import os
import torch
import wandb
from typing import Union, Dict
import sys
import os
from dotenv import load_dotenv
import time
load_dotenv()
ROOT_DIR_PATH = os.environ.get('ROOT_PATH')
sys.path.append(os.path.abspath(ROOT_DIR_PATH))  # Adds root directory to sys.path

class CheckpointManager:
    def __init__(self, epoch_interval: int = 10, keep_last: int = 1):
        """
        Args:
            save_dir (str): Path where checkpoint will be saved locally.
            save_every (int): Save checkpoint every N epochs.
            project (str): W&B project name (needed for deletion).
            keep_last (int): Number of recent W&B artifacts to retain.
        """
        self.project = wandb.run.project
        self.run_name = wandb.run.name
        self.save_dir = f"{ROOT_DIR_PATH}/checkpoints/{self.project}_checkpoint"
        os.makedirs(self.save_dir, exist_ok=True)
        self.epoch_interval = epoch_interval
        self.keep_last = keep_last
        self.checkpoint_filename = f"{self.run_name}_last_checkpoint.pth"
        self.checkpoint_path = os.path.join(self.save_dir, self.checkpoint_filename)

    def save_and_upload(self, current_epoch: int, best_epoch:int, model_state, optimizer_state,scheduler_state, scaler_state, extra: Dict = {}):
        """Save and upload the model if it's a checkpointing epoch."""
        if (current_epoch % self.epoch_interval != 0): ...
        else :
            print('start saving & loading the checkpoint as backup...')
            self.save_checkpoint(best_epoch, model_state, optimizer_state,scheduler_state, scaler_state, extra)
            self.upload_to_wandb()
            time.sleep(5)
            self.cleanup_old_wandb_artifacts()
        return None

    def save_checkpoint(self, best_epoch: int, model_state, optimizer_state, scheduler_state, scaler_state, extra: Dict):
        checkpoint = {
            "epoch": best_epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "scheduler_state_dict": scheduler_state,
            "scaler_state_dict": scaler_state,
            **extra
        }
        torch.save(checkpoint, self.checkpoint_path)
        print(f"Saved checkpoint as on epoch(best epoch) {best_epoch} to {self.checkpoint_path}")

    def upload_to_wandb(self):
        artifact_name = f"{self.run_name}_checkpoint"
        artifact = wandb.Artifact(name=artifact_name, type="model")
        artifact.add_file(self.checkpoint_path)
        artifact_ref = wandb.log_artifact(artifact)
        if artifact_ref is not None:
            artifact_ref.wait()  # Wait for completion
            print(f"Uploaded {artifact_name} to W&B")
        else:
            print(f"Upload failed or artifact not returned")

    def cleanup_old_wandb_artifacts(self):
        from wandb import Api
        api = Api()
        full_path = f"{wandb.run.entity}/{self.project}/{self.run_name}_checkpoint"
        versions = api.artifacts("model", full_path)
        if len(versions)>1:
            # Sort explicitly by creation time
            sorted_versions = sorted(
                [upload for upload in versions if upload.created_at is not None],
                key=lambda upload: upload.created_at,
                reverse=True
            )
            print(f"Keeping the latest: {[artifact.name for artifact in sorted_versions][0]}")
            for artifact in sorted_versions[1:]:
                try:
                    print(f"Deleting the last one: {artifact.name}")
                    artifact.delete()
                except Exception as e:
                    print(f"Could not delete {artifact.name}: {e}")
        else : 
            if len(versions)>0:
                print(f'not deleting anything as there is only {len(versions)} version: {[artifact.name for artifact in versions][0]}')
            else :
                print(f'no file found.')

    def load_last_checkpoint(self):
        """
        Download the latest W&B artifact checkpoint and return the state dicts.
        Returns:
            checkpoint (dict): Contains keys - 'epoch', 'model_state_dict', 'optimizer_state_dict', ...
        """
        artifact_name = f"{self.run_name}_checkpoint:latest"
        artifact = wandb.use_artifact(artifact_name, type="model")
        artifact_dir = artifact.download()
        checkpoint_path = os.path.join(artifact_dir, self.checkpoint_filename)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        print(f"Loaded checkpoint from {artifact.name} at epoch {checkpoint['epoch']}")
        return checkpoint