#!/bin/bash

# Activate virtualenv
source /home/wd/Documents/work_stuff/ViT_REPLICATION/_vit_rep_py310/bin/activate

# Move to project root (for relative imports)
cd /home/wd/Documents/work_stuff/ViT_REPLICATION

# Run training script
python scripts/train.py
