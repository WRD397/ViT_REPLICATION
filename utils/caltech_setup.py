import os
import sys
import shutil
import random
import zipfile
import urllib.request
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
ROOT_DIR_PATH = os.environ.get('ROOT_PATH')
sys.path.append(os.path.abspath(ROOT_DIR_PATH))

DATA_DIR = f"{ROOT_DIR_PATH}/data/CALTECH256"
ZIP_URL = "https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar"
ZIP_PATH = os.path.join(DATA_DIR, "256_ObjectCategories.tar")
EXTRACTED_FOLDER = os.path.join(DATA_DIR, "256_ObjectCategories")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

def download_caltech256():
    if not os.path.exists(ZIP_PATH):
        print("Downloading Caltech-256...")
        os.makedirs(DATA_DIR, exist_ok=True)
        urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
        print("Download complete!")
    else:
        print("Zip already exists, skipping download.")

def extract_tar():
    if not os.path.exists(EXTRACTED_FOLDER):
        print("Extracting dataset...")
        os.system(f"tar -xf {ZIP_PATH} -C {DATA_DIR}")
        print("Extraction complete!")
    else:
        print("Already extracted.")

def split_dataset():
    print("Splitting dataset into train/val...")
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    random.seed(RANDOM_SEED)

    for class_folder in os.listdir(EXTRACTED_FOLDER):
        class_path = os.path.join(EXTRACTED_FOLDER, class_folder)
        if not os.path.isdir(class_path):
            continue

        images = [f for f in os.listdir(class_path) if f.lower().endswith(".jpg")]
        random.shuffle(images)

        train_count = int(len(images) * TRAIN_RATIO)
        train_imgs = images[:train_count]
        val_imgs = images[train_count:]

        # Target folders
        train_target = os.path.join(TRAIN_DIR, class_folder)
        val_target = os.path.join(VAL_DIR, class_folder)
        os.makedirs(train_target, exist_ok=True)
        os.makedirs(val_target, exist_ok=True)

        # Copy files
        for img in train_imgs:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_target, img))
        for img in val_imgs:
            shutil.copy(os.path.join(class_path, img), os.path.join(val_target, img))

    print("Train/Val split done.")

def count_summary():
    def count_in_dir(dir_path):
        total_classes = len(os.listdir(dir_path))
        total_images = sum(len(files) for _, _, files in os.walk(dir_path))
        return total_classes, total_images

    print("-" * 40)
    for split, path in [("Train", TRAIN_DIR), ("Val", VAL_DIR)]:
        classes, images = count_in_dir(path)
        print(f"{split} set -> Classes: {classes}, Images: {images}")
    print("-" * 40)

def prepare_caltech256():

    if not os.path.isdir(DATA_DIR): 
        download_caltech256()
        extract_tar()
        split_dataset()
        count_summary()
    else : print(f"Dataset directory {DATA_DIR} already exists, zip downloaded.")

if __name__ == "__main__":
    prepare_caltech256()
