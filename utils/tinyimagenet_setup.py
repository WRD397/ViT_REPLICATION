import sys
import os
from dotenv import load_dotenv
load_dotenv()
ROOT_DIR_PATH = os.environ.get('ROOT_PATH')
sys.path.append(os.path.abspath(ROOT_DIR_PATH)) 
import urllib.request
import zipfile
import shutil

DESTINATION_PATH =f'{ROOT_DIR_PATH}/data/TINYIMAGENET200/'
ZIP_NAME = 'tiny-imagenet-200.zip'


def download_tiny_imagenet(save_path=f'{DESTINATION_PATH}{ZIP_NAME}'):
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    print(f"Downloading TinyImageNet to {save_path}...")
    urllib.request.urlretrieve(url, save_path)
    print("Download complete!")

def extract_dataset(zip_path=f'{DESTINATION_PATH}{ZIP_NAME}', extract_to=DESTINATION_PATH):
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete!")

def rearrange_train_folder(train_dir):
    print(f"Reorganizing training folder at {train_dir}...")
    for class_dir in os.listdir(train_dir):
        full_class_path = os.path.join(train_dir, class_dir)
        images_subdir = os.path.join(full_class_path, 'images')
        
        if os.path.isdir(images_subdir):
            for img_file in os.listdir(images_subdir):
                shutil.move(os.path.join(images_subdir, img_file),
                            os.path.join(full_class_path, img_file))
            os.rmdir(images_subdir)  # remove empty 'images' subdir

def rearrange_val_folder(val_dir):
    print(f"Reorganizing validation folder at {val_dir}...")
    
    val_images_dir = os.path.join(val_dir, 'images')
    val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')
    
    # Read val_annotations.txt and build image -> class_name map
    img_to_class = {}
    with open(val_annotations_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            img_filename = parts[0]
            class_name = parts[1]
            img_to_class[img_filename] = class_name

    # Move each image to class-named subdirectory
    for img_file, class_name in img_to_class.items():
        class_dir = os.path.join(val_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        src_path = os.path.join(val_images_dir, img_file)
        dst_path = os.path.join(class_dir, img_file)
        
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)

    # Clean up
    shutil.rmtree(val_images_dir)
    os.remove(val_annotations_file)

def prepare_tiny_imagenet():
    print('-----------')
    dst_dir = DESTINATION_PATH
    zip_fname = f"{DESTINATION_PATH}{ZIP_NAME}"

    if not os.path.isdir(dst_dir):
        print('creating the destination dir. & downloading')
        os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
        download_tiny_imagenet(zip_fname)
    else:
        print(f"Dataset directory {dst_dir} already exists, zip downloaded.")

    if not os.path.isdir(f'{DESTINATION_PATH}tiny-imagenet-200/'):
        extract_dataset(zip_path=zip_fname, extract_to=os.path.dirname(dst_dir))
        rearrange_train_folder(train_dir=f'{DESTINATION_PATH}tiny-imagenet-200/train')
        rearrange_val_folder(val_dir=f'{DESTINATION_PATH}tiny-imagenet-200/val')
        print(f"TinyImageNet is ready under: {dst_dir}")
    else : print(f'TinyImageNet already exists under: {dst_dir}')
    print('*** Data sanity check')
    train_path = f'{DESTINATION_PATH}tiny-imagenet-200/train'
    val_path = f'{DESTINATION_PATH}tiny-imagenet-200/val'
    count_folders_and_files(train_path)
    count_folders_and_files(val_path)
    print('-----------')

def count_folders_and_files(root_dir):
    class_folders = [d for d in os.listdir(root_dir)
                     if os.path.isdir(os.path.join(root_dir, d))]
    
    total_classes = len(class_folders)
    total_images = 0

    for class_folder in class_folders:
        class_path = os.path.join(root_dir, class_folder)
        image_files = [f for f in os.listdir(class_path)
                       if os.path.isfile(os.path.join(class_path, f)) and f.endswith('.JPEG')]
        total_images += len(image_files)
    
    print(f"Path: {root_dir}")
    print(f"Total class folders: {total_classes}")
    print(f"Total image files:  {total_images}")
    print("-" * 40)
    return total_classes, total_images


def main():
    prepare_tiny_imagenet()
    train_path = f'{DESTINATION_PATH}tiny-imagenet-200/train'
    val_path = f'{DESTINATION_PATH}tiny-imagenet-200/val'
    count_folders_and_files(train_path)
    count_folders_and_files(val_path)


if __name__ == '__main__':
    main()
    