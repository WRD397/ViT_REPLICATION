import sys
import os
from dotenv import load_dotenv
load_dotenv()
ROOT_DIR_PATH = os.environ.get('ROOT_PATH')
sys.path.append(os.path.abspath(ROOT_DIR_PATH)) 
import urllib.request
import zipfile
import shutil

DESTINATION_PATH =f'{ROOT_DIR_PATH}/data/TINYIMAGENET/'
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

def rearrange_val_folder(base_dir=f'{DESTINATION_PATH}tiny-imagenet-200/val'):
    print(f"Reorganizing validation folder at {base_dir}...")
    img_dir = os.path.join(base_dir, 'images')
    ann_file = os.path.join(base_dir, 'val_annotations.txt')

    with open(ann_file, 'r') as f:
        for line in f:
            file_name, class_name = line.split('\t')[:2]
            class_dir = os.path.join(base_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            src = os.path.join(img_dir, file_name)
            dst = os.path.join(class_dir, file_name)
            if os.path.exists(src):
                shutil.move(src, dst)

    shutil.rmtree(img_dir)
    print("Validation images reorganized!")

def prepare_tiny_imagenet():
    dst_dir = DESTINATION_PATH
    zip_fname = f"{DESTINATION_PATH}{ZIP_NAME}"

    if not os.path.isdir(dst_dir):
        print('creating the destination dir. & downloading')
        os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
        download_tiny_imagenet(zip_fname)
        print(f"TinyImageNet is ready under: {dst_dir}/")
    else:
        print(f"Dataset directory {dst_dir} already exists, zip downloaded.")

    if not os.path.isdir(f'{DESTINATION_PATH}tiny-imagenet-200/'):
        extract_dataset(zip_path=zip_fname, extract_to=os.path.dirname(dst_dir))
        rearrange_val_folder(base_dir=f'{DESTINATION_PATH}tiny-imagenet-200/val')
        print(f"TinyImageNet is ready under: {dst_dir}")
    else : print(f'TinyImageNet already exists under: {dst_dir}')


if __name__ == '__main__':
    prepare_tiny_imagenet()