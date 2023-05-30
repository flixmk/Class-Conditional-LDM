import h5py
import numpy as np
import os
from subprocess import call
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader, Dataset

from lightning.pytorch import LightningDataModule
import datasets
from tqdm import tqdm
import glob
from PIL import Image
import shutil

class H5PyTorchDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        with h5py.File(self.file_path, 'r') as f:
            self.latents = f['latents'][:]
            self.labels = f['labels'][:]
        print("length of labels: ", len(self.labels))

    def __getitem__(self, index):
        latent = self.latents[index]
        label = self.labels[index]
        if np.isscalar(label):
            label = np.array([label])

        # Convert the data to PyTorch tensors
        latent = torch.from_numpy(latent)
        label = torch.from_numpy(label)

        return {'latents': latent, 'labels': label}

    def __len__(self):
        return len(self.latents)

class LatentDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset_path: str = "./array_file_train.h5",
        val_dataset_path: str = "./sd_vae_val_latents.h5",
        train_batch_size: int = 32,
        val_batch_size: int = 8,
        num_workers: int = 32,
    ):
        super().__init__()
        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

#         try:
#             if os.path.isfile(self.train_dataset_path):
#                 print("Train dataset already exist")
#             else:
#                 download_file_name = self.train_dataset_path.split("/")[-1]
#                 call(f'wget -q https://huggingface.co/flix-k/sd_dependencies/resolve/main/{download_file_name}', shell=True)
#             if os.path.isfile(self.val_dataset_path):
#                 print("Val dataset already exist")
#             else:
#                 download_file_name = self.val_dataset_path.split("/")[-1]
#                 call(f'wget -q https://huggingface.co/flix-k/sd_dependencies/resolve/main/{download_file_name}', shell=True)
#         except:
#             pass
            
#         if not os.path.isfile("./finetuned_best.pt"):
#             call('wget -q https://huggingface.co/flix-k/sd_dependencies/resolve/main/finetuned_best.pt', shell=True)
#         if not os.path.isfile("./low_vloss_e6.pt"):
#             call('wget -q https://huggingface.co/flix-k/sd_dependencies/resolve/main/low_vloss_e6.pt', shell=True)

        self.train_dataset = H5PyTorchDataset(self.train_dataset_path)
        self.val_dataset = H5PyTorchDataset(self.val_dataset_path)
        
        # loading original data from huggingface
        orig_val_data = datasets.load_dataset("flix-k/oct-dataset-val1kv3", split='val')
        
        if not os.path.exists("./val_images"):
            os.mkdir("./val_images")

        files = glob.glob("./val_images/*")
        if len(files) < len(orig_val_data):
            for i in tqdm(range(len(orig_val_data))):
                img = Image.fromarray(np.array(orig_val_data[i]["image"]))
                image_name = orig_val_data[i]["caption"]
                img.save(f"./val_images/{image_name}-{i}.jpg")
                
        del orig_val_data

        # Path to the folder containing the images
        folder_path = "./val_images/"

        # Remove all existing folders in the parent folder
        for root, dirs, files in os.walk(folder_path, topdown=False):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                shutil.rmtree(dir_path)

        # Get the list of files in the folder
        files = os.listdir(folder_path)

        # Create a dictionary to store the class names and their corresponding images
        class_images = {}

        # Iterate through the files
        for file in files:
            # Split the file name and extension
            filename, extension = os.path.splitext(file)
            
            # Extract the class name from the file name
            class_name = filename.split("-")[0]
            
            # Create a new entry in the dictionary if the class name doesn't exist
            if class_name not in class_images:
                class_images[class_name] = []
            
            # Add the file to the list of images for the class
            class_images[class_name].append(file)

        # Create new folders for each class and copy the corresponding images
        for class_name, images in class_images.items():
            # Create a new folder for the class
            new_folder_path = os.path.join(folder_path, class_name)
            os.makedirs(new_folder_path, exist_ok=True)
            
            # Copy the images to the new folder
            for image in images:
                image_path = os.path.join(folder_path, image)
                new_image_path = os.path.join(new_folder_path, image)
                shutil.copy2(image_path, new_image_path)

        # Create a folder called "ALL" and move the remaining images to it
        remaining_folder_path = os.path.join(folder_path, "ALL")
        os.makedirs(remaining_folder_path, exist_ok=True)

        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                new_file_path = os.path.join(remaining_folder_path, file)
                shutil.copy2(file_path, new_file_path)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            collate_fn=None,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            collate_fn=None,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
