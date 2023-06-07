import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from transformers import CLIPTextModel, CLIPTokenizer
import torch

from tqdm import tqdm 
from diffusers import AutoencoderKL

class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, samples_per_class=None):
        super(CustomImageFolder, self).__init__(root, transform=transform, target_transform=target_transform)
        
        self.samples_per_class = samples_per_class
        
        if samples_per_class is not None:
            # New class_to_idx dictionary
            new_class_to_idx = {}
            # New samples list
            new_samples = []
            # New targets list
            new_targets = []
            
            # For each class in the original class_to_idx
            for class_name in self.class_to_idx:
                # Get all the samples for this class
                class_samples = [(s, t) for s, t in self.samples if t == self.class_to_idx[class_name]]
                # If there are more samples than samples_per_class, trim the list
                if len(class_samples) > samples_per_class:
                    class_samples = class_samples[:samples_per_class]
                
                # Append the samples to the new samples and targets list
                new_samples.extend(class_samples)
                new_targets.extend([self.class_to_idx[class_name]] * len(class_samples))
                # Set the class_to_idx for the new class
                new_class_to_idx[class_name] = self.class_to_idx[class_name]
            
            # Set the new class_to_idx, samples, and targets
            self.class_to_idx = new_class_to_idx
            self.samples = new_samples
            self.targets = new_targets
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            dict: {'images': sample, 'ids': target}
        """
        img, target = super(CustomImageFolder, self).__getitem__(index)
        target = self.classes[target] # Get class name
        
        return {'images': img, 'targets': target}

    
def collate_fn(examples):
    input_ids = [example["targets"] for example in examples]
    pixel_values = [example["images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {
        "targets": input_ids,
        "pixel_values": pixel_values,
    }
    return batch

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])

# dataset = CustomImageFolder(root="/home/flix/Documents/oct-data/CellData/OCT_resized/train/", transform=transform, samples_per_class=4000)

# # Define the split sizes. In this case, we will split 70% for train and 30% for validation.
# train_size = int(0.85 * len(dataset))
# val_size = len(dataset) - train_size

# # Split the dataset
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# # Now you can create DataLoaders for your training and validation datasets
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)


# print("Dataset size:", len(train_dataset))
# print("Dataset size:", len(val_dataset))

# for i in range(25):
#     print(train_dataset[i]['targets'])

# print(dataset.class_to_idx)