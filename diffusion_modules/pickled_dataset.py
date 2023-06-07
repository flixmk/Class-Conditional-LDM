import torch
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms
import pickle
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import lightning.pytorch as pl
from collections import defaultdict


def pickle_loader(path: str) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)

class PickleFolder(DatasetFolder):
    def __init__(
            self,
            root: str,
            samples_per_class: Optional[int] = None,
            classes_to_use: Optional[List[str]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = pickle_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(PickleFolder, self).__init__(root, loader, ('.pkl',),
                                            transform=transform,
                                            target_transform=target_transform,
                                            is_valid_file=is_valid_file)
                                            
        if samples_per_class is not None or classes_to_use is not None:
            class_sample_count = defaultdict(int)
            filtered_samples = []
            
            for path, target in self.samples:
                if classes_to_use is not None and self.classes[target] not in classes_to_use:
                    continue
                if samples_per_class is not None and class_sample_count[target] >= samples_per_class:
                    continue

                filtered_samples.append((path, target))
                class_sample_count[target] += 1

            self.samples = filtered_samples
        
        self.imgs = self.samples

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {"latents": sample, "target": target}





def collate_fn(examples):
    targets = [example["target"] for example in examples]
    pixel_values = [example["latents"].sample() for example in examples]
    pixel_values = torch.stack(pixel_values).squeeze(1)
    
    batch = {
        "latents": pixel_values,
        "classes": targets,
    }
    
    return batch

class PickleDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './latents', batch_size: int = 8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.pickle_train = PickleFolder(f'{self.data_dir}/train', samples_per_class=2000, classes_to_use=["DME"])
            self.pickle_val = PickleFolder(f'{self.data_dir}/val', samples_per_class=500)
            
            print("Trainset: ", len(self.pickle_train))
            print("Valset: ", len(self.pickle_val))

        # # Assign test dataset for use in dataloader(s)
        # if stage == 'test' or stage is None:
        #     self.pickle_test = PickleFolder(f'{self.data_dir}/test')

    def train_dataloader(self):
        return DataLoader(self.pickle_train, batch_size=self.batch_size, collate_fn=collate_fn, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.pickle_val, batch_size=self.batch_size, collate_fn=collate_fn)

    # def test_dataloader(self):
    #     return DataLoader(self.pickle_test, batch_size=self.batch_size, collate_fn=collate_fn)
