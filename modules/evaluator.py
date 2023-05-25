import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.nn.functional import kl_div
from sklearn.neighbors import NearestNeighbors
import numpy as np
from torchvision.io import read_image
from torch.utils.data import Dataset
from PIL import Image
from torchmetrics.image.inception import InceptionScore
import timm
from cleanfid import fid
from tqdm import tqdm

class ImageFolderFlat(Dataset):
    def __init__(self, folder, transform=None, extensions=("jpg", "jpeg", "png", "bmp", "webp", "tiff")):
        self.folder = folder
        self.transform = transform
        self.extensions = extensions
        self.image_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(self.extensions)]
        # read all images from self.image_paths
        self.images = []
        for image_path in tqdm(self.image_paths):
            image = Image.open(image_path)
            image = image.convert("RGB")
            if self.transform:
                image = self.transform(image)
            self.images.append(image)

    def __getitem__(self, index):
        image = self.images[index]
        return image

    def __len__(self):
        return len(self.image_paths)

class Evaluator:
    def __init__(self, real_folder, generated_folder, model=None, device=None):
        self.real_folder = real_folder
        self.generated_folder = generated_folder
        self.device = device
        print(self.device)
        # self.inception_model = models.inception_v3(pretrained=True).eval()
        # self.inception_model.to("cuda")

#         self.model = model
#         self.model.to("cuda")
        
        self.model = timm.create_model('inception_v3', pretrained=True, num_classes=4).to(self.device)
        self.model.load_state_dict(torch.load("./finetuned_best.pt"))
        self.model.eval()
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))

    def load_images(self, folder):
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ])
        dataset = ImageFolderFlat(folder, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
        return dataloader
    
    # def compute_fid(self, real_features, generated_features):
    #     mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    #     mu2, sigma2 = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)
    #     return fid_score.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    
    def compute_fid(self):
        try:
            fid.make_custom_stats("val", fdir="./val_images/ALL/")
        except:
            print("Stats already exist")
            
        score = fid.compute_fid(self.generated_folder, dataset_name="val",
                mode="clean", dataset_split="custom")

        try:
            fid.make_custom_stats("octv3-val", fdir="./val_images/ALL/", model=self.model, model_name="custom", device=self.device,)
        except:
            print("Stats already exist")
        
        score_trained = fid.compute_fid(self.generated_folder, dataset_name="octv3-val",
                mode="clean", dataset_split="custom", model_name="custom", custom_feat_extractor=self.model, device=self.device,)

        # score_trained = fid.compute_fid(
        #     self.generated_folder, 
        #     self.real_folder,
        #     mode="clean",
        #     custom_feat_extractor=self.model,
        #     device=self.device,
        # )
        return score, score_trained
    
    def compute_fid_from_folders(self, real_folder, generated_folder):
            
        score = fid.compute_fid(generated_folder, real_folder, mode="clean",)
        
        score_trained = fid.compute_fid(generated_folder, real_folder, 
                mode="clean", custom_feat_extractor=self.model, device=self.device,)

        # score_trained = fid.compute_fid(
        #     self.generated_folder, 
        #     self.real_folder,
        #     mode="clean",
        #     custom_feat_extractor=self.model,
        #     device=self.device,
        # )
        return score, score_trained


    def compute_inception_score(self, img_list):
        inception = InceptionScore(feature=self.model)
        image_tensor = torch.cat(img_list, dim=0)
        inception.update(image_tensor)
        inception_score, std = inception.compute()
        return inception_score, std

    def compute_novelty_score(self, real_features, generated_features):
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(real_features)
        distances, _ = nbrs.kneighbors(generated_features)
        novelty_score = np.mean(distances)
        return novelty_score
    
    def run(self):

#         real_dataloader = self.load_images(self.real_folder)
#         generated_dataloader = self.load_images(self.generated_folder)

#         real_features_list = []
#         generated_features_list = []
        
#         image_list = []
            
            

#         with torch.no_grad():
#             for real_images in tqdm(real_dataloader):
#                 # real_images = real_images.to(self.device)
#                 # real_features = self.model(real_images)
#                 # real_features_list.append(real_features.cpu().numpy())
#                 pass

#             for generated_images in tqdm(generated_dataloader):
#                 generated_images = generated_images.to(self.device)
#                 # generated_features = self.model(generated_images)
#                 # generated_features_list.append(generated_features.cpu().numpy())
                
#                 generated_images = generated_images.to(torch.uint8).float()
#                 image_list.append(generated_images)

#         # real_features = np.concatenate(real_features_list, axis=0)
#         # generated_features = np.concatenate(generated_features_list, axis=0)
        
        novelty_score = 0
        inception_score = 0
#         # novelty_score = self.compute_novelty_score(real_features, generated_features)
#         inception_score, std = self.compute_inception_score(image_list)
        fid, fid_trained = self.compute_fid()

        return fid, fid_trained, inception_score, novelty_score
    
    def fid(self, real_folder, generated_folder):
        fid, fid_trained = self.compute_fid_from_folders(real_folder, generated_folder)
        return fid, fid_trained
