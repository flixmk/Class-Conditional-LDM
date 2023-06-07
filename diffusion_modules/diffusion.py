import os
import sys
import numpy as np
import shutil
from typing import Optional, Tuple, List, Dict
from tqdm import tqdm

import torch
import torch.nn.functional as F

from lightning.pytorch import LightningModule

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler

from modules.evaluator import Evaluator

from cleanfid import fid
import timm
import shutil
import wandb
# from svdiff_pytorch import load_unet_for_svdiff
import torch.nn as nn
import itertools
import random

def compute_snr(timesteps, noise_scheduler):
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    snr = (alpha / sigma) ** 2
    return snr

def encode_strings(strings: List[str], tokenizer, text_encoder):
    # strings = ["CNV", "DME", "DRUSEN", "NORMAL"]
    # Create a dictionary to store the encoded strings
    encoded_strings = {}

    for idx, string in enumerate(strings):
        # Tokenize the string and create input tensor
        inputs = tokenizer(
            string, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids
        

        # Encode the string using the CLIPTextModel
        with torch.no_grad():
            encoded = text_encoder(inputs)[0]

        # Save the encoded string in the dictionary
        # check here!
        encoded_strings[idx] = encoded.squeeze(0)
        # encoded_strings[idx] = encoded.last_hidden_state[:, 0].cpu()

    return encoded_strings

def multi_modal_strings(string):

    predefined_strings = [
        f"{string}", 
        f"An OCT scan of {string}", 
        f"A scan using OCT technology depicting the disease: {string}", 
        f"Image that shows an example of {string}", 
        f"Picture that shows an example of the condition {string}",
        f"An image of {string}",
        f"A clinical image presenting {string}",
        f"This OCT scan illustrates {string}",
        f"An image showcasing the condition known as {string}",
        f"A visual representation of {string}",
        f"Photographic depiction of {string}",
        f"An example of an OCT scan showing {string}",
        f"A detailed OCT scan showing the disease: {string}",
        f"This picture illustrates the characteristics of {string}",
        f"Visual documentation of {string} via OCT imaging",
        f"An OCT-captured representation of {string}",
        f"Photographic evidence of the disease called {string}",
        f"An OCT image highlighting {string}",
        f"A high-resolution OCT image depicting {string}",
        f"An in-depth visual representation of {string} via OCT scan",
        f"Graphic evidence of the condition known as {string} using OCT technology",
        ]
    
    # Choose a random string from the predefined set
    chosen_string = random.choice(predefined_strings)
    return chosen_string




class AverageMeter:
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = self.count = self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class LatentDiffusionModel(LightningModule):
    def __init__(self, class_prompts=["CNV", "DME", "DRUSEN", "NORMAL"]):
        super().__init__()
        self.save_hyperparameters()
        self.class_prompts = class_prompts
        self.noise_offset = 0.1
        self.snr_gamma = 5.0
        self.average_meter_train = AverageMeter()
        self.average_meter_val = AverageMeter()
        self.use_svdiff = False
        self.train_text_encoder = False
        self.lr_unet = 1e-5
        self.lr_text_encoder = 1e-6

        # needed constantly
        model_id = "stabilityai/stable-diffusion-2-1-base"
        
        self.unet = self.get_unet()
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        text_encoder_id = "flix-k/custom_model_parts"
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        # text_encoder.gradient_checkpointing_enable()
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        # self.encoded_prompts = encode_strings(self.class_prompts, tokenizer, text_encoder)

        # delete the text encoder and tokenizer to save memory
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if not self.train_text_encoder:
            # self.text_encoder.to("cpu")
            self.text_encoder.requires_grad_(False)
            torch.cuda.empty_cache()

    def get_unet(self):

        if self.use_svdiff:
            unet = load_unet_for_svdiff("stabilityai/stable-diffusion-2-1-base", subfolder="unet")
            unet.enable_xformers_memory_efficient_attention()
            unet.enable_gradient_checkpointing()
            return unet
        else:
            unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="unet")
            unet.enable_xformers_memory_efficient_attention()
            # unet.enable_gradient_checkpointing()

            unet.half()  # convert to half precision

            for layer in unet.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.float()
                if isinstance(layer, nn.GroupNorm):
                    layer.float()
                if isinstance(layer, nn.LayerNorm):
                    layer.float()
            return unet

    def forward(self, x):
        # Implement the forward function if necessary.
        pass
    
    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, train=True)
    
    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, train=False)
    
    def vae_sample(self,list_):
        samples_list = list()
        for distr in list_:
            sample = distr.sample() * 0.18215
            samples_list.append(sample)
        samples_list = torch.stack(samples_list)
        samples_list = samples_list.to(memory_format=torch.contiguous_format).float()

        return samples_list.squeeze(1)

    def _common_step(self, batch, batch_idx, train=True):
        with torch.no_grad():
            latents = batch["latents"] * 0.18215

        noise = torch.randn_like(latents)
        if self.noise_offset:
            noise += self.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
            )
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        # if self.train_text_encoder:


        inputs_strings = [multi_modal_strings(self.class_prompts[i]) for i in batch["classes"]]
        inputs = self.tokenizer(
            inputs_strings, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids.to("cuda")
        encoder_hidden_states = self.text_encoder(inputs)[0]

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        if self.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            snr = compute_snr(timesteps, self.noise_scheduler)
            mse_loss_weights = (
                torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()
        detached_loss = loss.clone().detach_()    
        
        if train:
            for i, param_group in enumerate(self.optimizers().param_groups):
                lr = param_group['lr']
                if i == 0:
                    self.log('lr_unet', lr, logger=True, prog_bar=True)
                else:
                    self.log('lr_text_encoder', lr, logger=True, prog_bar=True)
            self.average_meter_train.update(detached_loss, bsz)
            if self.trainer.global_step % 10 == 0:
                try:
                    wandb.log({"avg_train_loss": self.average_meter_train.avg.item()}, step=self.trainer.global_step)
                except:
                    pass
                self.log("self_logs_avg_train_loss", self.average_meter_train.avg.item(), prog_bar=True, logger=True)
            return loss
        else:
            self.average_meter_val.update(detached_loss, bsz)

    def configure_optimizers(self):
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
        
        if self.train_text_encoder:
            # params_to_optimize = (
            #     itertools.chain(self.unet.parameters(), self.text_encoder.parameters()) if self.train_text_encoder else self.unet.parameters()
            # )
            optimizer = optimizer_cls(
                [
                    {"params": self.unet.parameters(), "lr": self.lr_unet},
                    {"params": self.text_encoder.parameters(), "lr": self.lr_text_encoder}
                ],
                lr=5e-5,  # This will be overridden for each group
                betas=(0.9, 0.999),
                weight_decay=1e-2,
                eps=1e-8,
            )
            
            max_steps = self.trainer.max_epochs * len(self.trainer.datamodule.train_dataloader())

            lr_lambda = lambda step: 1 - step / max_steps

            scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
                'interval': 'step',  # 'step' updates after each training step, 'epoch' updates after each epoch
            }

            return [optimizer], [scheduler]
        else:
            optimizer = optimizer_cls(
                self.unet.parameters(),
                lr=self.lr_unet,
                betas=(0.9, 0.999),
                weight_decay=1e-2,
                eps=1e-8,
            )
            max_steps = self.trainer.max_epochs * len(self.trainer.datamodule.train_dataloader())

            lr_lambda = lambda step: 1 - step / max_steps

            scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
                'interval': 'step',  # 'step' updates after each training step, 'epoch' updates after each epoch
            }
            return [optimizer], [scheduler]

    def on_training_epoch_start(self) -> None:
        pass

    def on_training_epoch_end(self) -> None:
        pass

    def on_validation_epoch_start(self) -> None:
        pass

    def on_validation_epoch_end(self):
        # pass
        self.generate_images(10)
        self.log("avg_val_loss", self.average_meter_val.avg.item(), logger=True)
#         self.unet.to("cpu")
#         torch.cuda.empty_cache()
#         if self.global_rank==0:
#              # load model
#             model_inc = timm.create_model('inception_v3', pretrained=True, num_classes=4)
#             model_inc.load_state_dict(torch.load("./finetuned_best.pt"))
#             model_inc.eval()
#             model_inc = torch.nn.Sequential(*(list(model_inc.children())[:-1]))

#             os.makedirs("./synth_data/ALL", exist_ok=True)

#             for c in self.class_prompts:
#                 for f in os.listdir(f"./synth_data/{c}"):
#                     if ".jpg" in f:
#                         shutil.copy(f"./synth_data/{c}/{f}", f"./synth_data/ALL/{f}")

#             evaluator = Evaluator(real_folder="./val_images/", generated_folder="./synth_data/ALL/", model=model_inc, device=self.device)
#             fid, fid_trained, inception_score, novelty_score = evaluator.run()

#             del model_inc
#             del evaluator
#             torch.cuda.empty_cache()
#             # print(f"FID: {fid}, FID-Trained: {fid_trained}, Inception Score: {inception_score}, Novelty Score: {novelty_score}")
#             # self.eval_images()
#         try:
#             wandb.log({
#                 "FID": fid, 
#                 "FID-Trained": fid_trained, 
#                 "Inception Score": inception_score, 
#                 "Novelty Score": novelty_score
#                 }, 
#                 step=self.trainer.global_step)
#         except:
#             pass
#         # self.log("FID", fid, logger=True)
#         # self.log("Trained", fid_trained, logger=True)
#         # self.log("Inception", inception_score, logger=True)
#         # self.log("Novelty", novelty_score, logger=True)
#         try:
#             wandb.log({"avg_val_loss": self.average_meter_val.avg.item()}, step=self.trainer.global_step)
#         except:
#             pass
        
#         # if self.global_rank==0:
#         #     for class_prompt in self.class_prompts:
#         #         img_list = list()
#         #         for i in range(10):
#         #             img_list.append(wandb.Image(f"./synth_data/{class_prompt}/{class_prompt}-({str(i).zfill(len(str(10)))})-gpu0.jpg"))
#         #             self.logger.log_image(key=f"{class_prompt}", images=img_list)
        
#             # self.log("avg_val_loss", self.average_meter_val.avg.item(), logger=True)
#         self.unet.to(self.device)
        
    def eval_images(self):
        if self.global_rank==0:
            os.makedirs("./synth_data/ALL", exist_ok=True)
            for c in self.class_prompts:
                for f in os.listdir(f"./synth_data/{c}"):
                    shutil.copy(f"./synth_data/{c}/{f}", f"./synth_data/ALL/{f}")
                    
                    
            try:
                fid.make_custom_stats("val", fdir="./val_images/")
            except:
                print("Stats already exist")
                
            score = fid.compute_fid("./synth_data/ALL/", dataset_name="val",
                    mode="clean", dataset_split="custom")
            self.log("FID", score, logger=True)
            

            
            try:
                fid.make_custom_stats("octv3-val", fdir="./val_images/", model=model, model_name="custom")
            except:
                print("Stats already exist")
            
            score = fid.compute_fid("./synth_data/ALL/", dataset_name="octv3-val",
                    mode="clean", dataset_split="custom", model_name="custom", custom_feat_extractor=model_inc)
            self.log("FID-Trained", score, logger=True)
            
            del model
            torch.cuda.empty_cache()
        
    def generate_images(self, num_images):
        pipeline = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            unet=self.unet,
            # vae=AutoencoderKL.from_pretrained("flix-k/custom_model_parts", subfolder="vae_trained_kl"),
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            safety_checker=None,
            )
        pipeline.set_progress_bar_config(disable=True)
        pipeline.enable_xformers_memory_efficient_attention()
        pipeline.to(self.device)
        num_gpus = self.trainer.world_size
        images_per_gpu = num_images // num_gpus
        save_path = "./synth_data/"
        for class_prompt in self.class_prompts:
            if self.global_rank==0:
                print(f"Generating images for class {class_prompt}")
            if save_path is not None:
                save_path_class = save_path + f"/{class_prompt}"
                isExist = os.path.exists(save_path_class) 
                if not isExist:
                    os.makedirs(save_path_class)
            for it in tqdm(range(images_per_gpu)):
                with torch.autocast("cuda"):
                    images = pipeline(
                        prompt = multi_modal_strings(class_prompt),
                        height = 512,
                        width = 512,
                        num_inference_steps = 25,
                        guidance_scale = 5,
                        negative_prompt = None,
                        num_images_per_prompt = 1,
                        ).images
                    for idx, image in enumerate(images):
                        id_num = idx + (it * 1)
                        id = str(id_num).zfill(len(str(num_images)))
                        image.save(f"{save_path_class}/{class_prompt}-({id})-gpu{self.trainer.global_rank}.jpg")

        del pipeline
        torch.cuda.empty_cache()
        
        if self.global_rank==0:
            for class_prompt in self.class_prompts:
                img_list = list()
                for i in range(10):
                    img_list.append(wandb.Image(f"./synth_data/{class_prompt}/{class_prompt}-({str(i).zfill(len(str(10)))})-gpu0.jpg"))
                    self.logger.log_image(key=f"{class_prompt}", images=img_list)
                        
    
    def get_tensors_from_dict(self, class_index: torch.Tensor, encoded_strings: Dict[int, torch.Tensor]) -> torch.Tensor:
        tensors = []

        for elem in class_index:
            if elem.item() in encoded_strings:
                tensors.append(encoded_strings[elem.item()])
            else:
                raise ValueError("The class_prompts are not complete. '{}' is missing.".format(elem))

        # Stack the tensors to create a batch
        batched_tensors = torch.stack(tensors)
        
        return batched_tensors
