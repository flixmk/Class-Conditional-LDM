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
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel

from modules.evaluator import Evaluator

from cleanfid import fid
import timm
import shutil
import wandb
import torch.nn as nn

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
            string, max_length=4, padding="max_length", truncation=True, return_tensors="pt"
        )
        

        # Encode the string using the CLIPTextModel
        with torch.no_grad():
            encoded = text_encoder(**inputs)[0]

        # Save the encoded string in the dictionary
        # check here!
        encoded_strings[idx] = encoded.squeeze(0)
        # encoded_strings[idx] = encoded.last_hidden_state[:, 0].cpu()

    return encoded_strings
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
        self.offset_noise = True
        self.with_prior_preservation = False
        self.snr_gamma = 5.0
        self.average_meter_train = AverageMeter()
        self.average_meter_val = AverageMeter()

        # needed constantly
        self.unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="unet")
        # self.unet = UNet2DConditionModel(
        #                             in_channels=3,
        #                             out_channels=3,
        #                             act_fn="silu",
        #                             attention_head_dim=[
        #                                 5,
        #                                 10,
        #                                 20,
        #                                 20
        #                             ],
        #                             block_out_channels=[
        #                                 320,
        #                                 640,
        #                                 1280,
        #                                 1280
        #                             ],
        #                             center_input_sample=False,
        #                             cross_attention_dim=1024,
        #                             down_block_types=[
        #                                 "CrossAttnDownBlock2D",
        #                                 "CrossAttnDownBlock2D",
        #                                 "CrossAttnDownBlock2D",
        #                                 "DownBlock2D"
        #                             ],
        #                             downsample_padding=1,
        #                             dual_cross_attention=False,
        #                             flip_sin_to_cos=True,
        #                             freq_shift=0,
        #                             layers_per_block=2,
        #                             mid_block_scale_factor=1,
        #                             norm_eps=1e-05,
        #                             norm_num_groups=32,
        #                             num_class_embeds=None,
        #                             only_cross_attention=False,
        #                             sample_size=64,
        #                             up_block_types=[
        #                                 "UpBlock2D",
        #                                 "CrossAttnUpBlock2D",
        #                                 "CrossAttnUpBlock2D",
        #                                 "CrossAttnUpBlock2D"
        #                             ],
        #                             use_linear_projection=True
        # )
        self.noise_scheduler = DDPMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="scheduler")
        self.noise_scheduler.prediction_type = "v_prediction"
        
        self.noice_scheduler = DDPMScheduler(
                                            beta_end= 0.0205, # 0.0205
                                            beta_schedule= "scaled_linear",
                                            beta_start= 0.0015, # 0.0015
                                            clip_sample= False,
                                            num_train_timesteps= 1000,
                                            prediction_type= "v_prediction",
                                            trained_betas= None
                                            )
        
        # self.unet.enable_xformers_memory_efficient_attention()
        self.unet.enable_gradient_checkpointing()

        self.unet.half()  # convert to half precision

        for param in self.unet.parameters():
             print(param.dtype)

        for layer in self.unet.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()
            if isinstance(layer, nn.GroupNorm):
                layer.float()
            if isinstance(layer, nn.LayerNorm):
                layer.float()

        for param in self.unet.parameters():
             print(param.dtype)

        text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="text_encoder")
        text_encoder.gradient_checkpointing_enable()
        tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="tokenizer")
        self.encoded_prompts = encode_strings(self.class_prompts, tokenizer, text_encoder)

        # delete the text encoder and tokenizer to save memory
        text_encoder.to("cpu")
        torch.cuda.empty_cache()
        del text_encoder
        del tokenizer
        torch.cuda.empty_cache()

    def forward(self, x):
        # Implement the forward function if necessary.
        pass
    
    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, train=True)
    
    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, train=False)

    def _common_step(self, batch, batch_idx, train=True):
        latents = batch["latents"].to(self.device) * 0.18215 # vae standard scaling factor

        if self.offset_noise:
            noise = torch.randn_like(latents) + 0.1 * torch.randn(
                latents.shape[0], latents.shape[1], 1, 1, device=latents.device
            )
        else:
            noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        encoder_hidden_states = self.get_tensors_from_dict(batch["labels"], self.encoded_prompts).to(self.device)

        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        # print("timesteps: ", torch.isnan(timesteps).any())
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        # print("noisy_latents: ", torch.isnan(noisy_latents).any())
        # Get the text embedding for conditioning
        encoder_hidden_states = self.get_tensors_from_dict(batch["labels"], self.encoded_prompts).to(self.device)
        # print("encoder_hidden_states: ", torch.isnan(encoder_hidden_states).any())
        # Predict the noise residual
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        # print("model_pred: ", torch.isnan(model_pred).any())
        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        if self.with_prior_preservation:
            # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
            target, target_prior = torch.chunk(target, 2, dim=0)

            # Compute instance loss
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # Compute prior loss
            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

            # Add the prior loss to the instance loss.
            loss = loss + self.prior_loss_weight * prior_loss
        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            # print(model_pred.shape, target.shape)
            # print(type(model_pred), type(target))

        detached_loss = loss.clone().detach_()    
        if train:
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
        optimizer = optimizer_cls(
            self.unet.parameters(),
            lr=1e-5,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-8,
        )
        return optimizer
    
    def on_training_epoch_start(self) -> None:
        pass

    def on_training_epoch_end(self) -> None:
        pass

    def on_validation_epoch_start(self) -> None:
        pass

    def on_validation_epoch_end(self):
        self.generate_images(10)
        self.unet.to("cpu")
        torch.cuda.empty_cache()
        if self.global_rank==0:
             # load model
            model_inc = timm.create_model('inception_v3', pretrained=True, num_classes=4)
            model_inc.load_state_dict(torch.load("./finetuned_best.pt"))
            model_inc.eval()
            model_inc = torch.nn.Sequential(*(list(model_inc.children())[:-1]))

            os.makedirs("./synth_data/ALL", exist_ok=True)

            for c in ["NORMAL", "CNV", "DRUSEN", "DME"]:
                for f in os.listdir(f"./synth_data/{c}"):
                    if ".jpg" in f:
                        shutil.copy(f"./synth_data/{c}/{f}", f"./synth_data/ALL/{f}")

            evaluator = Evaluator(real_folder="./val_images/", generated_folder="./synth_data/ALL/", model=model_inc, device=self.device)
            fid, fid_trained, inception_score, novelty_score = evaluator.run()

            # for class_prompt in self.class_prompts:
            #     fid_class_prompt, fid_trained_class_prompt = evaluator.fid(real_folder=f"./val_images/{class_prompt}", generated_folder=f"./synth_data/{class_prompt}/")
            #     try:
            #         wandb.log({
            #             f"FID-{class_prompt}": fid_class_prompt, 
            #             f"FID-Trained-{class_prompt}": fid_trained_class_prompt, 
            #             }, 
            #             step=self.trainer.global_step)
            #     except:
            #         pass
            del model_inc
            del evaluator
            torch.cuda.empty_cache()
            # print(f"FID: {fid}, FID-Trained: {fid_trained}, Inception Score: {inception_score}, Novelty Score: {novelty_score}")
            # self.eval_images()
        try:
            wandb.log({
                "FID": fid, 
                "FID-Trained": fid_trained, 
                "Inception Score": inception_score, 
                "Novelty Score": novelty_score
                }, 
                step=self.trainer.global_step)
        except:
            pass
        # self.log("FID", fid, logger=True)
        # self.log("Trained", fid_trained, logger=True)
        # self.log("Inception", inception_score, logger=True)
        # self.log("Novelty", novelty_score, logger=True)
        try:
            wandb.log({"avg_val_loss": self.average_meter_val.avg.item()}, step=self.trainer.global_step)
        except:
            pass
        
        if self.global_rank==0:
            for class_prompt in self.class_prompts:
                img_list = list()
                for i in range(10):
                    img_list.append(wandb.Image(f"./synth_data/{class_prompt}/{class_prompt}-({str(i).zfill(len(str(10)))})-gpu0.jpg"))
                    self.logger.log_image(key=f"{class_prompt}@{self.trainer.global_step}", images=img_list)
        
            # self.log("avg_val_loss", self.average_meter_val.avg.item(), logger=True)
        self.unet.to(self.device)
        
    def generate_images(self, num_images):
        pipeline = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            unet=self.unet,
            # vae=AutoencoderKL.from_pretrained("flix-k/custom_model_parts", subfolder="vae"),
            torch_dtype=torch.float16,
            safety_checker=None,
            )
        pipeline.set_progress_bar_config(disable=True)
        # pipeline.enable_xformers_memory_efficient_attention()
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
                        prompt = class_prompt,
                        height = 512,
                        width = 512,
                        num_inference_steps = 25,
                        guidance_scale = 7.5,
                        negative_prompt = None,
                        num_images_per_prompt = 1,
                        ).images
                    for idx, image in enumerate(images):
                        id_num = idx + (it * 1)
                        id = str(id_num).zfill(len(str(num_images)))
                        image.save(f"{save_path_class}/{class_prompt}-({id})-gpu{self.trainer.global_rank}.jpg")

        del pipeline
        torch.cuda.empty_cache()
                        
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