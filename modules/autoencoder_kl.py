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


class AutoencoderKL(LightningModule):
    def __init__(self, class_prompts=["CNV", "DME", "DRUSEN", "NORMAL"]):
        super().__init__()

    def forward(self, x):
        # Implement the forward function if necessary.
        pass
    
    def training_step(self, batch, batch_idx):
        pass
    
    def validation_step(self, batch, batch_idx):
        pass

    def _common_step(self, batch, batch_idx, train=True):
        pass

    def configure_optimizers(self):
        pass
    
    def on_training_epoch_start(self) -> None:
        pass

    def on_training_epoch_end(self) -> None:
        pass

    def on_validation_epoch_start(self) -> None:
        pass

    def on_validation_epoch_end(self):
        pass