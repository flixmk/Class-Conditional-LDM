
from lightning.pytorch import callbacks, cli_lightning_logo
from lightning.pytorch.cli import LightningCLI
from modules.pickled_dataset import PickleDataModule
from modules.diffusion import LatentDiffusionModel
from pytorch_lightning.loggers import WandbLogger
import torch

# command: python -W ignore train.py --trainer.devices 2 --trainer.strategy 'ddp' --trainer.accumulate_grad_batches 4 --trainer.logger WandbLogger --trainer.logger.project "foundation_model"
# python -W ignore train.py --trainer.devices 1 --trainer.logger WandbLogger --trainer.logger.project "foundation_model"


def cli_main():
    cli = LightningCLI(
        LatentDiffusionModel,
        PickleDataModule,
        seed_everything_default=1234,
        run=False,  # used to de-activate automatic fitting.
        trainer_defaults={"max_epochs": 1000, 
                          "accelerator": 'gpu',
                          "precision": 'bf16',
                          "check_val_every_n_epoch": 10,
                         "gradient_clip_val":1,},
        save_config_kwargs={"overwrite": True},
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    from huggingface_hub import HfApi, Repository
    from transformers import BertModel, BertTokenizer
    
    for layer in cli.trainer.model.unet.modules():
        layer.float()
    
    cli.trainer.model.unet.save_pretrained("./oct_model_fp32/")


if __name__ == "__main__":
    cli_lightning_logo()
    cli_main()
