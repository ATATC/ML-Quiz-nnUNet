from torch import device, nn
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR
from typing import override

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.swin_unet import SwinUnet


class SwinUnetTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 d: device = device('cuda')) -> None:
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, d)
        self.initial_lr = 1e-4
        self.weight_decay = 1e-5

    @override
    def configure_optimizers(self) -> tuple[Optimizer, LRScheduler]:
        optimizer = AdamW(self.network.parameters(),
                          lr=self.initial_lr,
                          weight_decay=self.weight_decay,
                          amsgrad=True)

        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        return optimizer, lr_scheduler

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: list[str] | tuple[str, ...],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return SwinUnet(
            img_size=400,
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            drop_rate=0.0,
            drop_path_rate=0.0,
            use_checkpoint=True,
        )
