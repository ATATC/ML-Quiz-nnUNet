from torch import device as t_device, nn
from torch._dynamo import OptimizedModule
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.swin_unet import SwinUnet


class SwinUnetTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: t_device = t_device("cuda")) -> None:
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-4
        self.weight_decay = 1e-5

    def configure_optimizers(self) -> tuple[Optimizer, LRScheduler]:
        optimizer = AdamW(self.network.parameters(),
                          lr=self.initial_lr,
                          weight_decay=self.weight_decay,
                          amsgrad=True)

        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        return optimizer, lr_scheduler

    def set_deep_supervision_enabled(self, enabled: bool) -> None:
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        mod.deep_supervision = enabled

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: list[str] | tuple[str, ...],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        print(f"Building network architecture: {num_input_channels} in, {num_output_channels} out")
        return SwinUnet(
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            drop_rate=0.0,
            drop_path_rate=0.0,
            use_checkpoint=True,
        )
