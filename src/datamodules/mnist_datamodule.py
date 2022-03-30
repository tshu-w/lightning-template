from typing import Optional

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


class MNISTDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.transforms = transforms.ToTensor()
        self.data = {}

    def prepare_data(self) -> None:
        MNIST(self.hparams.data_dir, train=True, download=True)
        MNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.data:
            dataset = MNIST(
                self.hparams.data_dir, train=True, transform=self.transforms
            )
            self.data["train"], self.data["val"] = random_split(dataset, [55000, 5000])

            self.data["test"] = MNIST(
                self.hparams.data_dir, train=False, transform=self.transforms
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.data["train"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
            shuffle=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.data["val"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
            shuffle=False,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.data["test"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
            shuffle=False,
        )
