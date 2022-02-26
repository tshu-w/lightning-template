import warnings
from functools import partial
from typing import Literal, Optional

from datasets import load_dataset
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

warnings.filterwarnings(
    "ignore", ".*Consider increasing the value of the `num_workers` argument*"
)

TASK_NAME = Literal[
    "cola",
    "sst2",
    "mrpc",
    "qqp",
    "stsb",
    "mnli",
    "qnli",
    "rte",
    "wnli",
    "ax",
]


class GLUEDataModule(LightningDataModule):
    task_text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
        "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
        "ax": ["premise", "hypothesis"],
    }

    glue_task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "stsb": 1,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
        "ax": 3,
    }

    def __init__(
        self,
        task_name: TASK_NAME = "mrpc",
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.task_name = task_name
        self.num_labels = self.glue_task_num_labels[task_name]
        self.text_fields = self.task_text_field_map[task_name]

    def prepare_data(self) -> None:
        # setup first to prevent datasets cache conflicts in multiple processes.
        self.setup()

    def setup(self, stage: Optional[str] = None) -> None:
        if not hasattr(self, "datasets"):
            convert_to_features = self.trainer.model.convert_to_features
            preprocess_fn = partial(self._preprocess, text_fields=self.text_fields)
            preprocess = lambda x: convert_to_features(preprocess_fn(x))

            datasets = load_dataset("glue", self.task_name)
            columns_names = self.text_fields + ["label", "idx"]
            self.datasets = datasets.map(
                preprocess,
                batched=True,
                remove_columns=columns_names,
            )

            self.datasets.set_format(type="torch")

            self.val_splits = [x for x in self.datasets.keys() if "validation" in x]
            self.test_splits = [x for x in self.datasets.keys() if "test" in x]

        self.collate_fn = getattr(self.trainer.model, "collate_fn", None)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.datasets["train"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
            persistent_workers=self.hparams.num_workers > 0,
            shuffle=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_dataloaders = [
            DataLoader(
                dataset=self.datasets[x],
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=self.collate_fn,
                persistent_workers=self.hparams.num_workers > 0,
                shuffle=False,
            )
            for x in self.val_splits
        ]

        return val_dataloaders[0] if len(val_dataloaders) == 1 else val_dataloaders

    def test_dataloader(self) -> EVAL_DATALOADERS:
        test_dataloaders = [
            DataLoader(
                dataset=self.datasets[x],
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=self.collate_fn,
                persistent_workers=self.hparams.num_workers > 0,
                shuffle=False,
            )
            for x in self.test_splits
        ]

        return test_dataloaders[0] if len(test_dataloaders) == 1 else test_dataloaders

    @staticmethod
    def _preprocess(batch, text_fields):
        if len(text_fields) > 1:
            text = list(zip(batch[text_fields[0]], batch[text_fields[1]]))
        else:
            text = batch[text_fields[0]]
        labels = batch["label"]

        return {"text": text, "labels": labels}
