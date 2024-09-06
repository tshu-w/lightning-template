from functools import partial
from typing import Any

import evaluate
import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizer,
    get_scheduler,
)


class GLUETransformer(L.LightningModule):
    def __init__(
        self,
        task_name: str,
        model_name_or_path: str,
        num_labels: int,
        max_length: int | None = None,
        weight_decay: float = 0.0,
        learning_rate: float = 2e-5,
        scheduler_type: str = "linear",
        warmup_steps: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.convert_to_features = partial(
            self._convert_to_features, tokenizer=tokenizer, max_length=max_length
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path
        )
        self.metric = evaluate.load("glue", task_name)

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, batch):
        return self.model.forward(**batch)

    def shared_step(self, batch) -> STEP_OUTPUT | None:
        output = self.forward(batch)
        loss, logits = output.loss, output.logits
        labels = batch["labels"]

        if self.hparams.num_labels >= 1:
            preds = torch.argmax(logits, dim=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        return {"loss": loss, "preds": preds, "labels": labels}

    def training_step(
        self, batch, batch_idx: int, dataloader_idx: int | None = None
    ) -> STEP_OUTPUT:
        return self.shared_step(batch)

    def validation_step(
        self, batch, batch_idx: int, dataloader_idx: int | None = None
    ) -> STEP_OUTPUT | None:
        output = self.shared_step(batch)
        self.validation_step_outputs.append(output)
        return output

    def test_step(
        self, batch, batch_idx: int, dataloader_idx: int | None = None
    ) -> STEP_OUTPUT | None:
        output = self.shared_step(batch)
        self.test_step_outputs.append(output)
        return output

    def shared_epoch_end(self, outputs, step: str) -> None:
        if hasattr(self.trainer.datamodule, f"{step}_splits"):
            splits = getattr(self.trainer.datamodule, f"{step}_splits")
            if len(splits) > 1:
                for i, output in enumerate(outputs):
                    split = splits[i].split("_")[-1]
                    preds = torch.cat([x["preds"] for x in output])
                    labels = torch.cat([x["labels"] for x in output])
                    loss = torch.stack([x["loss"] for x in output]).mean()

                    split_metrics = {
                        f"{step}/{split}_{k}": v
                        for k, v in self.metric.compute(
                            predictions=preds, references=labels
                        ).items()
                    }

                    self.log(f"{step}/{split}_loss", loss)
                    self.log_dict(split_metrics, prog_bar=True)

                return loss

        preds = torch.cat([x["preds"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs])
        loss = torch.stack([x["loss"] for x in outputs]).mean()

        metrics = {
            f"{step}/{k}": v
            for k, v in self.metric.compute(
                predictions=preds, references=labels
            ).items()
        }

        self.log(f"{step}/loss", loss)
        self.log_dict(metrics, prog_bar=True)

    def on_training_epoch_end(self) -> None:
        self.shared_epoch_end(self.training_step_outputs, "train")

    def on_validation_epoch_end(self) -> None:
        self.shared_epoch_end(self.validation_step_outputs, "val")
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self) -> None:
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
        )

        scheduler = get_scheduler(
            self.hparams.scheduler_type,
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

    @staticmethod
    def _convert_to_features(
        batch: dict[str, list] | list[Any],
        tokenizer: PreTrainedTokenizer,
        max_length: int | None = None,
    ) -> dict | Any:
        features = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        features["labels"] = batch["labels"]

        return features
