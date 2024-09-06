import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import Accuracy, MetricCollection


class MNISTModel(L.LightningModule):
    def __init__(
        self,
        input_size: int = 28 * 28,
        hidden_dim: int = 128,
        output_size: int = 10,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.l1 = torch.nn.Linear(input_size, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, output_size)

        metrics = MetricCollection({"acc": Accuracy(task="multiclass", num_classes=10)})
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))

        return x

    def shared_step(self, batch, step: str) -> STEP_OUTPUT | None:
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        metrics = getattr(self, f"{step}_metrics")
        metrics(preds, y)

        self.log(f"{step}/loss", loss)
        self.log_dict(metrics, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx: int) -> STEP_OUTPUT:
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> STEP_OUTPUT | None:
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx: int) -> STEP_OUTPUT | None:
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.hparams.learning_rate)

    def configure_callbacks(self):
        callbacks_kargs = {"monitor": "val/acc", "mode": "max"}
        early_stopping = EarlyStopping(patience=5, **callbacks_kargs)
        model_checkpoint = ModelCheckpoint(**callbacks_kargs)
        return [early_stopping, model_checkpoint]
