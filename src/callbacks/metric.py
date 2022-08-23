import json
import logging
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.metrics import metrics_to_scalars


class Metric(Callback):
    r"""
    Save logged metrics to ``Trainer.log_dir``.
    """

    def teardown(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        stage: Optional[str] = None,
    ) -> None:
        metrics = {}
        if stage == TrainerFn.FITTING:
            if (
                trainer.checkpoint_callback
                and trainer.checkpoint_callback.best_model_path
            ):
                ckpt_path = trainer.checkpoint_callback.best_model_path
                # inhibit disturbing logging
                logging.getLogger("pytorch_lightning.utilities.distributed").setLevel(
                    logging.WARNING
                )
                logging.getLogger("pytorch_lightning.accelerators.gpu").setLevel(
                    logging.WARNING
                )

                fn_kwargs = {
                    "model": pl_module,
                    "datamodule": trainer.datamodule,
                    "ckpt_path": ckpt_path,
                }

                val_metrics = {}
                if trainer._data_connector._val_dataloader_source.is_defined():
                    trainer.callbacks = []
                    trainer.validate(**fn_kwargs)
                    val_metrics = metrics_to_scalars(trainer.logged_metrics)

                test_metrics = {}
                if trainer._data_connector._test_dataloader_source.is_defined():
                    trainer.callbacks = []
                    trainer.test(**fn_kwargs)
                    test_metrics = metrics_to_scalars(trainer.logged_metrics)

                metrics = {**val_metrics, **test_metrics}
        else:
            metrics = metrics_to_scalars(trainer.logged_metrics)

        if metrics:
            metrics_str = json.dumps(metrics, ensure_ascii=False, indent=2)

            metrics_file = Path(trainer.log_dir) / "metrics.json"
            with metrics_file.open("w") as f:
                f.write(metrics_str)
