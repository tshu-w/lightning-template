from typing import Optional

import pytorch_lightning
from pytorch_lightning.loggers import tensorboard
from pytorch_lightning.utilities.cloud_io import get_filesystem

from . import base


class TensorBoardLogger(base.LightningLoggerBase, tensorboard.TensorBoardLogger):
    def __init__(
        self,
        save_dir: str = "./",
        name: str = "lightning_logs",
        version: Optional[str] = None,
        log_graph: bool = False,
        default_hp_metric: bool = True,
        prefix: str = "",
        sub_dir: Optional[str] = None,
        **kwargs
    ):
        super().__init__(save_dir, name, version)

        self._sub_dir = sub_dir
        self._log_graph = log_graph
        self._default_hp_metric = default_hp_metric
        self._prefix = prefix
        self._fs = get_filesystem(save_dir)

        self._experiment = None
        self.hparams = {}
        self._kwargs = kwargs


pytorch_lightning.loggers.TensorBoardLogger = TensorBoardLogger
