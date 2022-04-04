from typing import Optional

import pytorch_lightning
from pytorch_lightning.loggers import csv_logs

from . import base


class CSVLogger(base.LightningLoggerBase, csv_logs.CSVLogger):
    def __init__(
        self,
        save_dir: str = "./",
        name: str = "lightning_logs",
        version: Optional[str] = None,
        prefix: str = "",
        flush_logs_every_n_steps: int = 100,
    ):
        super().__init__(save_dir, name, version)

        self._prefix = prefix
        self._experiment = None
        self._flush_logs_every_n_steps = flush_logs_every_n_steps


pytorch_lightning.loggers.CSVLogger = CSVLogger
