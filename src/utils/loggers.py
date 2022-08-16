import os
from datetime import datetime

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger


@property
def log_dir(self) -> str:
    if self.loggers:
        dirpath = self.loggers[0].log_dir
    else:
        dirpath = self.default_root_dir

    dirpath = self.strategy.broadcast(dirpath)
    return dirpath


Trainer.log_dir = log_dir


def __resolve_ckpt_dir(self, trainer: Trainer) -> None:
    """Determines model checkpoint save directory at runtime. References attributes from the trainer's logger
    to determine where to save checkpoints. The base path for saving weights is set in this priority:
    1.  Checkpoint callback's path (if passed in)
    2.  The default_root_dir from trainer if trainer has no logger
    3.  The log_dir from trainer, if trainer has logger
    The base path gets extended with logger name and version (if these are available)
    and subfolder "checkpoints".
    """
    if self.dirpath is not None:
        return  # short circuit

    if trainer.loggers:
        ckpt_path = os.path.join(trainer.log_dir, "checkpoints")
    else:
        ckpt_path = os.path.join(trainer.default_root_dir, "checkpoints")

    ckpt_path = trainer.strategy.broadcast(ckpt_path)

    self.dirpath = ckpt_path


ModelCheckpoint._ModelCheckpoint__resolve_ckpt_dir = __resolve_ckpt_dir


@property
def TensorBoardLogger_version(self) -> str:
    """Get the experiment version.

    Returns:
        The experiment version if specified else current timestamp.
    """
    if self._version is None:
        self._version = datetime.now().strftime("%m-%dT%H%M%S")

    return self._version


TensorBoardLogger.version = TensorBoardLogger_version


@property
def WandbLogger_log_dir(self) -> str:
    if hasattr(self, "_log_dir"):
        return self._log_dir

    save_dir = self._save_dir
    name = self.experiment.name
    version = self.experiment.id
    self._log_dir = os.path.join(save_dir, name, version)

    return self._log_dir


WandbLogger.log_dir = WandbLogger_log_dir
