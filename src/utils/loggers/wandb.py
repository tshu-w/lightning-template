from typing import Optional, Union

import pytorch_lightning
from pytorch_lightning.loggers import wandb as wandb_logger
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import (
    _WANDB_GREATER_EQUAL_0_10_22,
    _WANDB_GREATER_EQUAL_0_12_10,
)
from pytorch_lightning.utilities.rank_zero import rank_zero_warn

try:
    import wandb
    from wandb.wandb_run import Run
except ModuleNotFoundError:
    # needed for test mocks, these tests shall be updated
    wandb, Run = None, None

from . import base


class WandbLogger(base.LightningLoggerBase, wandb_logger.WandbLogger):
    def __init__(
        self,
        save_dir: str = "./",
        name: str = "lightning_logs",
        version: Optional[str] = None,
        offline: Optional[bool] = False,
        id: Optional[str] = None,
        anonymous: Optional[bool] = None,
        project: Optional[str] = None,
        log_model: Union[str, bool] = False,
        experiment=None,
        prefix: Optional[str] = "",
        **kwargs,
    ):
        if wandb is None:
            raise ModuleNotFoundError(
                "You want to use `wandb` logger which is not installed yet,"
                " install it with `pip install wandb`."  # pragma: no-cover
            )

        if offline and log_model:
            raise MisconfigurationException(
                f"Providing log_model={log_model} and offline={offline} is an invalid configuration"
                " since model checkpoints cannot be uploaded in offline mode.\n"
                "Hint: Set `offline=False` to log your model."
            )

        if log_model and not _WANDB_GREATER_EQUAL_0_10_22:
            rank_zero_warn(
                f"Providing log_model={log_model} requires wandb version >= 0.10.22"
                " for logging associated model metadata.\n"
                "Hint: Upgrade with `pip install --upgrade wandb`."
            )

        super().__init__(save_dir, name, version)

        self._offline = offline
        self._log_model = log_model
        self._prefix = prefix
        self._experiment = experiment
        self._logged_model_time = {}
        self._checkpoint_callback = None
        # set wandb init arguments
        anonymous_lut = {True: "allow", False: None}
        self._wandb_init = dict(
            name=name,
            project=project,
            id=version or id,
            dir=save_dir,
            resume="allow",
            anonymous=anonymous_lut.get(anonymous, anonymous),
        )
        self._wandb_init.update(**kwargs)
        # start wandb run (to create an attach_id for distributed modes)
        if _WANDB_GREATER_EQUAL_0_12_10:
            wandb.require("service")
            _ = self.experiment


pytorch_lightning.loggers.WandbLogger = WandbLogger
