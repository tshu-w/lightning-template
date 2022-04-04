import os
from contextlib import contextmanager
from typing import Optional

import pytorch_lightning
from pytorch_lightning import __version__
from pytorch_lightning.loggers import neptune as neptune_logger
from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.loggers.neptune import (
    _INTEGRATION_VERSION_KEY,
    _LEGACY_NEPTUNE_INIT_KWARGS,
    _LEGACY_NEPTUNE_LOGGER_KWARGS,
    _NEPTUNE_AVAILABLE,
    _NEPTUNE_GREATER_EQUAL_0_9,
    neptune,
)

if _NEPTUNE_AVAILABLE and _NEPTUNE_GREATER_EQUAL_0_9:
    try:
        from neptune import new as neptune
        from neptune.new.exceptions import (
            NeptuneLegacyProjectException,
            NeptuneOfflineModeFetchException,
        )
        from neptune.new.run import Run
        from neptune.new.types import File as NeptuneFile
    except ModuleNotFoundError:
        import neptune
        from neptune.exceptions import (
            NeptuneLegacyProjectException,
            NeptuneOfflineModeFetchException,
        )
        from neptune.run import Run
        from neptune.types import File as NeptuneFile
else:
    # needed for test mocks, and function signatures
    neptune, Run, NeptuneFile = None, None, None

from . import base


@contextmanager
def chdir(path):
    cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(cwd)


class NeptuneLogger(base.LightningLoggerBase, neptune_logger.NeptuneLogger):
    def __init__(
        self,
        save_dir: str = "./",
        name: str = "lightning_logs",
        version: Optional[str] = None,
        project: Optional[str] = None,
        api_key: Optional[str] = None,
        run: Optional[str] = None,
        custom_run_id: Optional[str] = None,
        log_model_checkpoints: Optional[bool] = False,
        prefix: str = "",
        **neptune_run_kwargs,
    ) -> None:
        self._verify_input_arguments(
            project, api_key, name, run, custom_run_id, neptune_run_kwargs
        )
        if neptune is None:
            raise ModuleNotFoundError(
                "You want to use the `Neptune` logger which is not installed yet, install it with"
                " `pip install neptune-client`."
            )
        super().__init__(save_dir, name, version)

        self._project = project
        self._api_key = api_key
        self._run_name = name
        self._run_short_id = run
        self._custom_run_id = version or custom_run_id
        self._neptune_run_kwargs = neptune_run_kwargs

        self._log_model_checkpoints = log_model_checkpoints
        self._prefix = prefix

        self._run_instance = None

    @property
    def _neptune_init_args(self):
        args = self._neptune_run_kwargs

        if self._project is not None:
            args["project"] = self._project

        if self._api_key is not None:
            args["api_token"] = self._api_key

        if self._run_name is not None:
            args["name"] = self._run_name

        if self._run_short_id is not None:
            args["run"] = self._run_short_id

        if self._custom_run_id is not None:
            args["custom_run_id"] = self._custom_run_id

        return args

    @staticmethod
    def _verify_input_arguments(
        project: Optional[str],
        api_key: Optional[str],
        name: Optional[str],
        run: Optional[str],
        custom_run_id: Optional[str],
        neptune_run_kwargs: dict,
    ):
        legacy_kwargs_msg = (
            "Following kwargs are deprecated: {legacy_kwargs}.\n"
            "If you are looking for the Neptune logger using legacy Python API,"
            " it's still available as part of neptune-contrib package:\n"
            "  - https://docs-legacy.neptune.ai/integrations/pytorch_lightning.html\n"
            "The NeptuneLogger was re-written to use the neptune.new Python API\n"
            "  - https://neptune.ai/blog/neptune-new\n"
            "  - https://docs.neptune.ai/integrations-and-supported-tools/model-training/pytorch-lightning\n"
            "You should use arguments accepted by either NeptuneLogger.init() or neptune.init()"
        )

        # check if user used legacy kwargs expected in `NeptuneLegacyLogger`
        used_legacy_kwargs = [
            legacy_kwarg
            for legacy_kwarg in neptune_run_kwargs
            if legacy_kwarg in _LEGACY_NEPTUNE_INIT_KWARGS
        ]
        if used_legacy_kwargs:
            raise ValueError(legacy_kwargs_msg.format(legacy_kwargs=used_legacy_kwargs))

        # check if user used legacy kwargs expected in `NeptuneLogger` from neptune-pytorch-lightning package
        used_legacy_neptune_kwargs = [
            legacy_kwarg
            for legacy_kwarg in neptune_run_kwargs
            if legacy_kwarg in _LEGACY_NEPTUNE_LOGGER_KWARGS
        ]
        if used_legacy_neptune_kwargs:
            raise ValueError(
                legacy_kwargs_msg.format(legacy_kwargs=used_legacy_neptune_kwargs)
            )

        # check if user passed redundant neptune.init arguments when passed run
        any_neptune_init_arg_passed = (
            any(arg is not None for arg in [project, api_key, name, custom_run_id])
            or neptune_run_kwargs
        )
        if run is not None and any_neptune_init_arg_passed:
            raise ValueError(
                "When an already initialized run object is provided"
                " you can't provide other neptune.init() parameters.\n"
            )

    def __setstate__(self, state):
        self.__dict__ = state
        with chdir(self._save_dir):
            self._run_instance = neptune.init(**self._neptune_init_args)

    @property
    @rank_zero_experiment
    def run(self) -> Run:
        try:
            if not self._run_instance:
                with chdir(self._save_dir):
                    self._run_instance = neptune.init(**self._neptune_init_args)
                self._retrieve_run_data()
                # make sure that we've log integration version for newly created
                self._run_instance[_INTEGRATION_VERSION_KEY] = __version__

            return self._run_instance
        except NeptuneLegacyProjectException as e:
            raise TypeError(
                f"Project {self._project_name} has not been migrated to the new structure."
                " You can still integrate it with the Neptune logger using legacy Python API"
                " available as part of neptune-contrib package:"
                " https://docs-legacy.neptune.ai/integrations/pytorch_lightning.html\n"
            ) from e


pytorch_lightning.loggers.NeptuneLogger = NeptuneLogger
