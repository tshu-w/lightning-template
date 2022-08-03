import os
from datetime import datetime
from typing import Optional


class Logger:
    """Base class for experiment loggers."""

    def __init__(
        self,
        save_dir: str = "./",
        name: str = "lightning_logs",
        version: Optional[str] = None,
    ):
        self._save_dir = save_dir
        self._name = name
        if version is not None:
            self._version = version
        else:
            self._version = datetime.now().strftime("%m-%dT%H%M%S")

    @property
    def save_dir(self) -> str:
        """Return the root directory where experiment logs get saved."""
        return self._save_dir

    @property
    def name(self) -> str:
        """Return the experiment name."""
        return self._name

    @property
    def version(self) -> str:
        """Return the experiment version."""
        return self._version

    @property
    def log_dir(self) -> str:
        """Return the experiment log directory."""
        return os.path.join(self.save_dir, self.name, self.version)
