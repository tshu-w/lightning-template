import argparse
import os
from collections import defaultdict
from typing import Any, Iterable

import shtab
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI


class LitCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument("-n", "--name", default=None, help="Experiment name")
        parser.add_argument(
            "-d",
            "--debug",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Debug mode",
        )

        for arg in ["num_labels", "task_name"]:
            parser.link_arguments(
                f"data.init_args.{arg}",
                f"model.init_args.{arg}",
                apply_on="instantiate",
            )

    def before_instantiate_classes(self) -> None:
        config = self.config[self.subcommand]
        mode = "debug" if config.debug else self.subcommand

        config.trainer.default_root_dir = os.path.join("results", mode)

        if config.debug:
            self.save_config_callback = None
            config.trainer.logger = None

        logger = config.trainer.logger
        assert logger != True, "should assign trainer.logger with the specific logger."
        if logger:
            loggers = logger if isinstance(logger, Iterable) else [logger]
            for logger in loggers:
                logger.init_args.save_dir = os.path.join(
                    logger.init_args.get("save_dir", "results"), self.subcommand
                )
                if config.name:
                    logger.init_args.name = config.name

    def setup_parser(
        self,
        add_subcommands: bool,
        main_kwargs: dict[str, Any],
        subparser_kwargs: dict[str, Any],
    ) -> None:
        """Initialize and setup the parser, subcommands, and arguments."""
        # move default_config_files to subparser_kwargs
        if add_subcommands:
            default_configs = main_kwargs.pop("default_config_files", None)
            subparser_kwargs = defaultdict(dict, subparser_kwargs)
            for subcmd in self.subcommands():
                subparser_kwargs[subcmd]["default_config_files"] = default_configs

        self.parser = self.init_parser(**main_kwargs)
        shtab.add_argument_to(self.parser, ["-s", "--print-completion"])

        if add_subcommands:
            self._subcommand_method_arguments: dict[str, list[str]] = {}
            self._add_subcommands(self.parser, **subparser_kwargs)
        else:
            self._add_arguments(self.parser)
