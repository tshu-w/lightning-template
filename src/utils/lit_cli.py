import argparse
import os
from typing import Iterable

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


def get_cli_parser():
    # provide cli.parser for shtab.
    #
    # shtab --shell {bash,zsh,tcsh} src.utils.lit_cli.get_cli_parser
    # for more details see https://docs.iterative.ai/shtab/use/#cli-usage
    from jsonargparse import capture_parser

    parser = capture_parser(LitCLI)
    return parser
