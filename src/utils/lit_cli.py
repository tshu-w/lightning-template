import os
from collections.abc import Iterable

from lightning.pytorch.cli import LightningArgumentParser, LightningCLI


class LitCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        for arg in ["num_labels", "task_name"]:
            parser.link_arguments(
                f"data.init_args.{arg}",
                f"model.init_args.{arg}",
                apply_on="instantiate",
            )

    def before_instantiate_classes(self) -> None:
        config = self.config[self.subcommand]

        default_root_dir = config.trainer.default_root_dir
        logger = config.trainer.logger
        if logger and logger is not True:
            loggers = logger if isinstance(logger, Iterable) else [logger]
            for logger in loggers:
                logger.init_args.save_dir = os.path.join(
                    default_root_dir, self.subcommand
                )


def lit_cli():
    LitCLI(
        parser_kwargs={
            cmd: {
                "default_config_files": ["configs/presets/default.yaml"],
            }
            for cmd in ["fit", "validate", "test"]
        },
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    lit_cli()
