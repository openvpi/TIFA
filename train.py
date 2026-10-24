import pathlib

import click

from lib import logging
from lib.cli import DefaultGroup
from lib.config.schema import ConfigurationScope


def shared_options(func):
    options = [
        click.option(
            "--config", type=click.Path(
                exists=True, dir_okay=False, file_okay=True, readable=True, path_type=pathlib.Path
            ),
            required=True,
            help="Path to the configuration file."
        ),
        click.option(
            "--override", multiple=True,
            type=click.STRING, required=False,
            help="Override configuration values in dotlist format."
        ),
        click.option(
            "--work-dir", type=click.Path(
                dir_okay=True, file_okay=False, path_type=pathlib.Path
            ),
            required=False, default=pathlib.Path(__file__).parent / "experiments",
            show_default=True,
            help="Path to the working directory. The experiment subdirectory will be created here."
        ),
        click.option(
            "--exp-name", type=click.STRING,
            required=True,
            help="Experiment name. Checkpoints will be saved in subdirectory with this name."
        ),
        click.option(
            "--log-dir", type=click.Path(
                dir_okay=True, file_okay=False, path_type=pathlib.Path
            ),
            required=False,
            help="Directory to save logs. If not specified, logs will be saved in the checkpoints directory."
        ),
        click.option(
            "--restart", is_flag=True, default=False,
            help="Ignore existing checkpoints and start new training."
        ),
        click.option(
            "--resume-from", type=click.Path(
                exists=True, dir_okay=False, file_okay=True, readable=True, path_type=pathlib.Path
            ),
            required=False,
            help="Resume training from this specific checkpoint."
        ),
    ]
    for option in options[::-1]:
        func = option(func)
    return func


def _run_training(
        config: pathlib.Path, override: list[str],
        exp_name: str, work_dir: pathlib.Path,
        log_dir: pathlib.Path,
        restart: bool,
        resume_from: pathlib.Path,
        scope: int,
        pl_module_cls,
):
    from lightning_utilities.core.rank_zero import rank_zero_info
    from training.api import (
        load_config_for_training,
        find_latest_checkpoints,
        train_model,
    )

    config = load_config_for_training(config, scope=scope, overrides=override)
    ckpt_save_dir = work_dir / exp_name
    if log_dir is None:
        log_save_dir = ckpt_save_dir
    else:
        log_save_dir = log_dir / exp_name
    if not restart and resume_from is None:
        latest_checkpoints = find_latest_checkpoints(ckpt_save_dir, candidate_tags=[
            ckpt_config.tag
            for ckpt_config in config.training.trainer.checkpoints
            if not ckpt_config.weights_only
        ])
        if len(latest_checkpoints) > 1:
            raise ValueError(
                f"Cannot perform auto resuming because multiple latest checkpoints were found:\n"
                + "\n".join(f"  {ckpt}" for ckpt in latest_checkpoints)
                + "\nPlease manually choose a specific checkpoint using --resume-from."
            )
        elif len(latest_checkpoints) == 1:
            resume_from = latest_checkpoints[0]

    train_model(
        config=config, pl_module_cls=pl_module_cls,
        ckpt_save_dir=ckpt_save_dir, log_save_dir=log_save_dir,
        resume_from=resume_from
    )
    logging.success("Training completed.", callback=rank_zero_info)


@click.group(cls=DefaultGroup, help="Train a model.")
def main():
    pass


@main.default_command(help="Train a supervised forced-alignment model.")
@shared_options
def supervised(
        config, override,
        exp_name, work_dir,
        log_dir,
        restart,
        resume_from,
):
    from training.fa_module import ForcedAlignmentModule
    _run_training(
        config=config, override=override,
        exp_name=exp_name, work_dir=work_dir,
        log_dir=log_dir,
        restart=restart, resume_from=resume_from,
        scope=ConfigurationScope.FA,
        pl_module_cls=ForcedAlignmentModule,
    )


@main.command(name="ved", help="Train Voice Event Detector.")
@shared_options
def ved(config, override, exp_name, work_dir, log_dir, restart, resume_from):
    from training.ved_module import VoiceEventDetectorModule
    _run_training(
        config=config, override=override,
        exp_name=exp_name, work_dir=work_dir, log_dir=log_dir,
        restart=restart, resume_from=resume_from,
        scope=ConfigurationScope.VED, pl_module_cls=VoiceEventDetectorModule,
    )


# @main.command(name="ssl", help="Train a self-supervised model.")
# @shared_options
def ssl(
        config, override,
        exp_name, work_dir,
        log_dir,
        restart,
        resume_from,
):
    from training.fa_ssl_module import ForcedAlignmentSSLModule
    _run_training(
        config=config, override=override,
        exp_name=exp_name, work_dir=work_dir,
        log_dir=log_dir,
        restart=restart, resume_from=resume_from,
        scope=ConfigurationScope.FA_SSL,
        pl_module_cls=ForcedAlignmentSSLModule,
    )


if __name__ == '__main__':
    main()
