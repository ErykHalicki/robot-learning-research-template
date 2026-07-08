"""
Entry point. Builds and runs experiments, resolving checkpoints from wandb when asked.

The execution flow (experiment registry, Hydra runtime.choices -> `_name` stamping, the
tasks loop, and resume/load) is adapted from Boyuan Chen's research template
(https://github.com/buoyancy99/research-template, MIT). The SLURM/cluster submission layer
is intentionally omitted.

Run:  python main.py +name=my_run wandb.entity=<team>
      python main.py +name=smoke wandb.mode=disabled          # no wandb
"""

import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict


def run_local(cfg: DictConfig, checkpoint_path):
    # delayed import so config errors surface before we touch torch/accelerate
    from template.experiments import build_experiment
    from template.utils.wandb import finish_wandb, init_wandb

    # stamp each group's chosen yaml name onto the config, so the registry can look it up
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    choices = OmegaConf.to_container(hydra_cfg.runtime.choices)
    with open_dict(cfg):
        for group in ("experiment", "dataset", "algorithm"):
            if choices.get(group) is not None:
                cfg[group]._name = choices[group]

    output_dir = Path(hydra_cfg.runtime.output_dir)
    print(f"Outputs will be saved to: {output_dir}")
    latest = output_dir.parents[1] / "latest-run"
    latest.unlink(missing_ok=True)
    latest.symlink_to(output_dir, target_is_directory=True)

    experiment = build_experiment(cfg, output_dir, checkpoint_path)

    # one wandb run brackets all tasks (so eval-only runs log too); rank-0 only under DDP
    is_main = os.environ.get("RANK", "0") == "0"
    if is_main:
        init_wandb(cfg)
    # for those searching: this is where we run tasks like 'training', 'validation'
    for task in cfg.experiment.tasks:
        experiment.exec_task(task)
    if is_main:
        finish_wandb()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg: DictConfig):
    from template.utils.ckpt_utils import download_latest_checkpoint, is_run_id

    # a run name is required to start a new run, but not when resuming (the name is kept
    # from the run being resumed)
    if "name" not in cfg and not cfg.get("resume", None):
        raise ValueError("must specify a run name with the command line argument '+name=[name]'")
    if cfg.wandb.mode == "online" and not cfg.wandb.get("entity", None):
        raise ValueError(
            "wandb.mode=online requires wandb.entity (your wandb team/user name). "
            "Set wandb.entity=[entity], or use wandb.mode=offline / wandb.mode=disabled."
        )
    if cfg.wandb.project is None:
        cfg.wandb.project = Path(__file__).parent.name

    resume = cfg.get("resume", None)
    load = cfg.get("load", None)
    if resume and load:
        raise ValueError(
            "specify only one of resume= / load=. Resuming loads the checkpoint from the cloud run."
        )

    # resolve the checkpoint: a local path, or download the latest artifact of a wandb run
    checkpoint_path = None
    if load and not is_run_id(load):
        checkpoint_path = load
    load_id = resume or (load if load and is_run_id(load) else None)
    if load_id:
        run_path = f"{cfg.wandb.entity}/{cfg.wandb.project}/{load_id}"
        checkpoint_path = download_latest_checkpoint(run_path, Path("outputs/downloaded"))

    run_local(cfg, checkpoint_path)


if __name__ == "__main__":
    run()
