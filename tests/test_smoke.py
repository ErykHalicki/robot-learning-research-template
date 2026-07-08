import os

import pytest
from omegaconf import OmegaConf

from template.experiments.smoke import SmokeExperiment


def _cfg(conditioning=("observation",)):
    return OmegaConf.create(
        {
            "algorithm": {
                "name": "dummy",
                "obs_dim": 8,
                "target_dim": 2,
                "hidden": 16,
                "conditioning": list(conditioning),
                "predict": ["target"],
            },
            "dataset": {
                "backend": "dummy",
                "n": 128,
                "obs_dim": 8,
                "target_dim": 2,
                "weight_seed": 0,
                "seed": 0,
            },
            "eval": {
                "backend": "dummy",
                "n": 128,
                "obs_dim": 8,
                "target_dim": 2,
                "weight_seed": 0,   # same task as the dataset
                "seed": 1000,       # held-out observations
                "n_eval": 64,
            },
            "experiment": {
                "tasks": ["training", "validation"],
                "batch_size": 32,
                "lr": 1e-3,
                "epochs": 2,
                "log_every": 5,
                "mixed_precision": "no",
            },
        }
    )


def test_smoke(tmp_path):
    os.environ["WANDB_MODE"] = "disabled"
    exp = SmokeExperiment(_cfg(), tmp_path)
    exp.exec_task("training")
    exp.exec_task("validation")
    assert (tmp_path / "model.pt").exists()


def test_modality_handshake_fails(tmp_path):
    os.environ["WANDB_MODE"] = "disabled"
    exp = SmokeExperiment(_cfg(conditioning=("observation", "language")), tmp_path)
    with pytest.raises(ValueError, match="lacks modalities"):
        exp.exec_task("training")


def test_exec_task_unknown_raises(tmp_path):
    os.environ["WANDB_MODE"] = "disabled"
    exp = SmokeExperiment(_cfg(), tmp_path)
    with pytest.raises(ValueError, match="not defined"):
        exp.exec_task("nonexistent")


def test_resume_loads_checkpoint(tmp_path):
    os.environ["WANDB_MODE"] = "disabled"
    SmokeExperiment(_cfg(), tmp_path).exec_task("training")
    ckpt = tmp_path / "model.pt"

    # resume: load weights + optimizer, keep training
    resumed = SmokeExperiment(_cfg(), tmp_path, ckpt_path=ckpt)
    resumed.exec_task("training")
    assert ckpt.exists()


def test_training_only_experiment(tmp_path):
    os.environ["WANDB_MODE"] = "disabled"
    from template.experiments.base import BaseExperiment
    from template.experiments.training.base import TrainingMixin

    class TrainOnly(TrainingMixin, BaseExperiment):
        pass

    exp = TrainOnly(_cfg(), tmp_path)
    exp.exec_task("training")
    assert (tmp_path / "model.pt").exists()

    # a train-only experiment has no validation task; asking for it is a clean error
    with pytest.raises(ValueError, match="not defined"):
        exp.exec_task("validation")


def test_validation_only_loads_checkpoint(tmp_path):
    os.environ["WANDB_MODE"] = "disabled"
    SmokeExperiment(_cfg(), tmp_path).exec_task("training")
    ckpt = tmp_path / "model.pt"

    # eval-only run: no training this session, so validation loads the checkpoint first
    evalonly = SmokeExperiment(_cfg(), tmp_path, ckpt_path=ckpt)
    evalonly.exec_task("validation")
