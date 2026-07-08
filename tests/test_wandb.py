import pytest
from omegaconf import OmegaConf

from template.experiments.eval.base import EvalResult
from template.utils.wandb import (
    finish_wandb,
    init_wandb,
    log_eval,
    log_metrics,
    log_model_summary,
    wandb_enabled,
)


@pytest.fixture(autouse=True)
def _reset_wandb():
    # wandb keeps a process-global service that caches WANDB_DIR from the first init, so a
    # later test would write into an earlier test's torn-down tmp_path. Tear it down between.
    yield
    try:
        import wandb

        wandb.teardown()
    except Exception:
        pass


def test_disabled_fallback(monkeypatch, capsys):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    assert wandb_enabled() is False

    # init/finish are no-ops when disabled; logging falls back to stdout
    init_wandb(OmegaConf.create({"wandb": {"mode": "disabled"}}))
    log_metrics({"train/loss": 1.0})
    log_eval(EvalResult(metrics={"mse": 0.5}))
    finish_wandb()

    out = capsys.readouterr().out
    assert "train/loss" in out
    assert "mse" in out


def test_log_model_summary_disabled(monkeypatch, capsys):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    from omegaconf import OmegaConf

    from template.algorithms import build_algorithm

    model = build_algorithm(
        OmegaConf.create({"name": "dummy", "obs_dim": 8, "target_dim": 2, "hidden": 16})
    )
    log_model_summary(model)

    out = capsys.readouterr().out
    assert "params/total" in out
    assert "hidden_dim" in out  # per-algorithm override reached the logger


def _offline_env(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")


def test_offline_logging(monkeypatch, tmp_path):
    pytest.importorskip("wandb")
    _offline_env(monkeypatch, tmp_path)
    assert wandb_enabled() is True

    cfg = OmegaConf.create({"experiment": {"project": "test"}})
    init_wandb(cfg)
    log_metrics({"train/loss": 1.0}, step=0)
    log_eval(EvalResult(metrics={"mse": 0.5}), step=1)
    finish_wandb()

    assert list(tmp_path.glob("wandb/offline-run-*")), "no offline wandb run was created"


def test_log_eval_with_video_and_episodes(monkeypatch, tmp_path):
    pytest.importorskip("wandb")
    pytest.importorskip("moviepy")
    import numpy as np

    _offline_env(monkeypatch, tmp_path)

    init_wandb(OmegaConf.create({"experiment": {"project": "test"}}))
    result = EvalResult(
        metrics={"success_rate": 0.5},
        videos={"rollout": np.random.randint(0, 255, (4, 3, 16, 16), dtype=np.uint8)},
        episodes=[{"episode": 0, "success": 1}, {"episode": 1, "success": 0}],
    )
    log_eval(result, step=0)
    finish_wandb()

    assert list(tmp_path.glob("wandb/offline-run-*")), "no offline wandb run was created"


def test_experiment_logs_to_offline_wandb(monkeypatch, tmp_path):
    pytest.importorskip("wandb")
    _offline_env(monkeypatch, tmp_path)

    from template.experiments.smoke import SmokeExperiment

    cfg = OmegaConf.create(
        {
            "algorithm": {
                "name": "dummy",
                "obs_dim": 8,
                "target_dim": 2,
                "hidden": 16,
                "conditioning": ["observation"],
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
            "experiment": {
                "tasks": ["training"],
                "batch_size": 32,
                "lr": 1e-2,
                "epochs": 3,
                "log_every": 5,
                "mixed_precision": "no",
            },
            "wandb": {"entity": None, "project": "test", "mode": "offline"},
        }
    )
    # mirror main.py: one wandb run brackets the tasks
    init_wandb(cfg)
    SmokeExperiment(cfg, tmp_path / "runs").exec_task("training")
    finish_wandb()

    assert list(tmp_path.glob("wandb/offline-run-*")), "experiment did not create a wandb run"
