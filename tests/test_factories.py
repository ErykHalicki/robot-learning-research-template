import pytest
from omegaconf import OmegaConf

from template.algorithms import build_algorithm
from template.datasets import build_dataset
from template.experiments import build_experiment
from template.experiments.eval import build_eval


def test_unknown_dataset_backend_hints_extra():
    cfg = OmegaConf.create({"backend": "nope"})
    with pytest.raises(ValueError, match=r"lewam|nope|install"):
        build_dataset(cfg)


def test_unknown_eval_backend_hints_extra():
    cfg = OmegaConf.create({"backend": "nope"})
    with pytest.raises(ValueError, match=r"nope|install"):
        build_eval(cfg)


def test_unknown_algorithm_raises():
    cfg = OmegaConf.create({"name": "nope"})
    with pytest.raises(ValueError, match="nope"):
        build_algorithm(cfg)


def test_build_experiment_returns_registered_class():
    cfg = OmegaConf.create({"experiment": {"_name": "smoke"}})
    exp = build_experiment(cfg, output_dir="runs")
    assert exp.__class__.__name__ == "SmokeExperiment"


def test_unknown_experiment_raises():
    cfg = OmegaConf.create({"experiment": {"_name": "nope"}})
    with pytest.raises(ValueError, match="nope"):
        build_experiment(cfg)
