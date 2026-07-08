# robot-learning-research-template

A minimal, reusable scaffold for robot learning / ML research projects. Clean split between
the method, the data, and the runners, glued by hydra config and a thin experiment base class
over 🤗 Accelerate. `uv` for envs.

## Layout

```
main.py                  # entry point: registry -> build_experiment -> run tasks
src/template/            # rename `template` to your package name
  algorithms/    # the method. backend-free: never imports datasets/, experiments/, eval/.
  datasets/      # base batch format + lazy per-backend adapters (build_dataset factory).
  experiments/
    base.py      # BaseExperiment: builds algo/dataset, dispatches string tasks (exec_task).
    __init__.py  # exp_registry {name -> class} + build_experiment.
    smoke.py     # SmokeExperiment = TrainingMixin + EvalMixin + BaseExperiment.
    training/    # TrainingMixin.training(): the Accelerate loop.
    eval/        # EvalMixin.validation() + eval runners per backend (wandb-free EvalResult).
  utils/         # wandb, checkpoint, ckpt download, and other loop hooks.
  debug/         # one-off diagnostics.
configs/         # hydra groups: algorithm/ dataset/ experiment/ + top-level wandb/resume/load.
tests/           # smoke, modality-handshake, factories, resume/load, wandb.
docs/            # design notes.
```

## Core rules

- **One-directional imports.** `experiments` imports `algorithms` + `datasets`; `algorithms`
  imports from none of `datasets/`, `experiments/`, `eval/`. It may import any external package
  (torch, encoders, even a backend lib); the constraint is the internal layering.
- **Backends are optional and lazy.** A `backend` selects an *adapter*, not a specific
  dataset: `dataset.backend=lerobot` routes to `datasets/lerobot.py`, while a field like
  `dataset.repo_id=...` inside that config picks the actual dataset the adapter loads. The
  adapter is imported only when selected, so a base install pulls no backend; extras live in
  `pyproject.toml` (`[backend]`). `eval.backend` is the same idea for `build_eval` (which
  eval runner), with its own env/hardware params.
- **Eval configs are self-contained.** `eval` is its own config group with its own `backend`
  and env/hardware params; it never interpolates from `dataset`, so an eval-only run (no
  dataset at all) always resolves. Set `eval.backend` to match the dataset for a same-world
  held-out eval, or make them differ for sim-to-real (`dataset.backend=sim eval.backend=real_robot`).
- **Config over forks.** Architecture, dataset, conditioning are hydra groups, not code branches.
- **Experiments are a registry of tasks.** `main.py` stamps the chosen experiment yaml onto the
  config, looks its class up in `exp_registry`, and runs `cfg.experiment.tasks` in order via
  `exec_task`. An experiment composes only the task mixins it needs (train+eval, eval-only, train-only).
  Structure adapted from [buoyancy99/research-template](https://github.com/buoyancy99/research-template).

## Quickstart

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,wandb]"

# run the dummy smoke path (no wandb)
python main.py +name=smoke wandb.mode=disabled

# log to wandb (entity = your team/user)
python main.py +name=smoke wandb.entity=<team>

# tests
pytest -q
```

`+name=` is a required run name. Override anything else via hydra, e.g.
`python main.py +name=big experiment.epochs=20 algorithm.hidden=128`.

**Resume / load a checkpoint** (needs a wandb run that logged one):

```bash
python main.py +name=eval load=<run_id> experiment.tasks='[validation]'  # eval a past run
python main.py +name=cont resume=<run_id>                                # resume training
```

## Using this as a template

1. **Rename** `src/template/` to your package (and update `pyproject.toml` + imports).
2. Keep the dummy path green on a base install before touching anything.
3. Add your first backend: one `datasets/<backend>.py` + its eval counterpart behind an extra.
4. Replace the dummy algorithm, keeping the `loss` / `predict` signatures.
5. Iterate; keep the smoke run green after every change.
