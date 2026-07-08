# Research Template

A minimal, reusable starting point for ML research repos, distilled from the LeWAM v0.4 rework. It keeps the shape of buoyancy99/research-template and standardizes on `uv`. Use this to bootstrap a new project without re-deriving the structure each time.

The idea: a clean split between the method, the data (backend adapters), and the runners (training + eval), glued by hydra config and a thin experiment base class over Accelerate. Everything project-specific (the actual model, the actual datasets, the actual hardware) plugs into fixed slots.

## What goes in the template vs the project

The template ships the scaffolding and the contracts. It does NOT ship a real model, a real dataset, or real hardware code. Those are the first things a new project fills in.

**In the template (generic):**
- Repo layout under `src/<pkg>/` and the one-directional import rule.
- `BaseExperiment` over Accelerate (DDP + mixed precision), with hook points for eval/checkpoint/viz.
- Lazy backend factory pattern for `datasets/` and `experiments/eval/`.
- `EvalResult` contract and the single `utils/wandb.py` logger.
- Hydra config groups and the optional-extras dependency scheme in `pyproject.toml`.
- A dummy algorithm + dummy dataset that make the smoke run green out of the box.

**Left to each project (specific):**
- The real algorithm(s) in `algorithms/`.
- Real dataset adapters and any embodiment/hardware code.
- Real eval environments.

## Repository structure

Code lives under `src/<pkg>/` so it stays an installable package (external tooling must be able to `import <pkg>`).

```
main.py          # entry point at repo root: build_experiment -> run cfg.experiment.tasks
src/<pkg>/
  algorithms/    # the method. models, loss/step, sampling, and a clean inference API.
                 # may import any external package it needs (torch, encoders, even a
                 # backend lib), but must NOT import from datasets/, experiments/, or eval/.
  datasets/      # base batch/obs-dict interface + per-backend adapters, lazy-imported.
  experiments/
    base.py      # BaseExperiment: builds algo/dataset, dispatches string tasks (exec_task).
    __init__.py  # exp_registry {name -> class} + build_experiment.
    training/    # TrainingMixin.training(): the Accelerate loop.
    eval/        # EvalMixin.validation() + eval runners per backend, returning EvalResult.
  utils/         # wandb helpers, checkpoint I/O + download, viz, sync. called as hooks by the loop.
  debug/         # diagnostics and one-off debugging utilities.

configs/         # hydra config groups: algorithm/ dataset/ experiment/ + top-level wandb/resume/load
docs/            # design + reference docs
tests/
```

## Core rules

- **One-directional imports.** `experiments` imports `algorithms` + `datasets`; `algorithms` imports from neither `datasets/`, `experiments/`, nor `eval/`. Nothing imports "up". `algorithms/` may freely import external packages (torch, encoders, model libs, even a backend library); the constraint is on the internal layering, not on third-party deps.
- **The method layer stays decoupled.** `algorithms/` exposes one framework-neutral inference API and takes/returns tensors. Because it never reaches into the data or eval layers, the same model can train on any backend and be evaluated by any adapter.
- **Backend selects an adapter, not a dataset.** `dataset.backend` picks which source class `build_dataset` returns (`lerobot`, `swm`, `dummy`); a field inside that config (e.g. `dataset.repo_id`) picks the specific dataset the adapter loads. The adapter is imported only when selected, so a missing backend gives a clean "install <pkg>[x]" message, not an import error. Extras: `<pkg>[backendA]`, `<pkg>[backendB]`, `<pkg>[all]`. `eval.backend` does the same for `build_eval` (which eval runner), carrying its own env/hardware params.
- **Eval configs are self-contained.** `eval` is its own config group with its own `backend` and env/hardware params; it never interpolates from `dataset`, so an eval-only run (no dataset group) always resolves. Set `eval.backend` to the dataset's for a same-world held-out eval, or make them differ for sim-to-real, where the training data and the eval environment are deliberately different worlds (`dataset.backend=sim eval.backend=real_robot`). The only hard requirement is that the chosen eval env emit observations matching the algorithm's input contract.
- **Keep `uv`.** Not conda/pip. It is the one part of the reference template we reject.
- **Thin `BaseExperiment` over Accelerate.** Custom pieces (sync, viz, autotune) are plain hook functions in `utils/`, called by the loop, not framework callbacks.
- **Config over forks.** Architecture, backend, conditioning, and any variant are config choices (hydra groups), not code branches. Adding a variant is a new yaml, not a new script.

## The contracts

Three small contracts are the whole template. Fill in the slots and the loop runs.

### Dataset batch format (`datasets/base.py`)

Every backend adapter emits one common batch/obs-dict format so the model never sees a backend-specific dataset type. The exact keys are project-specific; the rule is that there is a single documented format both training and eval consume.

### Lazy backend factory (`datasets/__init__.py`)

```python
def build_dataset(cfg):
    if cfg.backend == "backendA": from .backend_a import SourceA; return SourceA(cfg)
    if cfg.backend == "backendB": from .backend_b import SourceB; return SourceB(cfg)
    raise ValueError(f"backend '{cfg.backend}' not installed. try: uv pip install <pkg>[{cfg.backend}]")
```

The same lazy-factory pattern is mirrored in `experiments/eval/` for eval environments.

### Eval result (`experiments/eval/base.py`)

Eval backends return a plain, wandb-free result:

```python
@dataclass
class EvalResult:
    metrics: dict[str, float]
    videos: dict[str, np.ndarray] = {}   # uint8 (T,C,H,W); logger wraps in wandb.Video
    episodes: list[dict] = []            # per-episode rows -> wandb.Table
```

One helper in `utils/wandb.py` is the only place that knows wandb, so no-wandb runs degrade to stdout/json. Backends stay wandb-free so control loops never block on network I/O; buffer to disk, upload after.

### Experiment execution (`main.py` + `experiments/`)

Execution follows [buoyancy99/research-template](https://github.com/buoyancy99/research-template) (MIT), minus PyTorch Lightning and the SLURM layer:

- `main.py` (repo root) is a thin Hydra entry. It reads `HydraConfig.runtime.choices` to stamp the selected yaml name onto `cfg.<group>._name`, requires a `+name=` run name, resolves `resume=`/`load=` to a checkpoint (local path or a wandb run's latest `model` artifact), opens one wandb run, then calls `build_experiment(cfg, output_dir, ckpt_path)`.
- `experiments/__init__.py` holds `exp_registry = {name -> class}` and `build_experiment`, keyed on `cfg.experiment._name` (same string as the yaml filename).
- `BaseExperiment` builds the algorithm/dataset via the factories and dispatches `cfg.experiment.tasks` (e.g. `[training, validation]`) to methods via `exec_task`. Task methods come from mixins (`TrainingMixin`, `EvalMixin`), so an experiment composes only the tasks it needs, an eval-only robotics run mixes just `EvalMixin`.

## pyproject / extras scheme

```toml
[project]
dependencies = [ "torch", "accelerate", "hydra-core", "wandb", ... ]  # base: model + training scaffolding, no backend

[project.optional-dependencies]
backendA = [ ... ]
backendB = [ ... ]
all = [ "<pkg>[backendA]", "<pkg>[backendB]" ]
```

A base install trains the dummy algorithm on the dummy dataset with no backend present.

## Bootstrapping a new project from the template

1. **Rename the package.** `src/<pkg>/`, the `[project] name`, and imports.
2. **Keep the dummy path green.** The shipped dummy algorithm + dummy dataset should run end to end (data -> train -> checkpoint -> eval) on a base install before you touch anything.
3. **Add your first backend.** Implement one `datasets/<backend>.py` and its eval counterpart behind an extra. Cheapest sim/toy world first.
4. **Add a real algorithm.** Replace the dummy, keeping the inference API signature.
5. **Wire an experiment.** Add a concrete experiment (compose the task mixins it needs) and register it in `exp_registry` under its yaml name; declare its `tasks`. Run via `python main.py +name=... experiment=<yourexp>`.
6. **Iterate.** Keep the smoke run green after every change; grow complexity only once the loop is solid.

## Creating the GitHub repo

Once this doc's structure is realized as an actual skeleton (empty slots + dummy path), publish it as a standalone template repo:

- Strip anything project-specific to LeWAM (no algorithms, no LeWAM datasets, no B601). Keep only the scaffolding, contracts, dummy path, and configs.
- Mark it a **template repository** in GitHub settings so future projects start with "Use this template".
- Suggested name: `research-template` (or `<yourhandle>-research-template`).
- Decide visibility (public reads better on applications; private if it will hold unpublished ideas).

`gh repo create <name> --template` is NOT what you want here (that consumes a template); to publish this AS a template, push it and then toggle the template flag:

```bash
gh repo create <name> --public --source . --push
gh repo edit <name> --template   # mark as a template repo
```
