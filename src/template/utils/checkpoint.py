from pathlib import Path

import torch


def save_checkpoint(model, optimizer, out_dir, step):
    """Save weights + optimizer + step to out_dir/model.pt. If a wandb run is active, also
    log it as a 'model' artifact so `resume=`/`load=<run id>` can fetch it later.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "model.pt"
    run_id = _run_id()
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "source_run_id": run_id,  # the wandb run that produced this ckpt (resume=/load= key)
        },
        path,
    )
    print(f"saved checkpoint: {path}" + (f"  (run {run_id})" if run_id else ""))
    _log_artifact(path)
    return path


def load_checkpoint(model, optimizer, path):
    """Load weights (and optimizer state, if an optimizer is given) from a .pt checkpoint.
    Returns the step it was saved at, so training can resume the counter.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("step", 0)


def _run_id():
    from .wandb import wandb_enabled

    if not wandb_enabled():
        return None
    import wandb

    return wandb.run.id if wandb.run is not None else None


def _log_artifact(path):
    run_id = _run_id()
    if run_id is None:
        return
    import wandb

    artifact = wandb.Artifact(name=f"model-{run_id}", type="model")
    artifact.add_file(str(path))
    wandb.log_artifact(artifact)
