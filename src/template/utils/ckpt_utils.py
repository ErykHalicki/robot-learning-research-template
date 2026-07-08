from pathlib import Path


def is_run_id(run_id):
    """True if a string looks like a wandb run id (8 alphanumeric chars)."""
    return len(run_id) == 8 and run_id.isalnum()


def _version_to_int(artifact):
    """Convert an artifact version of the form vX to the int X."""
    return int(artifact.version[1:])


def download_latest_checkpoint(run_path, download_dir):
    """Download the latest committed 'model' artifact of a wandb run and return the local
    .pt path. Adapted from Boyuan Chen's research template (MIT).
    """
    import wandb

    run = wandb.Api().run(run_path)
    latest = None
    for artifact in run.logged_artifacts():
        if artifact.type != "model" or artifact.state != "COMMITTED":
            continue
        if latest is None or _version_to_int(artifact) > _version_to_int(latest):
            latest = artifact
    if latest is None:
        raise ValueError(f"no model checkpoint artifact found for run {run_path}")

    root = Path(download_dir) / run_path
    root.mkdir(parents=True, exist_ok=True)
    latest.download(root=str(root))
    return next(root.glob("*.pt"))
