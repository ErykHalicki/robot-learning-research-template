import os


def _has_wandb():
    try:
        import wandb  # noqa: F401
        return True
    except ImportError:
        return False


def wandb_enabled():
    """The single source of truth for whether wandb is live. Set WANDB_MODE=disabled to
    force stdout/json fallback (used by the smoke test and no-network runs).
    """
    return os.environ.get("WANDB_MODE", "") != "disabled" and _has_wandb()


def init_wandb(cfg):
    wandb_cfg = cfg.get("wandb", {})
    mode = wandb_cfg.get("mode", None)
    if mode:
        os.environ["WANDB_MODE"] = mode
    if not wandb_enabled():
        return
    import wandb
    from omegaconf import OmegaConf
    # resume=<run id> reattaches to that run (continue its history); otherwise start fresh
    resume_id = cfg.get("resume", None)
    wandb.init(
        id=resume_id,
        resume="must" if resume_id else None,
        name=None if resume_id else cfg.get("name", None),
        entity=wandb_cfg.get("entity", None),
        project=wandb_cfg.get("project", None) or "research-template",
        config=OmegaConf.to_container(cfg, resolve=True),
    )


def log_model_summary(model):
    """Push the algorithm's static stats (model.summary()) into wandb.config."""
    summary = model.summary()
    if not wandb_enabled():
        print(summary)
        return
    import wandb
    wandb.config.update(summary, allow_val_change=True)


def log_metrics(metrics, step=None):
    if not wandb_enabled():
        print(metrics)
        return
    import wandb
    wandb.log(metrics, step=step)


def log_eval(result, step=None):
    if not wandb_enabled():
        print(result.metrics)
        return
    import wandb
    wandb.log({f"eval/{k}": v for k, v in result.metrics.items()}, step=step)
    for name, frames in result.videos.items():
        wandb.log({f"eval/video/{name}": wandb.Video(frames, fps=10, format="gif")}, step=step)
    if result.episodes:
        import pandas as pd
        wandb.log({"eval/episodes": wandb.Table(dataframe=pd.DataFrame(result.episodes))}, step=step)


def finish_wandb():
    if not wandb_enabled():
        return
    import wandb
    wandb.finish()
