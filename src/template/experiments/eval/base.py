from dataclasses import dataclass, field


@dataclass
class EvalResult:
    """The eval contract. Backends return this plain object and never touch wandb, so
    control loops never block on network I/O. utils/wandb.py is what wraps it for logging.
    """

    metrics: dict          # success_rate, mse, mean_episode_length, per-task ...
    videos: dict = field(default_factory=dict)     # name -> uint8 (T,C,H,W); wrapped in wandb.Video
    episodes: list = field(default_factory=list)   # per-episode rows -> wandb.Table


class EvalMixin:
    """Supplies the `validation` task: run this backend's eval environment on the model and
    log the EvalResult. Mix into any experiment that should be evaluated, including
    eval-only ones (no training task). If the model was not trained in this run, the
    requested checkpoint is loaded first.
    """

    def validation(self):
        from . import build_eval
        from ...utils.checkpoint import load_checkpoint
        from ...utils.wandb import log_eval

        model = self._build_algo()
        if not self._trained and self.ckpt_path is not None:
            load_checkpoint(model, None, self.ckpt_path)
        result = build_eval(self.root_cfg.eval).run(model)
        log_eval(result)
