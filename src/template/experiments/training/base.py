import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from ...utils.checkpoint import load_checkpoint, save_checkpoint
from ...utils.wandb import log_metrics, log_model_summary


class TrainingMixin:
    """Supplies the `training` task: a thin Accelerate loop (DDP + mixed precision) with
    wandb logging and checkpointing. Mix into an experiment that should train. Reads the
    algorithm/dataset/paths from the BaseExperiment it is composed with. The wandb run
    itself is opened/closed by main.py around all tasks, not here.
    """

    def training(self):
        exp = self.cfg
        acc = Accelerator(mixed_precision=exp.mixed_precision)
        self._build_algo()
        self._check_modalities()

        if acc.is_main_process:
            log_model_summary(self.algo)

        loader = DataLoader(self.dataset, batch_size=exp.batch_size, shuffle=True)
        opt = torch.optim.AdamW(self.algo.parameters(), lr=exp.lr)
        start_step = load_checkpoint(self.algo, opt, self.ckpt_path) if self.ckpt_path else 0
        model, opt, loader = acc.prepare(self.algo, opt, loader)

        step = start_step
        for _ in range(exp.epochs):
            for batch in loader:
                out = model.loss(batch)
                acc.backward(out["loss"])
                opt.step()
                opt.zero_grad()
                if step % exp.log_every == 0:
                    acc.print(f"step {step}  loss {out['loss'].item():.4f}")
                    if acc.is_main_process:
                        log_metrics({"train/loss": out["loss"].item()}, step=step)
                step += 1

        if acc.is_main_process:
            self.algo = acc.unwrap_model(model)
            self._trained = True
            save_checkpoint(self.algo, opt, self.output_dir, step)
