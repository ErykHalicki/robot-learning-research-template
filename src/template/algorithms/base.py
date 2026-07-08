import torch.nn as nn


class BaseAlgorithm(nn.Module):
    """The method. Backend-free: must NOT import from datasets/, experiments/, or eval/.

    It may import any external package it needs (torch, encoders, model libraries).
    Two methods form the whole contract the rest of the repo relies on:

      loss(batch)  -> dict containing a "loss" scalar for the training loop
      predict(obs) -> outputs, the framework-neutral inference API used by eval
    """

    def loss(self, batch):
        raise NotImplementedError

    def predict(self, obs):
        raise NotImplementedError

    def summary(self):
        """Static stats logged to wandb.config at startup. Defaults to param counts;
        override (and call super().summary()) to add per-algorithm stats like encoder
        size, block count, or latent dims.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"params/total": total, "params/trainable": trainable}
