# training

`TrainingMixin`: supplies the `training` task, a thin Accelerate loop (DDP + mixed
precision) with wandb metric logging and checkpointing. Mix into any experiment that trains.
The wandb run itself is opened/closed by `main.py` around all tasks.
