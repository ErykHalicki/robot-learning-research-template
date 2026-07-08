from torch.utils.data import Dataset


class BaseSource(Dataset):
    """Backend adapter base. Every backend emits ONE common batch/obs-dict format so
    the model never sees a backend-specific dataset type.

    Standard batch keys (extend per project; keep the format documented in one place):
      - "observation" : model input(s)
      - "target"      : supervision target(s)

    `provided_modalities` advertises what this source supplies, so an experiment can
    fail fast at build time if the algorithm needs a modality the data lacks.
    """

    provided_modalities: set[str] = set()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
