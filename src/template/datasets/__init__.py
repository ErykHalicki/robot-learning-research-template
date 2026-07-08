def build_dataset(cfg):
    """Lazy backend factory. A backend module is imported only when selected, so a
    base install never imports a backend that is not present.
    """
    backend = cfg.backend
    if backend == "dummy":
        from .dummy import DummySource
        return DummySource(cfg)
    # if backend == "lerobot":
    #     from .lerobot import LeRobotSource
    #     return LeRobotSource(cfg)
    # if backend == "swm":
    #     from .swm import SWMSource
    #     return SWMSource(cfg)
    raise ValueError(
        f"backend '{backend}' not installed. "
        f"try: uv pip install 'robot-learning-research-template[{backend}]'"
    )
