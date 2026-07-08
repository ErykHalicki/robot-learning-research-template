def build_eval(cfg):
    """Lazy eval-backend factory, mirroring datasets/. Same `backend=` dimension picks
    the matching eval environment, so training and eval always share one world.
    """
    backend = cfg.backend
    if backend == "dummy":
        from .dummy import DummyEval
        return DummyEval(cfg)
    # if backend == "lerobot":
    #     from .lerobot import LeRobotEval
    #     return LeRobotEval(cfg)
    # if backend == "swm":
    #     from .swm import SWMEval
    #     return SWMEval(cfg)
    raise ValueError(
        f"eval backend '{backend}' not installed. "
        f"try: uv pip install 'robot-learning-research-template[{backend}]'"
    )
