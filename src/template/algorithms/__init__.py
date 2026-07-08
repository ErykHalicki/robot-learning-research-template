def build_algorithm(cfg):
    if cfg.name == "dummy":
        from .dummy import DummyAlgorithm
        return DummyAlgorithm(cfg)
    raise ValueError(f"unknown algorithm '{cfg.name}'")
