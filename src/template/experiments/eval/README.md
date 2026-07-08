# eval

`EvalMixin` supplies the `validation` task (run the backend's eval env, log the result);
`build_eval` is the lazy per-backend factory. Each runner consumes the algorithm's inference
API and returns a wandb-free `EvalResult` (metrics, videos, episodes). Same `backend=` as the
dataset, so training and eval share one world.
