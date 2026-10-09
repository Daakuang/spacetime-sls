# Spatiotemporal Response Decay for Near-Optimal Distributed LQR via System Level Synthesis

Code and data accompanying the [paper](https://arxiv.org/abs/2610.11699) by **Chenchen Zhou and José Matias**.

Use Python 3.12. From this folder, install the dependencies and run:

```sh
python -m pip install -r requirements.txt
python reproduce.py
```

This generates Figures 4–7 (PDF and PNG) and Table I (CSV) in `results/paper/`
from the supplied data.

To recompute the experiments from the model and then generate the same outputs:

```sh
python reproduce.py --recompute
```

Results are saved in `results/recompute/`. Recomputing all exact-SLS quadratic
programs can take substantial time. Dynamic-controller costs use 880 response
terms; the static-controller cost uses a Lyapunov equation.

`reproduce.py` is the only entry point. The model and solvers are in `control.py`,
the experiments in `experiments.py`, and the plots in `plotting.py`.

BSD-3-Clause license; see `LICENSE`.
