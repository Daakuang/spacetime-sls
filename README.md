# Spatiotemporal Response Decay for Near-Optimal Distributed LQR via System Level Synthesis

Code and data accompanying the paper by **Chenchen Zhou and José Matias**.

Start with **`reproduce.py`** to run the computations and generate the plots.
Use Python 3.11 or newer and run the following commands from this folder:

```sh
python -m pip install -r requirements.txt
python reproduce.py
```

This generates numerical Figures 4–7 (PDF and PNG) and Table I (CSV and JSON)
in `results/paper/`, using the supplied data in `data/`.

To recompute all experiments from the model and generate the figures and table:

```sh
python reproduce.py recompute
```

Results are saved in `results/recompute/`. This includes the large exact-SLS
quadratic programs and can take substantial time.

To try an individual controller on a smaller mesh:

```sh
python reproduce.py exact --side 3 --kappa 1 --footprint 1 --memory 5
python reproduce.py direct --side 3 --kappa 2 --footprint 2 --memory 0
```

`--memory` is the last filter tap `T`; the response horizon is `T+1`.
Dynamic direct-controller costs use finite response prefixes. Exact solves
use all disturbance sources unless `--sources` is specified; selected-source
costs describe those sources only. Run `python reproduce.py --help` for options.

BSD-3-Clause license; see `LICENSE`.
