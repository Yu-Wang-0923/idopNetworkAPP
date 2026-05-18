# idopNetwork

A Python toolkit for complex systems data analysis, implementing:

- **Curve Fitting**: Power-law fitting `y = a·x^b` with data transformation and quasi-dynamic ranking
- **Functional Clustering (FunClu)**: EM Gaussian mixture clustering with power-mean models
- **Network Reconstruction (NetRecon)**: Legendre basis expansion + constrained sparse regression (IDOPRegressor)
- **Network Analysis (NetAnal)**: GLMY persistent path homology and network depth analysis

## Installation

```bash
pip install idopnetwork
```

For machine learning extras:

```bash
pip install idopnetwork[ml]
```

## Usage

```python
from idopnetwork.curve_fitting import fit_power_loglinear, get_power_function_sample
from idopnetwork.clustering import FunClu
from idopnetwork.network import IDOPRegressor
from idopnetwork.analysis import run_glmy
```

## Citation

If you use idopNetwork in your research, please cite:

> Wang, Y. et al. "idopNetwork: An integrative platform for complex systems analysis." (in preparation)
