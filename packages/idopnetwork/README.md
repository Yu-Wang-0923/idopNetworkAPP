# idopNetwork

A Python toolkit for complex systems data analysis, implementing:

- **Curve Fitting**: Power-law fitting `y = a·x^b` with data transformation and quasi-dynamic ranking
- **Functional Clustering (FunClu)**: EM Gaussian mixture clustering with power-mean models
- **Network Reconstruction (NetRecon)**: Legendre basis expansion + constrained sparse regression (IDOPRegressor)
- **Network Analysis (NetAnal)**: GLMY persistent path homology

## Installation

```bash
pip install idopnetwork
```

For PyTorch-based clustering (FunClu) extras:

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

## Multilayer network plots

Render a NetRecon Multi-Layer export as a grid with an inter-module network on
the first row and module subnetworks below it. Each panel uses the IDOP ring
layout and ranks its hubs by out-degree.

From a repository checkout, install the package locally and open the desktop
module selector (requires Tkinter):

```bash
python -m pip install -e ./packages/idopnetwork
python -m idopnetwork.network.multilayer_plot
```

To render selected modules from the command line:

```bash
python -m idopnetwork.network.multilayer_plot multilayer_export.zip --modules M1 M3 --out multilayer.svg
```

The output extension selects SVG, PNG, or PDF.

## Citation

If you use idopNetwork in your research, please cite:

> Wang, Y. et al. "idopNetwork: An integrative platform for complex systems analysis." (in preparation)
