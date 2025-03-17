# <img alt="Biolith logo" src="assets/biolith.svg" style="height: 2em;"> Biolith: <ins>B</ins>ayes<ins>i</ins>an Ec<ins>ol</ins>ogical Modeling <ins>i</ins>n Py<ins>th</ins>on

[![Test](https://github.com/timmh/biolith/actions/workflows/test.yml/badge.svg)](https://github.com/timmh/biolith/actions/workflows/test.yml) [![PyPI - Version](https://img.shields.io/pypi/v/biolith)](https://pypi.org/project/biolith/)

Biolith is a Python package designed for bayesian ecological modeling and analysis with a focus on occupancy modeling. It has similar goals to [Unmarked](https://github.com/biodiverse/unmarked) and [spOccupancy](https://github.com/biodiverse/spOccupancy/), but is written in Python and uses [NumPyro](https://num.pyro.ai) and [JAX](https://jax.readthedocs.io) to enable rapid model fitting and iteration.

## Features

- **Hackable**: Models are easy to understand and implement, no likelihood derivations needed.
- **Fast**: Models can be fit on GPUs, which is _fast_.
- **Familiar**: Everything is written in Python, making it easy to integrate into existing pipelines.

## Installation

You can install Biolith using pip:

```bash
pip install git+https://github.com/timmh/biolith
```

## Usage

Here is a simple example using simulated data to get you started:

```python
from biolith.models import occu, simulate
from biolith.utils import fit

# Simulate dataset
data, true_params = simulate()

# Fit model to simulated data
results = fit(occu, **data)

# Compare estimated occupancy probability to the true mean occupancy
print(f"Mean estimated psi: {results.samples['psi'].mean():.2f}")
print(f"Mean true occupancy: {true_params['z'].mean():.2f}")
```

## Real-world Example
To see a real-world example on camera trap data, see [this Jupyter Notebook](https://github.com/eco4cast/Statistical-Methods-Seminar-Series/tree/main/beery-haucke_biolith) from the EFI Statistical Methods Seminar Series or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1828fk-7DEsDL9reK5oYSOrsYA68cim-W)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please open an issue or email [haucke@mit.edu](mailto:haucke@mit.edu).
