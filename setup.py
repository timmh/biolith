from setuptools import setup, find_packages

VERSION = '0.0.3'
DESCRIPTION = 'Bayesian Ecological Modeling in Python '

# Read readme
from pathlib import Path
LONG_DESCRIPTION = (Path(__file__).parent / "README.md").read_text()

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="biolith", 
        version=VERSION,
        author="Timm Haucke",
        author_email="haucke@mit.edu",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        packages=find_packages(),
        install_requires=[
            # TODO: specify versions
            "numpy",
            "pandas",
            "jax",
            "numpyro",
            "funsor",  # currently required in order to do inference for models with discrete latent variables
        ],
        license="MIT",
        keywords=['python', 'occupancy', 'numpyro', 'bayesian', 'ecology'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
        ]
)
