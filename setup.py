from setuptools import setup, find_packages

VERSION = '0.0.7'
DESCRIPTION = 'Bayesian Ecological Modeling in Python '

# Read readme
from pathlib import Path
LONG_DESCRIPTION = (Path(__file__).parent / "README.md").read_text()

# Setting up
setup(
        name="biolith", 
        version=VERSION,
        author="Timm Haucke",
        author_email="haucke@mit.edu",
        project_urls={
            "Documentation": "https://timm.haucke.xyz/biolith/",
            "Source": "https://github.com/timmh/biolith",
        },
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        packages=find_packages(),
        install_requires=[
            "numpy==2.2.4",
            "pandas==2.2.3",
            "jax==0.5.2",
            "numpyro==0.18.0",
            "funsor==0.4.5",  # currently required in order to do inference for models with discrete latent variables
        ],
        license="MIT",
        keywords=['python', 'occupancy', 'numpyro', 'bayesian', 'ecology'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
        ]
)
