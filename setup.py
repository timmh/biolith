from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Bayesian ecological modeling in Python'
LONG_DESCRIPTION = 'A package that provides Bayesian ecological modeling in Python'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="biolith", 
        version=VERSION,
        author="Timm Haucke",
        author_email="haucke@mit.edu",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
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
