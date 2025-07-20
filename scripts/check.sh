#!/bin/sh
# Run formatting and static analysis checks

set -e

isort --profile black --check --diff biolith
docformatter --black --check -r biolith
black --check biolith
pylint biolith || echo  # be non-strict for now, TODO: fix pylint issues
pyright biolith || echo  # be non-strict for now, TODO: fix pyright issues
