#!/bin/sh
# Format Python code using isort, docformatter and black

set -e
isort --profile black biolith
docformatter --black --in-place -r biolith
black biolith
