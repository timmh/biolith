#!/bin/sh
# Format Python code using isort --profile black and black

set -e
isort --profile black biolith
black biolith
