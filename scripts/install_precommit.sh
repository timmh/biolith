#!/bin/sh
# Install git pre-commit hook
set -e
HOOK_DIR="$(git rev-parse --git-dir)/hooks"
mkdir -p "$HOOK_DIR"
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
ln -sf "$SCRIPT_DIR/pre-commit" "$HOOK_DIR/pre-commit"
echo "Pre-commit hook installed."
