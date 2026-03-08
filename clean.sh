#!/bin/sh
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

python clean.py
