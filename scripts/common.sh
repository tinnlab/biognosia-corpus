#!/usr/bin/env bash
# Shared setup for the pipeline scripts. Source this file, do not execute it.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PROJECT_ROOT

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.env"
    set +a
fi

# load_config <file>: source config/<file> and export the library paths derived from it
load_config() {
    local config_file="$PROJECT_ROOT/config/$1"
    if [ ! -f "$config_file" ]; then
        echo "ERROR: configuration file not found: $config_file" >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "$config_file"

    export DATA_DIR MODEL_CACHE_DIR
    export HF_HOME="$MODEL_CACHE_DIR"
    export NLTK_DATA="$DATA_DIR/resources/nltk_data"
    export CUDA_VISIBLE_DEVICES
    export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    if [ -n "${HF_TOKEN:-}" ]; then
        export HF_TOKEN
    fi
}

cd "$PROJECT_ROOT"
