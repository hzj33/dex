#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="$ROOT_DIR/.config/codex/config.toml"
ENVRC_PATH="$ROOT_DIR/.envrc"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing config: $CONFIG_PATH" >&2
  exit 1
fi

export CODEX_HOME="$ROOT_DIR/.config/codex"

if [[ -f "$ENVRC_PATH" ]]; then
  # Load project-local secrets before validating required auth variables.
  # shellcheck disable=SC1090
  source "$ENVRC_PATH"
fi

TOKEN_ENV_NAME="$(
CONFIG_PATH="$CONFIG_PATH" python3 - <<'PY'
import os
import pathlib
import tomllib

config_path = pathlib.Path(os.environ["CONFIG_PATH"])
data = tomllib.loads(config_path.read_text())
print(data["model_providers"]["gac"]["env_key"])
PY
)"

if [[ -z "$TOKEN_ENV_NAME" ]]; then
  echo "Failed to load provider env var name from $CONFIG_PATH" >&2
  exit 1
fi

if [[ -z "${!TOKEN_ENV_NAME:-}" ]]; then
  echo "Missing required environment variable: $TOKEN_ENV_NAME" >&2
  echo "Export it before running this launcher." >&2
  exit 1
fi

cd "$ROOT_DIR"
exec codex "$@"
