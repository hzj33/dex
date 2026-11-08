#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/dexh13_deploy_collect.sh [options] [extra dexh13_deploy_collect.py args...]

Options:
  --hand-side <right|left>      Select hand side metadata and default serial port. Default: right
  --tactile-detail             Enable getFingerTactileDetail() array collection.
  --no-tactile-detail          Disable tactile detail collection. Default.
  --handy-port <port>          Override DexH13 serial port.
  --camera-port <port>         Override camera port. Default: auto
  --quest-hand <l|r>           Quest controller hand. Default: l
  --arm-side <right|left>      RM65B arm side. Default follows --hand-side
  --no-command-rm65b           Do not send Quest-generated arm_control to RM65B.
  -h, --help                   Show this help.

Examples:
  bash scripts/dexh13_deploy_collect.sh --hand-side right
  bash scripts/dexh13_deploy_collect.sh --hand-side left --tactile-detail
  bash scripts/dexh13_deploy_collect.sh --hand-side left --handy-port /dev/ttyUSB0 --tactile-detail
EOF
}

HAND_SIDE="right"
TACTILE_DETAIL=0
HANDY_PORT=""
CAMERA_PORT="auto"
QUEST_HAND="l"
ARM_SIDE=""
COMMAND_RM65B=1
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hand-side)
      HAND_SIDE="$2"
      shift 2
      ;;
    --hand-side=*)
      HAND_SIDE="${1#*=}"
      shift
      ;;
    --tactile-detail)
      TACTILE_DETAIL=1
      shift
      ;;
    --no-tactile-detail)
      TACTILE_DETAIL=0
      shift
      ;;
    --handy-port)
      HANDY_PORT="$2"
      shift 2
      ;;
    --handy-port=*)
      HANDY_PORT="${1#*=}"
      shift
      ;;
    --camera-port)
      CAMERA_PORT="$2"
      shift 2
      ;;
    --camera-port=*)
      CAMERA_PORT="${1#*=}"
      shift
      ;;
    --quest-hand)
      QUEST_HAND="$2"
      shift 2
      ;;
    --quest-hand=*)
      QUEST_HAND="${1#*=}"
      shift
      ;;
    --arm-side)
      ARM_SIDE="$2"
      shift 2
      ;;
    --arm-side=*)
      ARM_SIDE="${1#*=}"
      shift
      ;;
    --no-command-rm65b)
      COMMAND_RM65B=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "$HAND_SIDE" != "right" && "$HAND_SIDE" != "left" ]]; then
  echo "ERROR: --hand-side must be 'right' or 'left', got: $HAND_SIDE" >&2
  exit 2
fi

if [[ "$QUEST_HAND" != "l" && "$QUEST_HAND" != "r" ]]; then
  echo "ERROR: --quest-hand must be 'l' or 'r', got: $QUEST_HAND" >&2
  exit 2
fi

if [[ -z "$ARM_SIDE" ]]; then
  ARM_SIDE="$HAND_SIDE"
fi

if [[ "$ARM_SIDE" != "right" && "$ARM_SIDE" != "left" ]]; then
  echo "ERROR: --arm-side must be 'right' or 'left', got: $ARM_SIDE" >&2
  exit 2
fi

if [[ -z "$HANDY_PORT" ]]; then
  if [[ "$HAND_SIDE" == "left" ]]; then
    HANDY_PORT="/dev/ttyUSB_left"
  else
    HANDY_PORT="/dev/ttyUSB0"
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

CMD=(
  python sim2real/dexh13_deploy_collect.py
  --save-data
  --profile-loop
  --hand-side "$HAND_SIDE"
  --handy-port "$HANDY_PORT"
  --camera-port "$CAMERA_PORT"
  --arm-state-mode quest-rm65b
  --quest-hand "$QUEST_HAND"
  --arm-side "$ARM_SIDE"
  --no-send-between-policy
)

if [[ "$COMMAND_RM65B" -eq 1 ]]; then
  CMD+=(--command-rm65b)
fi

if [[ "$TACTILE_DETAIL" -eq 1 ]]; then
  CMD+=(--tactile-detail)
fi

CMD+=("${EXTRA_ARGS[@]}")

printf '[dexh13-deploy-collect] hand_side=%s handy_port=%s tactile_detail=%s arm_side=%s quest_hand=%s\n' \
  "$HAND_SIDE" "$HANDY_PORT" "$TACTILE_DETAIL" "$ARM_SIDE" "$QUEST_HAND"
printf '[dexh13-deploy-collect] command:'
printf ' %q' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
