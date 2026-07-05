#!/usr/bin/env bash
# fetch_deps.sh — pull external libraries this project depends on.
#
# Idempotent: clones each dep on first run, no-ops on subsequent runs.
#
#   ./fetch_deps.sh                  # ensure all deps exist (clone if missing)
#   ./fetch_deps.sh --update         # fast-forward every dep to latest remote
#   ./fetch_deps.sh --local [path]   # snapshot a LOCAL working tree instead of
#                                    # cloning (default path: ../TTTN). Use this
#                                    # to build against unpushed TTTN changes;
#                                    # switch back with --update once pushed.
set -e
cd "$(dirname "$0")"

DEPS_DIR="deps"
mkdir -p "$DEPS_DIR"

MODE="ensure"
LOCAL_PATH="../TTTN"
case "${1:-}" in
    --update) MODE="update" ;;
    --local)  MODE="local"; [[ -n "${2:-}" ]] && LOCAL_PATH="$2" ;;
esac

# ─── dependency table ────────────────────────────────────────────────
# name | url | branch
DEPS=(
    "TTTN|https://github.com/benmeyersUSC/TTNN.git|main"
)

for entry in "${DEPS[@]}"; do
    IFS='|' read -r name url branch <<< "$entry"
    target="$DEPS_DIR/$name"

    if [ "$MODE" = "local" ]; then
        echo "[deps] snapshotting $name from local tree: $LOCAL_PATH"
        rsync -a --delete --exclude ".git" --exclude "checkpoints*" \
              --exclude "ens_runs" --exclude "cmake-build-*" --exclude "build" \
              "$LOCAL_PATH/" "$target/"
        continue
    fi

    if [ ! -d "$target/.git" ]; then
        if [ -d "$target" ]; then
            echo "[deps] $name present as local snapshot (rerun with --local to refresh,"
            echo "       or delete deps/$name and rerun to switch to the git clone)"
        else
            echo "[deps] cloning $name from $url ($branch)"
            git clone --depth 1 --branch "$branch" "$url" "$target"
        fi
    elif [ "$MODE" = "update" ]; then
        echo "[deps] updating $name"
        git -C "$target" fetch --depth 1 origin "$branch"
        git -C "$target" reset --hard "origin/$branch"
    else
        echo "[deps] $name present (use --update to refresh)"
    fi
done

echo "[deps] done"
