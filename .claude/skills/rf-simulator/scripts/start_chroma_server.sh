#!/usr/bin/env bash
# start_chroma_server.sh — run the Chroma HTTP server on sunlab in the
# background, bound to 127.0.0.1 only (defense in depth — even if the
# UGA firewall ever opens up, the server is loopback-only and only
# reachable through the SSH tunnel).
#
# Path resolution mirrors store.py:
#   1. SIONNA_SKILL_CHROMA_PATH env var
#   2. ~/.local/share/sionna-skill/chroma_db (XDG default)
#   3. .claude/skills/rf-simulator/memory/chroma_db (legacy in-tree)
#
# Usage:
#   bash .claude/skills/rf-simulator/scripts/start_chroma_server.sh           # foreground
#   bash .claude/skills/rf-simulator/scripts/start_chroma_server.sh --bg      # background
#   bash .claude/skills/rf-simulator/scripts/start_chroma_server.sh --stop    # kill running server
#   CHROMA_TOKEN=$(openssl rand -hex 32) bash .../start_chroma_server.sh --bg
#
# When CHROMA_TOKEN is set, the server requires Authorization: Bearer
# <token> on all requests. Pass the same token to clients via
# Settings(chroma_client_auth_credentials=...).
set -euo pipefail

PORT="${CHROMA_PORT:-8000}"
LOG="${CHROMA_LOG:-/tmp/chroma.log}"
PIDFILE="${CHROMA_PIDFILE:-/tmp/chroma.pid}"

resolve_path() {
    if [[ -n "${SIONNA_SKILL_CHROMA_PATH:-}" ]]; then
        echo "$SIONNA_SKILL_CHROMA_PATH"; return
    fi
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local legacy="$script_dir/../memory/chroma_db"
    if [[ -d "$legacy" ]] && [[ -n "$(ls -A "$legacy" 2>/dev/null)" ]]; then
        cd "$legacy" && pwd; return
    fi
    echo "$HOME/.local/share/sionna-skill/chroma_db"
}

DB_PATH="$(resolve_path)"
mkdir -p "$DB_PATH"

stop_server() {
    if [[ -f "$PIDFILE" ]]; then
        local pid
        pid=$(cat "$PIDFILE")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            echo "stopped chroma (pid=$pid)"
        fi
        rm -f "$PIDFILE"
    else
        pkill -f "chroma run" 2>/dev/null || true
        echo "no pidfile; sent pkill chroma run"
    fi
}

case "${1:-}" in
    --stop) stop_server; exit 0 ;;
esac

# Auth env vars consumed by chromadb when set
if [[ -n "${CHROMA_TOKEN:-}" ]]; then
    export CHROMA_SERVER_AUTHN_PROVIDER="chromadb.auth.token_authn.TokenAuthenticationServerProvider"
    export CHROMA_SERVER_AUTHN_CREDENTIALS="$CHROMA_TOKEN"
    echo "[start_chroma_server] token auth enabled (\${CHROMA_TOKEN:0:8}...)"
fi

CMD=(chroma run --host 127.0.0.1 --port "$PORT" --path "$DB_PATH")

case "${1:-}" in
    --bg)
        # Don't start a second server on the same port
        if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            echo "chroma already running (pid=$(cat "$PIDFILE")); use --stop first"
            exit 1
        fi
        nohup "${CMD[@]}" > "$LOG" 2>&1 &
        echo $! > "$PIDFILE"
        sleep 1
        echo "chroma running (pid=$(cat "$PIDFILE")) at 127.0.0.1:$PORT"
        echo "  db_path: $DB_PATH"
        echo "  log:     $LOG"
        ;;
    *)
        echo "starting chroma in foreground at 127.0.0.1:$PORT"
        echo "  db_path: $DB_PATH"
        exec "${CMD[@]}"
        ;;
esac
