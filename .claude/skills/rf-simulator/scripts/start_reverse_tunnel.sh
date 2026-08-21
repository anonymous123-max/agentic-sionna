#!/usr/bin/env bash
# start_reverse_tunnel.sh — open a reverse SSH tunnel from sunlab → vast.ai
# so the rented GPU can reach the chroma server running at sunlab:8000.
#
# Pattern:
#   sunlab:127.0.0.1:8000  ◄──── ssh -R 8000:localhost:8000 ────  vast.ai
#                                                                  (process on
#                                                                   vast.ai talks
#                                                                   to localhost:8000)
#
# Usage:
#   bash .claude/skills/rf-simulator/scripts/start_reverse_tunnel.sh \
#       --vast-host ssh1.vast.ai \
#       --vast-port 23456 \
#       --vast-user root \
#       [--key ~/.ssh/vast_tunnel] \
#       [--local-port 8000] [--remote-port 8000] \
#       [--stop]
#
# Run on sunlab (NOT on your laptop). The tunnel persists in background
# via autossh, auto-reconnecting if the SSH session drops.
#
# Prereqs (one-time on sunlab):
#   sudo apt-get install autossh   # or pacman/dnf equivalent
#   ssh-keygen -t ed25519 -f ~/.ssh/vast_tunnel -N ''
#   # Add ~/.ssh/vast_tunnel.pub to vast.ai (web UI → Account → SSH Keys)
#
# Vast.ai instances are rented per session — VAST_HOST + VAST_PORT change
# each time. Copy them from the instance's "Connect" panel.
set -euo pipefail

VAST_HOST="${VAST_HOST:-}"
VAST_PORT="${VAST_PORT:-}"
VAST_USER="${VAST_USER:-root}"
KEY="${VAST_TUNNEL_KEY:-$HOME/.ssh/vast_tunnel}"
LOCAL_PORT="${LOCAL_PORT:-8000}"
REMOTE_PORT="${REMOTE_PORT:-8000}"
PIDFILE="${TUNNEL_PIDFILE:-/tmp/vast_tunnel.pid}"
LOG="${TUNNEL_LOG:-/tmp/vast_tunnel.log}"
ACTION="start"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --vast-host)   VAST_HOST="$2"; shift 2 ;;
        --vast-port)   VAST_PORT="$2"; shift 2 ;;
        --vast-user)   VAST_USER="$2"; shift 2 ;;
        --key)         KEY="$2"; shift 2 ;;
        --local-port)  LOCAL_PORT="$2"; shift 2 ;;
        --remote-port) REMOTE_PORT="$2"; shift 2 ;;
        --stop)        ACTION="stop"; shift ;;
        --status)      ACTION="status"; shift ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

case "$ACTION" in
    stop)
        if [[ -f "$PIDFILE" ]]; then
            pid=$(cat "$PIDFILE")
            kill "$pid" 2>/dev/null && echo "stopped tunnel (pid=$pid)"
            rm -f "$PIDFILE"
        else
            pkill -f "autossh.*-R $REMOTE_PORT:127.0.0.1:$LOCAL_PORT" 2>/dev/null || true
            echo "no pidfile; pkill autossh issued"
        fi
        exit 0
        ;;
    status)
        if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            echo "tunnel UP (pid=$(cat "$PIDFILE"))"
            ps -p "$(cat "$PIDFILE")" -o pid,etime,cmd
        else
            echo "tunnel DOWN"
            exit 1
        fi
        exit 0
        ;;
esac

if [[ -z "$VAST_HOST" ]] || [[ -z "$VAST_PORT" ]]; then
    echo "ERROR: --vast-host and --vast-port required"
    echo "       grab them from the vast.ai instance Connect panel"
    exit 1
fi
if [[ ! -f "$KEY" ]]; then
    echo "ERROR: SSH key not found at $KEY"
    echo "       generate with: ssh-keygen -t ed25519 -f $KEY -N ''"
    echo "       then add ${KEY}.pub to vast.ai (Account → SSH Keys)"
    exit 1
fi
if ! command -v autossh >/dev/null; then
    echo "ERROR: autossh not installed (apt-get install autossh)"
    exit 1
fi

# Don't double-up tunnels
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "tunnel already running (pid=$(cat "$PIDFILE")); use --stop first"
    exit 1
fi

# autossh respawns if ssh dies; -M 0 disables monitoring port (relies on
# ServerAliveInterval/CountMax for liveness, lighter than older versions).
autossh -M 0 -f -N \
    -i "$KEY" \
    -o ServerAliveInterval=60 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -o StrictHostKeyChecking=accept-new \
    -R "$REMOTE_PORT:127.0.0.1:$LOCAL_PORT" \
    -p "$VAST_PORT" \
    "$VAST_USER@$VAST_HOST" \
    > "$LOG" 2>&1

# autossh forks; capture the parent (autossh, not ssh)
sleep 1
pid=$(pgrep -f "autossh.*-R $REMOTE_PORT:127.0.0.1:$LOCAL_PORT" | head -1)
if [[ -z "$pid" ]]; then
    echo "ERROR: autossh failed to start; see $LOG"
    cat "$LOG"
    exit 1
fi
echo "$pid" > "$PIDFILE"

echo "tunnel UP (pid=$pid)"
echo "  sunlab:127.0.0.1:$LOCAL_PORT  ◄──  vast.ai:127.0.0.1:$REMOTE_PORT"
echo "  vast.ai endpoint: $VAST_USER@$VAST_HOST:$VAST_PORT"
echo "  log:              $LOG"
echo
echo "On vast.ai, point chroma client at localhost:$REMOTE_PORT:"
echo "  python3 -c \"import chromadb; print(chromadb.HttpClient(host='localhost', port=$REMOTE_PORT).heartbeat())\""
