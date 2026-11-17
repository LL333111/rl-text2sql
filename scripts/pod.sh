#!/usr/bin/env bash
# Helpers for working with the Runpod pod through the SSH proxy (which gives a
# PTY shell only: no scp, no exec). Files travel as base64 over the terminal.
#
#   POD_SSH=<podid>-<suffix>@ssh.runpod.io  POD_KEY=~/.ssh/<key>
#   scripts/pod.sh push            # send tracked code (no results/) to /workspace/rl-text2sql
#   scripts/pod.sh pull <path>...  # fetch files/dirs under the repo root on the pod
#   scripts/pod.sh wait <file> <marker> [seconds]   # block until <marker> appears in <file>
set -euo pipefail
: "${POD_SSH:?set POD_SSH}" "${POD_KEY:?set POD_KEY}"
REPO=/workspace/rl-text2sql
ssh_pty() { timeout "${1}" ssh -tt -i "$POD_KEY" -o IdentitiesOnly=yes -o ServerAliveInterval=30 \
              -o StrictHostKeyChecking=accept-new "$POD_SSH" 2>&1; }

case "$1" in
  push)
    tmp=$(mktemp)
    git archive --format=tar.gz -o "$tmp" HEAD -- . ':!results' ':!PLAN.md'
    sum=$(md5sum "$tmp" | cut -c1-32 | tr -d '\\')
    { echo "stty -echo; base64 -d > /workspace/code.tgz <<'EOF'"; base64 -w 76 "$tmp"; echo EOF
      echo "echo; echo SUM=\$(md5sum /workspace/code.tgz | cut -c1-32); cd $REPO && tar xzf /workspace/code.tgz && sed -i 's/\r\$//' scripts/*.sh && echo PUSH_OK; exit"
    } | ssh_pty 300 | grep -E 'SUM=|PUSH_OK' | tr -d '\r' | tee /dev/stderr | grep -q "SUM=$sum"
    rm -f "$tmp" ;;
  pull)
    shift
    out=$(mktemp)
    { echo "stty -echo; cd $REPO; echo; echo ==B64==; tar czf - $* | base64 -w 76; echo ==END==; exit"
    } | ssh_pty 600 | tr -d '\r' | sed -n '/^==B64==$/,/^==END==$/p' | sed '1d;$d' > "$out"
    base64 -d "$out" | tar xzf -
    rm -f "$out"; echo "pulled: $*" ;;
  wait)
    file=$2; marker=$3; secs=${4:-3000}
    { echo "stty -echo; until grep -q '$marker' $file 2>/dev/null; do sleep 15; done; echo; echo ==TAIL==; tail -n 25 $file; exit"
    } | ssh_pty "$secs" | tr -d '\r' | sed -n '/^==TAIL==$/,$p' ;;
  *) echo "usage: pod.sh push|pull|wait" >&2; exit 2 ;;
esac
