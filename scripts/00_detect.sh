#!/usr/bin/env bash
set -u
echo "=== python3 ==="
python3 --version
echo "=== pip module ==="
python3 -m pip --version 2>&1 | head -2
echo "=== venv module ==="
if python3 -m venv --help >/dev/null 2>&1; then echo "venv OK"; else echo "venv MISSING"; fi
echo "=== ensurepip ==="
python3 - <<'PY'
try:
    import ensurepip
    print("ensurepip OK")
except Exception as e:
    print("ensurepip MISSING:", e)
PY
echo "=== apt packages ==="
dpkg -l 2>/dev/null | grep -E 'python3-venv|python3-pip|python3.12-venv' | awk '{print $2, $3}'
echo "=== sudo (non-interactive) ==="
if sudo -n true 2>/dev/null; then echo "sudo NOPASSWD OK"; else echo "sudo needs password"; fi
