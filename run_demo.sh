#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
exec "${SAARTHI_PYTHON:-python}" -m saarthi_app "$@"
