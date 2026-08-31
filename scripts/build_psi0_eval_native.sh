#!/usr/bin/env bash
# scripts/build_psi0_eval_native.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CARGO_HOME="${CARGO_HOME:-/tmp/psi0-eval-cargo-home}"
export CARGO_NET_OFFLINE=true
cd "$root/native/psi0_eval_runtime"
cargo build --offline --locked --release \
  --manifest-path "$root/native/psi0_eval_runtime/Cargo.toml" \
  --target x86_64-unknown-linux-gnu
for name in \
  psi0-eval-install-input \
  psi0-eval-install-pc2-input \
  psi0-eval-remote-helper \
  psi0-eval-run-pc2-evaluator \
  psi0-eval-policy-relay
do
  binary="$root/native/psi0_eval_runtime/target/x86_64-unknown-linux-gnu/release/$name"
  test -x "$binary"
  if readelf -lWd "$binary" | grep -Eq 'INTERP|NEEDED'; then
    echo "$name is not a static no-dependency executable" >&2
    exit 1
  fi
done
