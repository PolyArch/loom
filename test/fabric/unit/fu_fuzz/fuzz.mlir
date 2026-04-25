// RUN: env LOOM_BIN=loom LOOM_FUZZ_FUS=12 LOOM_FUZZ_SEED=12345 \
// RUN:   python3 %S/fuzz_match.py | FileCheck %s

// CHECK: OK:
