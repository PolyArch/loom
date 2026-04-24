// Verifies that LOOM_INDEX_WIDTH overrides the default index width (32).
// With LOOM_INDEX_WIDTH=16, a 4-input mux must use bits<16> for sel; bits<32>
// should be rejected.

// RUN: env LOOM_INDEX_WIDTH=16 loom %s -split-input-file -verify-diagnostics

func.func @mux_index_env_ok(%sel: !fabric.bits<16>,
                             %a: !fabric.bits<8>, %b: !fabric.bits<8>,
                             %c: !fabric.bits<8>, %d: !fabric.bits<8>) -> !fabric.bits<8> {
  %0 = fabric.op [@dataflow.mux] (%sel, %a, %b, %c, %d)
       : (!fabric.bits<16>, !fabric.bits<8>, !fabric.bits<8>,
          !fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
  return %0 : !fabric.bits<8>
}

// -----

func.func @mux_index_env_rejects_default(%sel: !fabric.bits<32>,
                                          %a: !fabric.bits<8>, %b: !fabric.bits<8>,
                                          %c: !fabric.bits<8>, %d: !fabric.bits<8>) -> !fabric.bits<8> {
  // expected-error @+1 {{sel port (input #0) width 32 must be 16}}
  %0 = fabric.op [@dataflow.mux] (%sel, %a, %b, %c, %d)
       : (!fabric.bits<32>, !fabric.bits<8>, !fabric.bits<8>,
          !fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
  return %0 : !fabric.bits<8>
}
