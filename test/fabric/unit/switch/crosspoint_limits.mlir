// RUN: loom %s -split-input-file -verify-diagnostics

// A low-crosspoint asymmetric selector remains quiet.
fabric.switch @AsymmetricQuiet [spatial]
    (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
     !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
     !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
    -> (!fabric.bits<32>)
    [{connectivity_table = ["111111111"]}]

// -----

// The product, not either dimension alone, triggers the warning.
// expected-warning @+1 {{fabric.switch crossbar has 72 crosspoints; values above 64 may be implementation-inefficient}}
fabric.switch @ProductWarning [spatial]
    (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
     !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
     !fabric.bits<32>, !fabric.bits<32>)
    -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
        !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
        !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
    [{connectivity_table = ["11111111", "11111111", "11111111",
                            "11111111", "11111111", "11111111",
                            "11111111", "11111111", "11111111"]}]

// -----

// The hard boundary remains valid and warns.
// expected-warning @+1 {{fabric.switch crossbar has 256 crosspoints; values above 64 may be implementation-inefficient}}
fabric.switch @ProductBoundary [spatial]
    (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
     !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
     !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
     !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
    -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
        !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
        !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
        !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
    [{connectivity_table = ["1111111111111111", "1111111111111111",
                            "1111111111111111", "1111111111111111",
                            "1111111111111111", "1111111111111111",
                            "1111111111111111", "1111111111111111",
                            "1111111111111111", "1111111111111111",
                            "1111111111111111", "1111111111111111",
                            "1111111111111111", "1111111111111111",
                            "1111111111111111", "1111111111111111"]}]

// -----

// expected-error @+1 {{switch crossbar has 272 crosspoints, exceeding maximum 256}}
fabric.switch @ProductOverflow [spatial]
    (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
     !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
     !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
     !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
     !fabric.bits<32>)
    -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
        !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
        !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
        !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
    [{connectivity_table = ["11111111111111111", "11111111111111111",
                            "11111111111111111", "11111111111111111",
                            "11111111111111111", "11111111111111111",
                            "11111111111111111", "11111111111111111",
                            "11111111111111111", "11111111111111111",
                            "11111111111111111", "11111111111111111",
                            "11111111111111111", "11111111111111111",
                            "11111111111111111", "11111111111111111"]}]
