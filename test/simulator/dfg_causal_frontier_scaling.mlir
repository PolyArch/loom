// RUN: not timeout --signal=TERM --kill-after=%loom-timeout-ultrafast %loom-timeout-ultrafast loom-dfg-sim %s --graph uncovered_nested_frontier --output %t.json 2>&1 | FileCheck %s

// The retirement query starts from the uncovered memory event. Unrelated
// selector diamonds must not make validation enumerate their path product.
// CHECK: retirement frontier does not causally cover dataflow.load done

module {
  dataflow.graph private @uncovered_nested_frontier(
      %start: none,
      %s0: i1, %s1: i1, %s2: i1, %s3: i1,
      %s4: i1, %s5: i1, %s6: i1, %s7: i1,
      %s8: i1, %s9: i1, %s10: i1, %s11: i1,
      %s12: i1, %s13: i1, %s14: i1, %s15: i1,
      %s16: i1, %s17: i1, %s18: i1, %s19: i1,
      %s20: i1, %s21: i1, %s22: i1,
      %index: index, %memory: memref<1xi32>) -> ()
      attributes {input_segments = array<i32: 24, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %value, %loaded = dataflow.load %memory[%index] %start : memref<1xi32>

    %d0:2 = dataflow.demux %s0, %start : (i1, none) -> (none, none)
    %m0 = dataflow.mux %s0, %d0#0, %d0#1 : (i1, none, none) -> none
    %d1:2 = dataflow.demux %s1, %m0 : (i1, none) -> (none, none)
    %m1 = dataflow.mux %s1, %d1#0, %d1#1 : (i1, none, none) -> none
    %d2:2 = dataflow.demux %s2, %m1 : (i1, none) -> (none, none)
    %m2 = dataflow.mux %s2, %d2#0, %d2#1 : (i1, none, none) -> none
    %d3:2 = dataflow.demux %s3, %m2 : (i1, none) -> (none, none)
    %m3 = dataflow.mux %s3, %d3#0, %d3#1 : (i1, none, none) -> none
    %d4:2 = dataflow.demux %s4, %m3 : (i1, none) -> (none, none)
    %m4 = dataflow.mux %s4, %d4#0, %d4#1 : (i1, none, none) -> none
    %d5:2 = dataflow.demux %s5, %m4 : (i1, none) -> (none, none)
    %m5 = dataflow.mux %s5, %d5#0, %d5#1 : (i1, none, none) -> none
    %d6:2 = dataflow.demux %s6, %m5 : (i1, none) -> (none, none)
    %m6 = dataflow.mux %s6, %d6#0, %d6#1 : (i1, none, none) -> none
    %d7:2 = dataflow.demux %s7, %m6 : (i1, none) -> (none, none)
    %m7 = dataflow.mux %s7, %d7#0, %d7#1 : (i1, none, none) -> none
    %d8:2 = dataflow.demux %s8, %m7 : (i1, none) -> (none, none)
    %m8 = dataflow.mux %s8, %d8#0, %d8#1 : (i1, none, none) -> none
    %d9:2 = dataflow.demux %s9, %m8 : (i1, none) -> (none, none)
    %m9 = dataflow.mux %s9, %d9#0, %d9#1 : (i1, none, none) -> none
    %d10:2 = dataflow.demux %s10, %m9 : (i1, none) -> (none, none)
    %m10 = dataflow.mux %s10, %d10#0, %d10#1 : (i1, none, none) -> none
    %d11:2 = dataflow.demux %s11, %m10 : (i1, none) -> (none, none)
    %m11 = dataflow.mux %s11, %d11#0, %d11#1 : (i1, none, none) -> none
    %d12:2 = dataflow.demux %s12, %m11 : (i1, none) -> (none, none)
    %m12 = dataflow.mux %s12, %d12#0, %d12#1 : (i1, none, none) -> none
    %d13:2 = dataflow.demux %s13, %m12 : (i1, none) -> (none, none)
    %m13 = dataflow.mux %s13, %d13#0, %d13#1 : (i1, none, none) -> none
    %d14:2 = dataflow.demux %s14, %m13 : (i1, none) -> (none, none)
    %m14 = dataflow.mux %s14, %d14#0, %d14#1 : (i1, none, none) -> none
    %d15:2 = dataflow.demux %s15, %m14 : (i1, none) -> (none, none)
    %m15 = dataflow.mux %s15, %d15#0, %d15#1 : (i1, none, none) -> none
    %d16:2 = dataflow.demux %s16, %m15 : (i1, none) -> (none, none)
    %m16 = dataflow.mux %s16, %d16#0, %d16#1 : (i1, none, none) -> none
    %d17:2 = dataflow.demux %s17, %m16 : (i1, none) -> (none, none)
    %m17 = dataflow.mux %s17, %d17#0, %d17#1 : (i1, none, none) -> none
    %d18:2 = dataflow.demux %s18, %m17 : (i1, none) -> (none, none)
    %m18 = dataflow.mux %s18, %d18#0, %d18#1 : (i1, none, none) -> none
    %d19:2 = dataflow.demux %s19, %m18 : (i1, none) -> (none, none)
    %m19 = dataflow.mux %s19, %d19#0, %d19#1 : (i1, none, none) -> none
    %d20:2 = dataflow.demux %s20, %m19 : (i1, none) -> (none, none)
    %m20 = dataflow.mux %s20, %d20#0, %d20#1 : (i1, none, none) -> none
    %d21:2 = dataflow.demux %s21, %m20 : (i1, none) -> (none, none)
    %m21 = dataflow.mux %s21, %d21#0, %d21#1 : (i1, none, none) -> none
    %d22:2 = dataflow.demux %s22, %m21 : (i1, none) -> (none, none)
    %m22 = dataflow.mux %s22, %d22#0, %d22#1 : (i1, none, none) -> none

    dataflow.graph.return values() streams() memories()
        complete(%m22 : none)
  }
}
