// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: loom-lower %t.dir/conditional-loop.mlir -o /dev/null
// RUN: not loom-lower %t.dir/value.mlir -o %t.dir/value.out.mlir 2>&1 | FileCheck %s --check-prefix=VALUE
// RUN: not loom-lower %t.dir/stream.mlir -o %t.dir/stream.out.mlir 2>&1 | FileCheck %s --check-prefix=STREAM
// RUN: not loom-lower %t.dir/completion.mlir -o %t.dir/completion.out.mlir 2>&1 | FileCheck %s --check-prefix=COMPLETION
// RUN: test ! -e %t.dir/value.out.mlir
// RUN: test ! -e %t.dir/stream.out.mlir
// RUN: test ! -e %t.dir/completion.out.mlir

// VALUE: graph @stream_to_value value output #0 is not statically exact-one
// STREAM: graph @partial_stream_commit stream output #0 has no statically proven close/commit
// COMPLETION: graph @stream_driven_completion completion witness #0 is not statically one-shot

// A loop selected by one branch has one close event only on that branch. The
// final mux publishes that close or the one-shot bypass from the same outer
// selector, so the merged completion remains exact-one.
//--- conditional-loop.mlir
module {
  dataflow.graph private @conditional_loop(
      %start: none, %count: i32) -> ()
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    %zero = dataflow.constant %start {const_value = 0 : i32} : i32
    %one = dataflow.constant %start {const_value = 1 : i32} : i32
    %empty = arith.cmpi eq, %count, %zero : i32
    %starts:2 = dataflow.demux %empty, %start
        : (i1, none) -> (none, none)
    %lowers:2 = dataflow.demux %empty, %zero
        : (i1, i32) -> (i32, i32)
    %limits:2 = dataflow.demux %empty, %count
        : (i1, i32) -> (i32, i32)
    %steps:2 = dataflow.demux %empty, %one
        : (i1, i32) -> (i32, i32)
    %iv, %phase = dataflow.stream %lowers#0, %limits#0, %steps#0
        step add while slt : i32
    %control = dataflow.carry %phase, %starts#0, %lanes#1 : none
    %lanes:2 = dataflow.demux %phase, %control
        : (i1, none) -> (none, none)
    %complete = dataflow.mux %empty, %lanes#0, %starts#1
        : (i1, none, none) -> none
    dataflow.graph.return %complete : none
  }
}

// A direct stream synchronized only with scalar data is not execution-bounded.
//--- value.mlir
module {
  dataflow.graph private @stream_to_value(
      %start: none, %scalar: i32, %input: i32) -> i32
      attributes {input_segments = array<i32: 1, 1, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %published:2 = dataflow.sync %scalar, %input
        : (i32, i32) -> (i32, i32)
    %complete = dataflow.sync %start : (none) -> none
    dataflow.graph.return values(%published#1 : i32) streams() memories()
        complete(%complete : none)
  }
}

// Multiple direct stream inputs do not form one receive or one committed
// output stream.
//--- stream.mlir
module {
  dataflow.graph private @partial_stream_commit(
      %start: none, %input: i32, %other: i32) -> i32
      attributes {input_segments = array<i32: 0, 2, 0>,
                  result_segments = array<i32: 0, 1, 0>} {
    %published:3 = dataflow.sync %start, %input, %other
        : (none, i32, i32) -> (none, i32, i32)
    dataflow.graph.return values() streams(%published#1 : i32) memories()
        complete(%published#0 : none)
  }
}

// A completion path sourced from a zero-or-more stream is not one-shot.
//--- completion.mlir
module {
  dataflow.graph private @stream_driven_completion(
      %start: none, %input: none, %other: none) -> ()
      attributes {input_segments = array<i32: 0, 2, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    %published:3 = dataflow.sync %start, %input, %other
        : (none, none, none) -> (none, none, none)
    dataflow.graph.return values() streams() memories()
        complete(%published#0 : none)
  }
}
