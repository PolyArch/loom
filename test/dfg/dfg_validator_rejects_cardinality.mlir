// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: loom-lower %t.dir/conditional-loop.mlir -o /dev/null
// RUN: not loom-lower %t.dir/value.mlir -o %t.dir/value.out.mlir 2>&1 | FileCheck %s --check-prefix=VALUE
// RUN: not loom-lower %t.dir/filtered-value.mlir -o %t.dir/filtered-value.out.mlir 2>&1 | FileCheck %s --check-prefix=FILTERED-VALUE
// RUN: not loom-lower %t.dir/stream.mlir -o %t.dir/stream.out.mlir 2>&1 | FileCheck %s --check-prefix=STREAM
// RUN: not loom-lower %t.dir/completion.mlir -o %t.dir/completion.out.mlir 2>&1 | FileCheck %s --check-prefix=COMPLETION
// RUN: not loom-lower %t.dir/direct-stream-rendezvous.mlir -o %t.dir/direct.out.mlir 2>&1 | FileCheck %s --check-prefix=DIRECT
// RUN: not loom-dfg-sim %t.dir/direct-stream-rendezvous.mlir --graph direct_stream_rendezvous --arg 0=7 --arg 0=9 --output %t.dir/direct.json 2>&1 | FileCheck %s --check-prefix=DIRECT
// RUN: not loom-dfg-sim %t.dir/stream-activated-schedule.mlir --graph stream_activated_schedule --arg 0=0 --arg 1=7 --output %t.dir/stream-activated.json 2>&1 | FileCheck %s --check-prefix=STREAM-ACTIVATED
// RUN: not loom-dfg-sim %t.dir/stateful-activated-schedule.mlir --graph stateful_activated_schedule --arg 0=7 --arg 0=9 --output %t.dir/stateful-activated.json 2>&1 | FileCheck %s --check-prefix=STATEFUL-ACTIVATED
// RUN: not loom-lower %t.dir/recursive-nested-activation.mlir -o %t.dir/recursive.out.mlir 2>&1 | FileCheck %s --check-prefix=RECURSIVE-NESTED
// RUN: not loom-lower %t.dir/parallelize-under-supplied.mlir -o %t.dir/parallelize-under-supplied.out.mlir 2>&1 | FileCheck %s --check-prefix=PARALLELIZE-UNDER-SUPPLIED
// RUN: not loom-lower %t.dir/serialize-under-supplied.mlir -o %t.dir/serialize-under-supplied.out.mlir 2>&1 | FileCheck %s --check-prefix=SERIALIZE-UNDER-SUPPLIED
// RUN: test ! -e %t.dir/value.out.mlir
// RUN: test ! -e %t.dir/filtered-value.out.mlir
// RUN: test ! -e %t.dir/stream.out.mlir
// RUN: test ! -e %t.dir/completion.out.mlir
// RUN: test ! -e %t.dir/direct.out.mlir
// RUN: test ! -e %t.dir/direct.json
// RUN: test ! -e %t.dir/stream-activated.json
// RUN: test ! -e %t.dir/stateful-activated.json
// RUN: test ! -e %t.dir/recursive.out.mlir
// RUN: test ! -e %t.dir/parallelize-under-supplied.out.mlir
// RUN: test ! -e %t.dir/serialize-under-supplied.out.mlir

// VALUE: graph @stream_to_value value output #0 is not statically exact-one
// FILTERED-VALUE: graph @filtered_stream_to_value value output #0 is not statically exact-one
// STREAM: graph @partial_stream_commit stream output #0 has no statically proven close/commit
// COMPLETION: graph @stream_driven_completion completion witness #0 is not statically one-shot
// DIRECT: graph @direct_stream_rendezvous value output #0 is not statically exact-one
// STREAM-ACTIVATED: graph @stream_activated_schedule value output #0 is not statically exact-one
// STATEFUL-ACTIVATED: graph @stateful_activated_schedule value output #0 is not statically exact-one
// RECURSIVE-NESTED: graph @recursive_nested_activation value output #0 is not statically exact-one
// PARALLELIZE-UNDER-SUPPLIED: graph @parallelize_under_supplied stream output #0 has no statically proven close/commit
// SERIALIZE-UNDER-SUPPLIED: graph @serialize_under_supplied stream output #0 has no statically proven close/commit

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

// A nested schedule cannot establish its own activation through its false
// close. The cyclic activation must fail cardinality analysis without
// recursively rebuilding the same nested analysis.
//--- recursive-nested-activation.mlir
module {
  dataflow.graph private @recursive_nested_activation(%start: none) -> i32
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %zero = dataflow.constant %start {const_value = 0 : i32} : i32
    %one = dataflow.constant %start {const_value = 1 : i32} : i32
    %outer_iv, %outer_phase = dataflow.stream %zero, %one, %one
        step add while slt : i32
    %inner_iv, %inner_phase = dataflow.stream %inner_close#0, %one, %one
        step add while slt : i32
    %inner_value = dataflow.invariant %inner_phase, %zero : i32
    %inner_close:2 = dataflow.demux %inner_phase, %inner_value
        : (i1, i32) -> (i32, i32)
    %outer_value = dataflow.carry %outer_phase, %zero, %inner_close#0 : i32
    %value_close:2 = dataflow.demux %outer_phase, %outer_value
        : (i1, i32) -> (i32, i32)
    %control = dataflow.invariant %outer_phase, %start : none
    %complete:2 = dataflow.demux %outer_phase, %control
        : (i1, none) -> (none, none)
    dataflow.graph.return values(%value_close#0 : i32) streams() memories()
        complete(%complete#0 : none)
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

// Filtering a fixed schedule preserves order but does not make a conditional
// lane exact-one.
//--- filtered-value.mlir
module {
  dataflow.graph private @filtered_stream_to_value(
      %start: none, %select: i1, %input: i32) -> i32
      attributes {input_segments = array<i32: 1, 1, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %zero = dataflow.constant %start {const_value = 0 : i32} : i32
    %two = dataflow.constant %start {const_value = 2 : i32} : i32
    %one = dataflow.constant %start {const_value = 1 : i32} : i32
    %iv, %phase = dataflow.stream %zero, %two, %one
        step add while slt : i32
    %control = dataflow.invariant %phase, %start : none
    %events:2 = dataflow.demux %phase, %control
        : (i1, none) -> (none, none)
    %static = arith.trunci %iv : i32 to i1
    %site_events:2 = dataflow.demux %static, %events#1
        : (i1, none) -> (none, none)
    %always = dataflow.constant %site_events#0
        {const_value = true} : i1
    %inactive = dataflow.constant %site_events#1
        {const_value = false} : i1
    %selected = dataflow.constant %site_events#1
        {const_value = true} : i1
    %conditional = dataflow.mux %select, %inactive, %selected
        : (i1, i1, i1) -> i1
    %active = dataflow.mux %static, %always, %conditional
        : (i1, i1, i1) -> i1
    %active_ordinals:2 = dataflow.demux %active, %iv
        : (i1, i32) -> (i32, i32)
    %route = arith.trunci %active_ordinals#1 : i32 to i1
    %lanes:2 = dataflow.demux %route, %input
        : (i1, i32) -> (i32, i32)
    dataflow.graph.return values(%lanes#1 : i32) streams() memories()
        complete(%events#0 : none)
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

// An exact-one activation does not bound a direct stream input. Both the
// publication gate and the simulator entry gate must reject the same graph.
//--- direct-stream-rendezvous.mlir
module {
  dataflow.graph private @direct_stream_rendezvous(
      %start: none, %input: i32) -> i32
      attributes {input_segments = array<i32: 0, 1, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %published:2 = dataflow.sync %start, %input
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%published#1 : i32) streams() memories()
        complete(%published#0 : none)
  }
}

// A stream token cannot become a one-shot schedule activation merely because
// arithmetic cancels its payload value.
//--- stream-activated-schedule.mlir
module {
  dataflow.graph private @stream_activated_schedule(
      %start: none, %activity: i32, %input: i32) -> i32
      attributes {input_segments = array<i32: 0, 2, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %zero = dataflow.constant %start {const_value = 0 : i32} : i32
    %one = dataflow.constant %start {const_value = 1 : i32} : i32
    %two = dataflow.constant %start {const_value = 2 : i32} : i32
    %masked = arith.subi %activity, %activity : i32
    %limit = arith.addi %two, %masked : i32
    %iv, %phase = dataflow.stream %zero, %limit, %one
        step add while slt : i32
    %route = arith.trunci %iv : i32 to i1
    %lanes:2 = dataflow.demux %route, %input
        : (i1, i32) -> (i32, i32)
    %complete = dataflow.sync %start : (none) -> none
    dataflow.graph.return values(%lanes#0 : i32) streams() memories()
        complete(%complete : none)
  }
}

// A stateful actor output is not a one-shot activation even when downstream
// arithmetic maps every payload to the same schedule domain.
//--- stateful-activated-schedule.mlir
module {
  dataflow.graph private @stateful_activated_schedule(
      %start: none, %input: i32) -> i32
      attributes {input_segments = array<i32: 0, 1, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %zero = dataflow.constant %start {const_value = 0 : i32} : i32
    %one = dataflow.constant %start {const_value = 1 : i32} : i32
    %two = dataflow.constant %start {const_value = 2 : i32} : i32
    %source_iv, %source_phase = dataflow.stream %zero, %two, %one
        step add while slt : i32
    %dynamic_zero = arith.subi %source_iv, %source_iv : i32
    %dynamic_one = arith.addi %dynamic_zero, %one : i32
    %iv, %phase = dataflow.stream %dynamic_zero, %dynamic_one, %dynamic_one
        step add while slt : i32
    %source_control = dataflow.invariant %source_phase, %start : none
    %source_close:2 = dataflow.demux %source_phase, %source_control
        : (i1, none) -> (none, none)
    %schedule_control = dataflow.invariant %phase, %start : none
    %schedule_close:2 = dataflow.demux %phase, %schedule_control
        : (i1, none) -> (none, none)
    %route = arith.trunci %iv : i32 to i1
    %lanes:2 = dataflow.demux %route, %input
        : (i1, i32) -> (i32, i32)
    %complete:2 = dataflow.sync %source_close#0, %schedule_close#0
        : (none, none) -> (none, none)
    dataflow.graph.return values(%lanes#0 : i32) streams() memories()
        complete(%complete#0 : none)
  }
}

// A scalar phase close cannot project through parallelize when only one data
// token is available for two true scalar phase items.
//--- parallelize-under-supplied.mlir
module {
  dataflow.graph private @parallelize_under_supplied(%start: none) -> i1
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 1, 0>} {
    %zero = dataflow.constant %start {const_value = 0 : i8} : i8
    %two = dataflow.constant %start {const_value = 2 : i8} : i8
    %one = dataflow.constant %start {const_value = 1 : i8} : i8
    %item, %scalar_phase = dataflow.stream %zero, %two, %one
        step add while ult : i8
    %only_item = dataflow.constant %start {const_value = 7 : i8} : i8
    %vector, %mask, %group_phase =
      dataflow.parallelize %only_item, %scalar_phase
        : (i8, i1) -> (vector<2xi8>, vector<2xi1>, i1)
    %units = dataflow.invariant %group_phase, %start : none
    %close:2 = dataflow.demux %group_phase, %units
        : (i1, none) -> (none, none)
    dataflow.graph.return values() streams(%group_phase : i1) memories()
        complete(%close#0 : none)
  }
}

// A group phase close cannot project through serialize when one vector and
// mask pair cannot cover both true group phase items.
//--- serialize-under-supplied.mlir
module {
  dataflow.graph private @serialize_under_supplied(%start: none) -> i1
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 1, 0>} {
    %zero = dataflow.constant %start {const_value = 0 : i8} : i8
    %two = dataflow.constant %start {const_value = 2 : i8} : i8
    %one = dataflow.constant %start {const_value = 1 : i8} : i8
    %ordinal, %group_phase = dataflow.stream %zero, %two, %one
        step add while ult : i8
    %packed = dataflow.constant %start {const_value = 513 : i16} : i16
    %packed_mask = dataflow.constant %start {const_value = 3 : i2} : i2
    %vector = dataflow.unpack %packed : i16 -> vector<2xi8>
    %mask = dataflow.unpack %packed_mask : i2 -> vector<2xi1>
    %scalar, %scalar_phase =
      dataflow.serialize %vector, %mask, %group_phase
        : (vector<2xi8>, vector<2xi1>, i1) -> (i8, i1)
    %units = dataflow.invariant %scalar_phase, %start : none
    %close:2 = dataflow.demux %scalar_phase, %units
        : (i1, none) -> (none, none)
    dataflow.graph.return values() streams(%scalar_phase : i1) memories()
        complete(%close#0 : none)
  }
}
