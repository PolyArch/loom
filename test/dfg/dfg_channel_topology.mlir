// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: loom-lower %t.dir/valid.mlir -o /dev/null
// RUN: not loom-lower %t.dir/missing-producer.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: not loom-lower %t.dir/missing-consumer.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=NO-CONSUMER
// RUN: not loom-lower %t.dir/unused.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNUSED
// RUN: not loom-lower %t.dir/duplicate-producer.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=DUPLICATE
// RUN: not loom-lower %t.dir/rank.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=RANK
// RUN: not loom-lower %t.dir/bounds.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=BOUNDS
// RUN: not loom-lower %t.dir/escape.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=ESCAPE

// MISSING: channel topology has no producer binding
// NO-CONSUMER: channel topology has no consumer binding
// UNUSED: channel topology has no producer binding
// DUPLICATE: channel topology has multiple producer bindings
// RANK: source_map result rank 0 does not match producer rank 1
// BOUNDS: source_map result #0 is not in bounds for the producer domain
// ESCAPE: channel operand is not a permitted binding of 'func.call'

//--- common.mlir

//--- valid.mlir
module {
  dataflow.graph private @produce(
      %start: none, %init: i32, %limit: i32, %step: i32) -> i32
      attributes {input_segments = array<i32: 3, 0, 0>,
                  result_segments = array<i32: 0, 1, 0>} {
    %iv, %phase = dataflow.stream %init, %limit, %step
        step add while slt : i32
    %units = dataflow.invariant %phase, %start : none
    %close:2 = dataflow.demux %phase, %units
        : (i1, none) -> (none, none)
    dataflow.graph.return values() streams(%iv : i32) memories()
        complete(%close#0 : none)
  }

  dataflow.graph private @consume(%start: none, %input: i32) -> ()
      attributes {input_segments = array<i32: 0, 1, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return %start : none
  }

  dataflow.thread private @producer(
      %channel: !dataflow.channel<i32>,
      %init: i32, %limit: i32, %step: i32)
      ctrl (%ctrl: none) iv (%iv: index) {
    %done = dataflow.graph.launch @produce deps(%ctrl)
        values(%init, %limit, %step) stream_inputs() memories()
        stream_outputs(%channel)
        : (none, i32, i32, i32, !dataflow.channel<i32>) -> none
    dataflow.thread.yield %done : none
  }

  dataflow.thread private @consumer(
      %channel: !dataflow.channel<i32>)
      ctrl (%ctrl: none) iv (%iv: index) {
    %done = dataflow.graph.launch @consume deps(%ctrl) values()
        stream_inputs(%channel source_map affine_map<(d0) -> (d0)>)
        memories() stream_outputs()
        : (none, !dataflow.channel<i32>) -> none
    dataflow.thread.yield %done : none
  }

  func.func @valid(%channel: !dataflow.channel<i32>, %extent: index) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %limit = arith.index_cast %extent : index to i32
    %producer = dataflow.thread.launch @producer(
        %channel, %c0, %limit, %c1) grid(%extent)
        : (!dataflow.channel<i32>, i32, i32, i32)
          -> !dataflow.thread_token
    %consumer = dataflow.thread.launch @consumer(%channel) grid(%extent)
        : (!dataflow.channel<i32>) -> !dataflow.thread_token
    return
  }

}

//--- missing-producer.mlir
module {
  dataflow.graph private @consume(%start: none, %input: i32) -> ()
      attributes {input_segments = array<i32: 0, 1, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return %start : none
  }
  dataflow.thread private @consumer(
      %channel: !dataflow.channel<i32>)
      ctrl (%ctrl: none) iv (%iv: index) {
    %done = dataflow.graph.launch @consume deps(%ctrl) values()
        stream_inputs(%channel source_map affine_map<(d0) -> (d0)>)
        memories() stream_outputs()
        : (none, !dataflow.channel<i32>) -> none
    dataflow.thread.yield %done : none
  }
  func.func @missing_producer(
      %channel: !dataflow.channel<i32>, %extent: index) {
    %consumer = dataflow.thread.launch @consumer(%channel) grid(%extent)
        : (!dataflow.channel<i32>) -> !dataflow.thread_token
    return
  }
}

//--- duplicate-producer.mlir
module {
  dataflow.graph private @produce(
      %start: none, %init: i32, %limit: i32, %step: i32) -> i32
      attributes {input_segments = array<i32: 3, 0, 0>,
                  result_segments = array<i32: 0, 1, 0>} {
    %iv, %phase = dataflow.stream %init, %limit, %step
        step add while slt : i32
    %units = dataflow.invariant %phase, %start : none
    %close:2 = dataflow.demux %phase, %units
        : (i1, none) -> (none, none)
    dataflow.graph.return values() streams(%iv : i32) memories()
        complete(%close#0 : none)
  }
  dataflow.thread private @producer(
      %channel: !dataflow.channel<i32>,
      %init: i32, %limit: i32, %step: i32)
      ctrl (%ctrl: none) iv (%iv: index) {
    %done = dataflow.graph.launch @produce deps(%ctrl)
        values(%init, %limit, %step) stream_inputs() memories()
        stream_outputs(%channel)
        : (none, i32, i32, i32, !dataflow.channel<i32>) -> none
    dataflow.thread.yield %done : none
  }
  func.func @duplicate_producer(
      %channel: !dataflow.channel<i32>, %extent: index) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %limit = arith.index_cast %extent : index to i32
    %first = dataflow.thread.launch @producer(
        %channel, %c0, %limit, %c1) grid(%extent)
        : (!dataflow.channel<i32>, i32, i32, i32)
          -> !dataflow.thread_token
    %second = dataflow.thread.launch @producer(
        %channel, %c0, %limit, %c1) grid(%extent)
        : (!dataflow.channel<i32>, i32, i32, i32)
          -> !dataflow.thread_token
    return
  }
}

//--- missing-consumer.mlir
module {
  dataflow.graph private @produce(
      %start: none, %init: i32, %limit: i32, %step: i32) -> i32
      attributes {input_segments = array<i32: 3, 0, 0>,
                  result_segments = array<i32: 0, 1, 0>} {
    %iv, %phase = dataflow.stream %init, %limit, %step
        step add while slt : i32
    %units = dataflow.invariant %phase, %start : none
    %close:2 = dataflow.demux %phase, %units
        : (i1, none) -> (none, none)
    dataflow.graph.return values() streams(%iv : i32) memories()
        complete(%close#0 : none)
  }
  dataflow.thread private @producer(
      %channel: !dataflow.channel<i32>,
      %init: i32, %limit: i32, %step: i32)
      ctrl (%ctrl: none) iv (%iv: index) {
    %done = dataflow.graph.launch @produce deps(%ctrl)
        values(%init, %limit, %step) stream_inputs() memories()
        stream_outputs(%channel)
        : (none, i32, i32, i32, !dataflow.channel<i32>) -> none
    dataflow.thread.yield %done : none
  }
  func.func @missing_consumer(
      %channel: !dataflow.channel<i32>, %extent: index) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %limit = arith.index_cast %extent : index to i32
    %producer = dataflow.thread.launch @producer(
        %channel, %c0, %limit, %c1) grid(%extent)
        : (!dataflow.channel<i32>, i32, i32, i32)
          -> !dataflow.thread_token
    return
  }
}

//--- unused.mlir
module {
  func.func @unused(%channel: !dataflow.channel<i32>) {
    return
  }
}

//--- rank.mlir
module {
  dataflow.graph private @produce(
      %start: none, %init: i32, %limit: i32, %step: i32) -> i32
      attributes {input_segments = array<i32: 3, 0, 0>,
                  result_segments = array<i32: 0, 1, 0>} {
    %iv, %phase = dataflow.stream %init, %limit, %step
        step add while slt : i32
    %units = dataflow.invariant %phase, %start : none
    %close:2 = dataflow.demux %phase, %units
        : (i1, none) -> (none, none)
    dataflow.graph.return values() streams(%iv : i32) memories()
        complete(%close#0 : none)
  }
  dataflow.graph private @consume(%start: none, %input: i32) -> ()
      attributes {input_segments = array<i32: 0, 1, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return %start : none
  }
  dataflow.thread private @producer(
      %channel: !dataflow.channel<i32>,
      %init: i32, %limit: i32, %step: i32)
      ctrl (%ctrl: none) iv (%iv: index) {
    %done = dataflow.graph.launch @produce deps(%ctrl)
        values(%init, %limit, %step) stream_inputs() memories()
        stream_outputs(%channel)
        : (none, i32, i32, i32, !dataflow.channel<i32>) -> none
    dataflow.thread.yield %done : none
  }
  dataflow.thread private @consumer(
      %channel: !dataflow.channel<i32>)
      ctrl (%ctrl: none) iv (%iv: index) {
    %done = dataflow.graph.launch @consume deps(%ctrl) values()
        stream_inputs(%channel source_map affine_map<(d0) -> ()>)
        memories() stream_outputs()
        : (none, !dataflow.channel<i32>) -> none
    dataflow.thread.yield %done : none
  }
  func.func @rank_mismatch(
      %channel: !dataflow.channel<i32>, %extent: index) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %limit = arith.index_cast %extent : index to i32
    %producer = dataflow.thread.launch @producer(
        %channel, %c0, %limit, %c1) grid(%extent)
        : (!dataflow.channel<i32>, i32, i32, i32)
          -> !dataflow.thread_token
    %consumer = dataflow.thread.launch @consumer(%channel) grid(%extent)
        : (!dataflow.channel<i32>) -> !dataflow.thread_token
    return
  }
}

//--- bounds.mlir
module {
  dataflow.graph private @produce(
      %start: none, %init: i32, %limit: i32, %step: i32) -> i32
      attributes {input_segments = array<i32: 3, 0, 0>,
                  result_segments = array<i32: 0, 1, 0>} {
    %iv, %phase = dataflow.stream %init, %limit, %step
        step add while slt : i32
    %units = dataflow.invariant %phase, %start : none
    %close:2 = dataflow.demux %phase, %units
        : (i1, none) -> (none, none)
    dataflow.graph.return values() streams(%iv : i32) memories()
        complete(%close#0 : none)
  }
  dataflow.graph private @consume(%start: none, %input: i32) -> ()
      attributes {input_segments = array<i32: 0, 1, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return %start : none
  }
  dataflow.thread private @producer(
      %channel: !dataflow.channel<i32>,
      %init: i32, %limit: i32, %step: i32)
      ctrl (%ctrl: none) iv (%iv: index) {
    %done = dataflow.graph.launch @produce deps(%ctrl)
        values(%init, %limit, %step) stream_inputs() memories()
        stream_outputs(%channel)
        : (none, i32, i32, i32, !dataflow.channel<i32>) -> none
    dataflow.thread.yield %done : none
  }
  dataflow.thread private @consumer(
      %channel: !dataflow.channel<i32>)
      ctrl (%ctrl: none) iv (%iv: index) {
    %done = dataflow.graph.launch @consume deps(%ctrl) values()
        stream_inputs(%channel source_map affine_map<(d0) -> (d0)>)
        memories() stream_outputs()
        : (none, !dataflow.channel<i32>) -> none
    dataflow.thread.yield %done : none
  }
  func.func @out_of_bounds(%channel: !dataflow.channel<i32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %g2 = arith.constant 2 : index
    %g4 = arith.constant 4 : index
    %producer = dataflow.thread.launch @producer(
        %channel, %c0, %c2, %c1) grid(%g2)
        : (!dataflow.channel<i32>, i32, i32, i32)
          -> !dataflow.thread_token
    %consumer = dataflow.thread.launch @consumer(%channel) grid(%g4)
        : (!dataflow.channel<i32>) -> !dataflow.thread_token
    return
  }
}

//--- escape.mlir
module {
  func.func private @sink(!dataflow.channel<i32>)
  func.func @escape(%channel: !dataflow.channel<i32>) {
    func.call @sink(%channel) : (!dataflow.channel<i32>) -> ()
    return
  }
}
