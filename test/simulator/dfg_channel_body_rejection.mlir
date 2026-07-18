// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: not loom-dfg-sim %t.dir/poison.mlir --graph poison_channel --output %t.poison.json 2>&1 | FileCheck %s --check-prefix=POISON
// RUN: not loom-dfg-sim %t.dir/call.mlir --graph call_channel --output %t.call.json 2>&1 | FileCheck %s --check-prefix=CALL

// POISON: finalized program contains channel producer 'ub.poison'
// CALL: finalized program contains channel producer 'func.call'

//--- poison.mlir
module {
  dataflow.graph private @poison_channel(%start: none) -> () {
    %channel = ub.poison : !dataflow.channel<i32>
    dataflow.graph.return %start : none
  }
}

//--- call.mlir
module {
  func.func private @make_channel() -> !dataflow.channel<i32>

  dataflow.graph private @call_channel(%start: none) -> () {
    %channel = func.call @make_channel() : () -> !dataflow.channel<i32>
    dataflow.graph.return %start : none
  }
}
