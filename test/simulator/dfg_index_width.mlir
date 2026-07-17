// RUN: env LOOM_INDEX_WIDTH=32 loom-dfg-sim %s --graph index_width_fallback --arg 0=none --arg 1=4294967296 --memref 2=0 --output %t.fallback32.json
// RUN: FileCheck %s --check-prefix=FALLBACK32 < %t.fallback32.json
// RUN: env LOOM_INDEX_WIDTH=64 loom-dfg-sim %s --graph index_width_fallback --arg 0=none --arg 1=4294967296 --memref 2=0 --output %t.fallback64.json
// RUN: FileCheck %s --check-prefix=FALLBACK64 < %t.fallback64.json
// RUN: env LOOM_INDEX_WIDTH=64 loom-dfg-sim %s --graph index_width_explicit32 --arg 0=none --arg 1=4294967296 --memref 2=0 --output %t.explicit32.json
// RUN: FileCheck %s --check-prefix=EXPLICIT32 < %t.explicit32.json
// RUN: loom-dfg-sim %s --graph invalid_index_width_with_stream --arg 0=none --arg 1=1 --arg 2=0 --arg 3=64 --arg 4=1 --output %t.invalid.json
// RUN: FileCheck %s --check-prefix=INVALID < %t.invalid.json
// RUN: grep -c 'index bit width must be in \[1, 64\], got 128' %t.invalid.json | FileCheck %s --check-prefix=INVALID-COUNT

// FALLBACK32-DAG: "index:0"
// FALLBACK32-DAG: "!llvm.ptr:ptr+4"

// FALLBACK64-DAG: "index:4294967296"
// FALLBACK64-DAG: "!llvm.ptr:ptr+8"

// EXPLICIT32-DAG: "index:0"
// EXPLICIT32-DAG: "!llvm.ptr:ptr+4"

// INVALID-DAG: "status": "blocked"
// INVALID-DAG: "dataflow.stream": 65
// INVALID-COUNT: 1

module {
  dataflow.graph.func private @index_width_fallback(
      %ctrl: none, %value: i64, %base: !llvm.ptr)
      -> (none, index, !llvm.ptr) {
    %index = arith.index_cast %value : i64 to index
    %next = llvm.getelementptr %base[1]
        : (!llvm.ptr) -> !llvm.ptr, index
    dataflow.graph.return %ctrl, %index, %next : none, index, !llvm.ptr
  }

  module attributes {
    dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
  } {
    dataflow.graph.func private @index_width_explicit32(
        %ctrl: none, %value: i64, %base: !llvm.ptr)
        -> (none, index, !llvm.ptr) {
      %index = arith.index_cast %value : i64 to index
      %next = llvm.getelementptr %base[1]
          : (!llvm.ptr) -> !llvm.ptr, index
      dataflow.graph.return %ctrl, %index, %next : none, index, !llvm.ptr
    }
  }

  module attributes {
    dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 128>>
  } {
    dataflow.graph.func private @invalid_index_width_with_stream(
        %ctrl: none, %value: i64, %init: i64, %limit: i64, %step: i64)
        -> (none, index, i1) {
      %index = arith.index_cast %value : i64 to index
      %iv, %phase = dataflow.stream %init, %limit, %step
          {step_op = "+=", cont_cond = "<"} : i64
      dataflow.graph.return %ctrl, %index, %phase : none, index, i1
    }
  }
}
