// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// memref I/O is not allowed on subgraph (reserved for graph).
func.func @subgraph_rejects_memref_input(%mem: memref<8xi32>) {
  // expected-error @+1 {{operand #0 must be variadic of}}
  dataflow.subgraph(%m = %mem : memref<8xi32>) -> () {
    dataflow.yield
  }
  return
}

// -----
// llvm ops are not in fabric.op's allowlist.
func.func @subgraph_rejects_llvm(%a: i32, %b: i32) -> i32 {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    // expected-error @+1 {{is not allowed inside dataflow.subgraph}}
    %s = llvm.add %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}

// -----
// ub.poison is not in fabric.op's allowlist.
func.func @subgraph_rejects_ub_poison(%a: i32) -> i32 {
  %r = dataflow.subgraph(%x = %a : i32) -> i32 {
    // expected-error @+1 {{is not allowed inside dataflow.subgraph}}
    %p = ub.poison : i32
    dataflow.yield %x : i32
  }
  return %r : i32
}

// -----
// arith.constant is forbidden (use dataflow.constant via fabric.op).
func.func @subgraph_rejects_arith_constant() {
  dataflow.subgraph() -> () {
    // expected-error @+1 {{is not allowed inside dataflow.subgraph}}
    %c = arith.constant 1 : i32
    dataflow.yield
  }
  return
}

// -----
// dataflow.graph cannot be nested inside subgraph either.
func.func @subgraph_rejects_nested_graph(%x: i32) {
  dataflow.subgraph(%a = %x : i32) -> () {
    // expected-error @+1 {{is not allowed inside dataflow.subgraph}}
    dataflow.graph(%b = %a : i32) -> () {
      dataflow.yield
    }
    dataflow.yield
  }
  return
}

// -----
// dataflow.subgraph cannot be nested inside subgraph (subgraphs are leaves).
func.func @subgraph_rejects_nested_subgraph(%x: i32) {
  dataflow.subgraph(%a = %x : i32) -> () {
    // expected-error @+1 {{is not allowed inside dataflow.subgraph}}
    dataflow.subgraph(%b = %a : i32) -> () {
      dataflow.yield
    }
    dataflow.yield
  }
  return
}
