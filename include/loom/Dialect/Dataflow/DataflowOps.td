#ifndef LOOM_DIALECT_DATAFLOW_DATAFLOWOPS_TD
#define LOOM_DIALECT_DATAFLOW_DATAFLOWOPS_TD

include "loom/Dialect/Dataflow/DataflowDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Dataflow_Op<string mnemonic, list<Trait> traits = []>
    : Op<Dataflow_Dialect, mnemonic, traits>;

// dataflow.stream: generates index + will_continue streams from loop bounds
def Dataflow_StreamOp : Dataflow_Op<"stream", []> {
  let summary = "Generate index and control streams from loop bounds";
  let description = [{
    Generates an index stream [start, start+step, start+2*step, ...] and a
    will_continue boolean stream. The index stream has N+1 elements for N
    loop body iterations; will_continue has N true followed by 1 false.

    step_op: "+=", "-=", "*=", "/=", "<<=", ">>="
    cont_cond: "<", "<=", ">", ">=", "!="
  }];

  let arguments = (ins
    Index:$start,
    Index:$step,
    Index:$bound,
    StrAttr:$step_op,
    StrAttr:$cont_cond
  );
  let results = (outs
    Index:$index,
    I1:$willContinue
  );

  let assemblyFormat = [{
    $start `,` $step `,` $bound
    attr-dict `:` `(` type($start) `,` type($step) `,` type($bound) `)`
    `->` `(` type($index) `,` type($willContinue) `)`
  }];
}

// dataflow.gate: before-to-after stream adapter
def Dataflow_GateOp : Dataflow_Op<"gate", []> {
  let summary = "Gate streams to separate before-region from body";
  let description = [{
    Takes a before-region value stream (N+1 elements) and a will_continue
    stream (N+1 elements), produces body value stream (N elements) and
    body condition stream (N elements) with one-element shift alignment.
  }];

  let arguments = (ins
    AnyType:$beforeValue,
    I1:$beforeCond
  );
  let results = (outs
    AnyType:$afterValue,
    I1:$afterCond
  );

  let assemblyFormat = [{
    $beforeValue `,` $beforeCond attr-dict
    `:` type($beforeValue) `,` type($beforeCond)
    `->` type($afterValue) `,` type($afterCond)
  }];
}

// dataflow.carry: loop-carried dependency
def Dataflow_CarryOp : Dataflow_Op<"carry", [RecursiveMemoryEffects]> {
  let summary = "Loop-carried dependency state machine";
  let description = [{
    Initial stage: consume %a and emit it. Move to block stage.
    Block stage: if %d is true, consume %b and emit %b.
    If %d is false, return to initial stage.
  }];

  let arguments = (ins
    I1:$d,
    AnyType:$a,
    AnyType:$b
  );
  let results = (outs
    AnyType:$o
  );

  let assemblyFormat = [{
    $d `,` $a `,` $b attr-dict
    `:` type($d) `,` type($a) `,` type($b) `->` type($o)
  }];
}

// dataflow.invariant: loop-invariant repeater
def Dataflow_InvariantOp : Dataflow_Op<"invariant", []> {
  let summary = "Loop-invariant token repeater";
  let description = [{
    Stores a value and repeats it each iteration while control is true.
    When control becomes false, releases the stored value.
  }];

  let arguments = (ins
    I1:$d,
    AnyType:$a
  );
  let results = (outs
    AnyType:$o
  );

  let assemblyFormat = [{
    $d `,` $a attr-dict `:` type($d) `,` type($a) `->` type($o)
  }];
}

#endif
