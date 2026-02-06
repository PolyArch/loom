//===- DataflowOps.td - Dataflow operations definition --------*- tablegen -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_DIALECT_DATAFLOW_DATAFLOWOPS_TD
#define LOOM_DIALECT_DATAFLOW_DATAFLOWOPS_TD

include "loom/Dialect/Dataflow/DataflowDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Dataflow_Op<string mnemonic, list<Trait> traits = []>
    : Op<Dataflow_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// dataflow.carry
//===----------------------------------------------------------------------===//

def Dataflow_CarryOp : Dataflow_Op<"carry", [
    RecursiveMemoryEffects
  ]> {
  let summary = "Loop-carried dependency state machine";
  let description = [{
    A loop-carried state machine that produces a value stream aligned to a
    control stream. It has three inputs: d (control), a (initial value),
    b (loop-carried value), and one output o.

    Semantics:
    - Initial stage: consume a, emit a on o, and move to block stage.
      d is not consumed in this step.
    - Block stage: consume each d. If d is true, consume b and emit b on o.
      If d is false, consume only d and return to initial stage.

    The length of o matches the length of d for each iteration burst.
    Example (T=true, F=false):
      a: [A, B]
      b: [C, D, E, F, G, H]
      d: [T, T, F, T, T, T, T, F]
      o: [A, C, D, B, E, F, G, H]
  }];
  let arguments = (ins I1:$d, AnyType:$a, AnyType:$b);
  let results = (outs AnyType:$o);

  let assemblyFormat = "$d `,` $a `,` $b attr-dict `:` type($d) `,` type($a) `,` type($b) `->` type($o)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// dataflow.invariant
//===----------------------------------------------------------------------===//

def Dataflow_InvariantOp : Dataflow_Op<"invariant", []> {
  let summary = "Loop-invariant token repeater";
  let description = [{
    A loop-invariant value repeater. It consumes a (invariant value) and
    d (control) and produces o. It behaves like dataflow.carry but reuses
    a instead of consuming a separate b stream.

    Semantics:
    - Initial stage: consume a, emit a on o, and move to block stage.
      d is not consumed in this step.
    - Block stage: for each d, if d is true, emit the stored a on o.
      If d is false, consume d and return to initial stage.

    The length of o matches the length of d for each iteration burst.
    Example (T=true, F=false):
      a: [A, B]
      d: [T, T, F, T, T, T, T, F]
      o: [A, A, A, B, B, B, B, B]
  }];
  let arguments = (ins I1:$d, AnyType:$a);
  let results = (outs AnyType:$o);

  let assemblyFormat = "$d `,` $a attr-dict `:` type($d) `,` type($a) `->` type($o)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// dataflow.stream
//===----------------------------------------------------------------------===//

def Dataflow_StreamOp : Dataflow_Op<"stream", []> {
  let summary = "Configurable index stream generator";
  let description = [{
    Generates a raw index stream and a raw will_continue stream for a
    loop-like control pattern. The streams are length N+1 for a loop with
    N body iterations. This supports loop-carried values that produce N+1
    outputs.

    Attributes:
    - step_op: "+=", "-=", "*=", "/=", "<<=", ">>="
      Defines how to update the index each iteration. This is a hardware
      parameter and cannot change at runtime.
    - stop_cond: "<", "<=", ">", ">=", "!="
      Defines the continue condition for the loop. This is a runtime
      configuration parameter.

    Semantics:
    - raw_index emits: start, next, ..., last, and one extra value.
    - raw_will_continue emits: T for each body iteration, then F to mark
      loop exit.

    Where:
    - continue when (idx stop_cond bound) is true.
    - next = idx (step_op) step.

    Example: step_op="+=", stop_cond="<", start=0, step=1, bound=5
      raw_index: [0, 1, 2, 3, 4, 5]
      raw_will_continue: [T, T, T, T, T, F]

    Zero-trip example: step_op="+=", stop_cond="<", start=3, step=1, bound=3
      raw_index: [3]
      raw_will_continue: [F]
  }];
  let arguments = (ins Index:$start, Index:$step, Index:$bound,
                   OptionalAttr<StrAttr>:$step_op,
                   OptionalAttr<StrAttr>:$stop_cond);
  let results = (outs Index:$index, I1:$willContinue);

  let assemblyFormat = "$start `,` $step `,` $bound attr-dict";

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "::mlir::Value":$start, "::mlir::Value":$step, "::mlir::Value":$bound), [{
      ::mlir::Type indexType = $_builder.getIndexType();
      ::mlir::Type i1Type = $_builder.getI1Type();
      $_state.addOperands({start, step, bound});
      $_state.addTypes({indexType, i1Type});
    }]>
  ];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// dataflow.gate
//===----------------------------------------------------------------------===//

def Dataflow_GateOp : Dataflow_Op<"gate", []> {
  let summary = "Before-to-after adapter for loop streams";
  let description = [{
    A stream adapter that shifts two streams to align "before" and "after"
    regions of a loop. It consumes before_value and before_cond and produces
    after_value and after_cond by cutting the tail of value and the head of
    cond (one-element shift).

    Semantics:
      after_value[i] = before_value[i]
      after_cond[i]  = before_cond[i+1]

    This removes one element from each stream. If before_cond has length 1,
    the outputs are empty. The operation is the "cut head / cut tail" stage
    used to align scf.before (N+1) to scf.after (N).

    Important timing detail:
    - The head of before_cond is dropped (cut head).
    - The tail of before_value is dropped (cut tail).
    - The resulting streams are time-shifted: after_cond[0] corresponds to
      after_value[1], not after_value[0]. In other words, after_cond indicates
      whether the *next* after-region execution is the last one.
    This reflects the fact that "is this the last after iteration" is only
    known after the next before-region condition is computed.

    Example:
      before_value: [A, B, C, D, E]
      before_cond:  [T, T, T, T, F]
      after_value:  [A, B, C, D]
      after_cond:   [T, T, T, F]

    Example (length 1):
      before_value: [A]
      before_cond:  [F]
      after_value:  []
      after_cond:   []
  }];
  let arguments = (ins AnyType:$beforeValue, I1:$beforeCond);
  let results = (outs AnyType:$afterValue, I1:$afterCond);

  let assemblyFormat = "$beforeValue `,` $beforeCond attr-dict `:` type($beforeValue) `,` type($beforeCond) `->` type($afterValue) `,` type($afterCond)";
  let hasVerifier = 1;
}

#endif // LOOM_DIALECT_DATAFLOW_DATAFLOWOPS_TD
