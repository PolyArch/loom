#ifndef DATAFLOW_OPS_TD
#define DATAFLOW_OPS_TD

include "Dataflow/IR/DataflowDialect.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Dataflow_StreamOp : Dataflow_Op<"stream", [Pure,
    AllTypesMatch<["lb", "ub", "step", "index"]>]> {
  let summary = "produce an index stream with a trailing out-of-range sentinel";
  let description = [{
    Emits a stream of `(index, rwc)` pairs driven by three scalar integer
    operands and two predicate attributes:

      * `step_op`  : how `index` is advanced between emissions; one of
                     `+=`, `*=`, `-=`, `/=`, `<<=`, `>>=`.
      * `cont_cond`: continue predicate; when true of the *current* index
                     and `ub`, the pair is emitted with `rwc = true` and
                     the next index is computed. When false, one final
                     pair with `rwc = false` is emitted. One of
                     `<`, `<=`, `>`, `>=`, `!=`.

    `lb`, `ub`, `step` and `index` share a single signless integer type;
    `rwc` is always `i1`.

    Example: `lb=0, ub=5, step=1, step_op="+=", cont_cond="<"` yields
             `(0,T), (1,T), (2,T), (3,T), (4,T), (5,F)`.
  }];

  let arguments = (ins
      SignlessIntegerLike:$lb,
      SignlessIntegerLike:$ub,
      SignlessIntegerLike:$step,
      StrAttr:$step_op,
      StrAttr:$cont_cond
  );
  let results = (outs SignlessIntegerLike:$index, I1:$rwc);

  let assemblyFormat = [{
    $lb `,` $ub `,` $step attr-dict `:` type($index)
  }];
  let hasVerifier = 1;
}

#endif // DATAFLOW_OPS_TD
