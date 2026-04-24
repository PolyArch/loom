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

def Dataflow_CarryOp : Dataflow_Op<"carry", [Pure,
    AllTypesMatch<["init", "carry", "output"]>]> {
  let summary =
      "two-state carry: emit init once, then gate subsequent carries by cond";
  let description = [{
    Two-state (init / carry) token-level element.

      * Start state is `init`.
      * In state `init`: wait for one `%init` token; forward it to
        `%output`; transition to `carry`.
      * In state `carry`: wait for one `%cond : i1` and one `%carry`
        token simultaneously; consume both.
          - if `cond` is true : forward `%carry` to `%output`, stay in
            `carry`;
          - if `cond` is false: do not emit, return to state `init`.

    `%init`, `%carry` and `%output` share a single type; `%cond` is
    `i1`.
  }];

  let arguments = (ins I1:$cond, AnyType:$init, AnyType:$carry);
  let results = (outs AnyType:$output);

  let assemblyFormat = [{
    $cond `,` $init `,` $carry attr-dict `:` type($output)
  }];
}

def Dataflow_InvariantOp : Dataflow_Op<"invariant", [Pure,
    AllTypesMatch<["init", "output"]>]> {
  let summary =
      "latch an init value and replay it once per true cond until reset";
  let description = [{
    Two-state element that latches an init value and replays it.

      * Start state is `init`.
      * In state `init`: wait for one `%init` token, record its value,
        forward it to `%output`, and transition to `carry`.
      * In state `carry`: wait for one `%cond : i1` token. `%init` is
        not consumed in this state.
          - if `cond` is true : re-emit the recorded value to
            `%output`, stay in `carry`;
          - if `cond` is false: clear the recorded value, emit nothing,
            return to state `init`.

    `%init` and `%output` share a single type; `%cond` is `i1`.
  }];

  let arguments = (ins I1:$cond, AnyType:$init);
  let results = (outs AnyType:$output);

  let assemblyFormat = [{
    $cond `,` $init attr-dict `:` type($output)
  }];
}

def Dataflow_GateOp : Dataflow_Op<"gate", [Pure,
    AllTypesMatch<["before_value", "after_value"]>]> {
  let summary =
      "open a value channel on the first true, reclose on the next false";
  let description = [{
    Two-state gate over a (cond, value) pair. Both inputs are always
    consumed together on each firing.

      * Start state is `init`.
      * In state `init`:
          - on `(false, X)`: emit nothing on either output;
          - on `(true,  X)`: emit only `X` on `%after_value` (no token
            on `%after_cond`), transition to `continue`.
      * In state `continue`:
          - on `(true,  X)`: forward the pair as `(true, X)` on
            `%after_cond` and `%after_value` respectively, stay in
            `continue`;
          - on `(false, X)`: emit only `false` on `%after_cond` (no
            token on `%after_value`), return to state `init`.

    `%before_value` and `%after_value` share a single type;
    `%before_cond` and `%after_cond` are `i1`.
  }];

  let arguments = (ins I1:$before_cond, AnyType:$before_value);
  let results = (outs I1:$after_cond, AnyType:$after_value);

  let assemblyFormat = [{
    $before_cond `,` $before_value attr-dict `:` type($after_value)
  }];
}

def Dataflow_ConstantOp : Dataflow_Op<"constant", [Pure]> {
  let summary = "emit a recorded constant on each incoming ctrl token";
  let description = [{
    One-shot trigger: on every `%ctrl : none` token that arrives, the
    op fires once and emits the value stored in the `const_value`
    attribute on `%const`.

    `const_value` must be a typed attribute whose type equals the
    result type of `%const`.
  }];

  let arguments = (ins NoneType:$ctrl, AnyAttr:$const_value);
  let results = (outs AnyType:$value);

  let assemblyFormat = "$ctrl attr-dict `:` type($value)";
  let hasVerifier = 1;
}

def Dataflow_SyncOp : Dataflow_Op<"sync", [Pure]> {
  let summary = "wait for all inputs, then forward them all as outputs";
  let description = [{
    Variadic rendezvous. Once every input has a token, all tokens are
    consumed and the corresponding outputs fire simultaneously.

    Operand count equals result count; types match positionally.
  }];

  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    $inputs attr-dict `:` functional-type($inputs, $outputs)
  }];
  let hasVerifier = 1;
}

#endif // DATAFLOW_OPS_TD
