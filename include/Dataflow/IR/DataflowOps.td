#ifndef DATAFLOW_OPS_TD
#define DATAFLOW_OPS_TD

include "Dataflow/IR/DataflowDialect.td"
include "Dataflow/IR/DataflowEnums.td"
include "Dataflow/IR/DataflowInterfaces.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/RegionKindInterface.td"

def Dataflow_GraphType : AnyTypeOf<[
    AnyInteger, AnyFloat, Index, NoneType, AnyMemRef]>;

//===----------------------------------------------------------------------===//
// Thread Channel Endpoint Ops
//===----------------------------------------------------------------------===//

def Dataflow_ChannelSendOp : Dataflow_Op<"channel.send", [
    TypesMatchWith<"'message' type matches channel element type",
                   "channel", "message",
                   "::llvm::cast<::dataflow::ChannelType>($_self).getElementType()">
]> {
  let summary = "send one message through a thread-level channel";
  let description = [{
    Blocks as required to submit one message to the ordered channel. This
    operation is valid only transitively inside a `dataflow.thread` body.
  }];

  let arguments = (ins Dataflow_ChannelType:$channel, AnyType:$message);
  let assemblyFormat =
      "$channel `,` $message attr-dict `:` qualified(type($channel))";
  let hasVerifier = 1;
}

def Dataflow_ChannelReceiveOp : Dataflow_Op<"channel.receive", [
    TypesMatchWith<"'message' type matches channel element type",
                   "channel", "message",
                   "::llvm::cast<::dataflow::ChannelType>($_self).getElementType()">
]> {
  let summary = "receive one message from a thread-level channel";
  let description = [{
    Blocks until the next ordered message is available and returns it. This
    operation is valid only transitively inside a `dataflow.thread` body.
  }];

  let arguments = (ins Dataflow_ChannelType:$channel);
  let results = (outs AnyType:$message);
  let assemblyFormat = "$channel attr-dict `:` qualified(type($channel))";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Streaming Ops
//
// Ops whose primary semantics is generating / shaping / latching a stream of
// value tokens. They are typically driven by i1 phase tokens or by integer
// recurrence operands that produce IV streams.
//===----------------------------------------------------------------------===//

def Dataflow_StreamOp : Dataflow_Op<"stream", [Pure,
    CanonicalDataflowActor,
    AllTypesMatch<["init", "limit", "step", "iv"]>]> {
  let summary = "produce valid IV tokens and an explicit closing phase token";
  let description = [{
    Executes one predicate-terminated integer recurrence per activation.
    `step_kind` selects the update applied after each true decision, and
    `predicate` compares the current value against `limit`.

      * `step_kind`: the canonical integer update operation: `add`, `sub`,
                     `mul`, `sdiv`, `udiv`, `shl`, `ashr`, or `lshr`.
      * `predicate`: an upstream `arith.cmpi` predicate applied to the current
                     value and `limit`.

    Idle consumes one `init` / `limit` / `step` triple. A true decision emits
    the current value on `iv`, emits `true` on `phase`, advances the current
    value, and remains active. A false decision emits only `false` on `phase`
    and returns to Idle. No sentinel IV is emitted, and subsequent activation
    triples may reuse the same operation.

    `init`, `limit`, `step`, and `iv` share one scalar signless integer type;
    `phase` is always `i1`. For `init=0`, `limit=5`, `step=1`, `step_kind=add`,
    and `predicate=slt`, `iv` is `0,1,2,3,4` and `phase` is `T,T,T,T,T,F`.
  }];

  let arguments = (ins
      AnySignlessInteger:$init,
      AnySignlessInteger:$limit,
      AnySignlessInteger:$step,
      Dataflow_StreamStepKindAttr:$step_kind,
      Arith_CmpIPredicateAttr:$predicate
  );
  let results = (outs AnySignlessInteger:$iv, I1:$phase);

  let hasCustomAssemblyFormat = 1;
}

def Dataflow_CarryOp : Dataflow_Op<"carry", [Pure,
    CanonicalDataflowActor,
    AllTypesMatch<["init", "carry", "output"]>]> {
  let summary =
      "two-state carry: emit init once, then gate subsequent carries by cond";
  let description = [{
    Two-state (init / carry) token-level element.

      * Start state is `init`.
      * In state `init`: wait for one `%init` token; forward it to
        `%output`; transition to `carry`.
      * In state `carry`: inspect the next `%cond : i1` token.
          - if `cond` is true : wait for and consume `%carry`, consume the
            condition, forward `%carry` to `%output`, and stay in `carry`;
          - if `cond` is false: consume only the condition, emit nothing, and
            return to state `init`.

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
    CanonicalDataflowActor,
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
    CanonicalDataflowActor,
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

def Dataflow_ParallelizeOp : Dataflow_Op<"parallelize", [
    Pure, CanonicalDataflowActor]> {
  let summary = "group scalar tokens into fixed-width lane streams plus mask";
  let description = [{
    Groups a scalar integer stream and a continuation stream into
    `vec_size` parallel lane streams and one `iN` mask stream. True
    continuation tokens fill pending lanes; a false continuation token
    flushes any pending partial group and resets the lane pointer.

    The optional stride operand advances the lane pointer by a dynamic
    amount. If omitted, stride is one.
  }];

  let arguments = (ins
      AnySignlessInteger:$data,
      I1:$cont,
      Optional<AnySignlessInteger>:$stride,
      I64Attr:$vec_size);
  let results = (outs
      Variadic<AnySignlessInteger>:$outputs,
      AnySignlessInteger:$mask);

  let assemblyFormat = [{
    $data `,` $cont (`,` $stride^)? attr-dict `:`
    functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def Dataflow_PackOp : Dataflow_Op<"pack", [Pure, CanonicalDataflowActor]> {
  let summary = "pack masked lane streams into one wide integer stream";
  let description = [{
    Waits for one mask token, consumes only the lane tokens whose mask
    bits are set, and emits a wide signless integer whose least
    significant slot is lane zero. Inactive slots are zero-filled.
  }];

  let arguments = (ins
      Variadic<AnySignlessInteger>:$inputs,
      AnySignlessInteger:$mask,
      I64Attr:$vec_size);
  let results = (outs AnySignlessInteger:$packed);

  let assemblyFormat = [{
    $inputs `mask` $mask attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def Dataflow_UnpackOp : Dataflow_Op<"unpack", [Pure, CanonicalDataflowActor]> {
  let summary = "unpack a wide integer stream into masked lane streams";
  let description = [{
    Consumes one packed integer token and one mask token. For each set
    mask bit, emits the corresponding lane value; unset lanes emit no
    token. Lane zero is the least significant packed slot.
  }];

  let arguments = (ins
      AnySignlessInteger:$packed,
      AnySignlessInteger:$mask,
      I64Attr:$vec_size);
  let results = (outs Variadic<AnySignlessInteger>:$outputs);

  let assemblyFormat = [{
    $packed `,` $mask attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def Dataflow_SerializeOp : Dataflow_Op<"serialize", [
    Pure, CanonicalDataflowActor]> {
  let summary = "serialize masked lane streams into scalar data and cont";
  let description = [{
    Waits for one mask token, consumes only active lane tokens in lane
    order, emits each active lane on the scalar data stream with a true
    continuation token, and emits one trailing false continuation token.
  }];

  let arguments = (ins
      Variadic<AnySignlessInteger>:$inputs,
      AnySignlessInteger:$mask,
      I64Attr:$vec_size);
  let results = (outs AnySignlessInteger:$data, I1:$cont);

  let assemblyFormat = [{
    $inputs `mask` $mask attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Control Ops
//
// Ops whose primary semantics is controlling *which* tokens flow and *when*:
// constant sources triggered by ctrl tokens, rendezvous (sync), and
// selection routing (mux / demux).
//===----------------------------------------------------------------------===//

def Dataflow_ConstantOp : Dataflow_Op<"constant", [
    Pure, CanonicalDataflowActor]> {
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

def Dataflow_SyncOp : Dataflow_Op<"sync", [Pure, CanonicalDataflowActor]> {
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

def Dataflow_MuxOp : Dataflow_Op<"mux", [Pure, CanonicalDataflowActor]> {
  let summary = "N-to-1 selection: forward the sel-picked input to output";
  let description = [{
    N-input, 1-output multiplexer (N > 1). `sel` picks one input port
    index `k`. The op fires when tokens are available on *both* `%sel`
    and `%inputs[k]`. Both are consumed and the value is forwarded to
    `%output`. Tokens on the non-selected inputs are **not** consumed
    (they remain buffered; the other lanes block).

    `sel` type:
      * exactly 2 inputs -> `i1`
      * more than 2 inputs -> `index`

    All inputs and the output share one type.
  }];

  let arguments = (ins AnyTypeOf<[I1, Index]>:$sel,
                       Variadic<AnyType>:$inputs);
  let results = (outs AnyType:$output);

  let assemblyFormat = [{
    $sel `,` $inputs attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def Dataflow_DemuxOp : Dataflow_Op<"demux", [Pure, CanonicalDataflowActor]> {
  let summary = "1-to-N selection: route the input to the sel-picked output";
  let description = [{
    1-input, N-output demultiplexer (N > 1). `sel` picks one output
    port index `k`. The op fires when tokens are available on both
    `%sel` and `%input`; both are consumed and the value is forwarded
    to `%outputs[k]`. No token is produced on the non-selected output
    ports.

    `sel` type:
      * exactly 2 outputs -> `i1`
      * more than 2 outputs -> `index`

    The input and all outputs share one type.
  }];

  let arguments = (ins AnyTypeOf<[I1, Index]>:$sel, AnyType:$input);
  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    $sel `,` $input attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Memory Ops
//
// Streaming load / store against a memref, orchestrated by none-typed ctrl /
// done tokens. The memref's element type constrains the data port type.
//===----------------------------------------------------------------------===//

def Dataflow_LoadOp : Dataflow_Op<"load", [
    CanonicalDataflowActor,
    TypesMatchWith<"'data' type matches memref element type",
                   "mem", "data",
                   "::llvm::cast<::mlir::MemRefType>($_self).getElementType()">
]> {
  let summary = "streaming load: addr + ctrl -> data + done";
  let description = [{
    On the simultaneous arrival of an `%addr : index` token and a
    `%ctrl : none` token, consumes both, loads
    `%mem[%addr]`, emits the read value on `%data`, and emits a
    `none` token on `%done`.

    The memref element type constrains `%data`'s type.
  }];

  let arguments = (ins AnyMemRef:$mem, Index:$addr, NoneType:$ctrl);
  let results = (outs AnyType:$data, NoneType:$done);

  let assemblyFormat = [{
    $mem `[` $addr `]` $ctrl attr-dict `:` type($mem)
  }];
}

def Dataflow_StoreOp : Dataflow_Op<"store", [
    CanonicalDataflowActor,
    TypesMatchWith<"'data' type matches memref element type",
                   "mem", "data",
                   "::llvm::cast<::mlir::MemRefType>($_self).getElementType()">
]> {
  let summary = "streaming store: addr + data + ctrl -> done";
  let description = [{
    On the simultaneous arrival of an `%addr : index`, a `%data` value
    (of the memref element type), and a `%ctrl : none`, consumes all
    three, writes `%data` into `%mem[%addr]`, and emits a `none` token
    on `%done`.
  }];

  let arguments = (ins AnyMemRef:$mem, Index:$addr, AnyType:$data,
                       NoneType:$ctrl);
  let results = (outs NoneType:$done);

  let assemblyFormat = [{
    $mem `[` $addr `]` $data $ctrl attr-dict `:` type($mem)
  }];
}

//===----------------------------------------------------------------------===//
// Symbol-bearing function-like ops
//
// The canonical module-scope definition and launch surfaces.
//===----------------------------------------------------------------------===//

def Dataflow_ThreadOp : Dataflow_Op<"thread", [
    AutomaticAllocationScope,
    IsolatedFromAbove,
    HasParent<"::mlir::ModuleOp">,
    SingleBlockImplicitTerminator<"ThreadYieldOp">,
    FunctionOpInterface,
    RecursiveMemoryEffects
]> {
  let summary = "Symbol-bearing function-like AccCore kernel definition";
  let description = [{
    Module-scope, function-like callable that holds an AccCore kernel
    body. It does not itself execute; one or more
    `dataflow.thread.launch` ops materialize launches of it.

    The body's entry block has the layout
    `(args_*, thread_ctrl: none, iv_*: index)` (per spec section
    5.4.1). The first N block args mirror `function_type.inputs`; the
    trailing `thread_ctrl` and grid index args are NOT in
    `function_type` (they are launch-instance extras). Specifically:

      * Args[0 .. N-1]   match `function_type.inputs` position-wise.
      * Args[N]          is `none` -- the per-launch `thread_ctrl`
                         slot, used as the AccCore start signal and
                         consumed by root `dataflow.graph.launch` ops
                         in the body as a dependency event.
      * Args[N+1 .. end] are all `index` -- one per grid dim.

    The custom assembly format prints these trailing extras after the
    function-style signature using a separate `ctrl ( ... )` clause
    (the `thread_ctrl` slot) and an `iv ( ... )` clause (the
    grid-index slots). Either / both clauses are optional; threads
    written without them are accepted at parse time only when the op
    is external (i.e., body is empty), since a body-having thread
    must carry the trailing `thread_ctrl` slot per the verifier.

    The op is `IsolatedFromAbove`; values flow in only through the
    matching `dataflow.thread.launch` body operands.

    The `staticGrid*` and `mapping` attributes are accepted as optional
    dictionary attributes. They do not currently change execution semantics.
  }];

  let arguments = (ins
      SymbolNameAttr:$sym_name,
      TypeAttrOf<FunctionType>:$function_type,
      OptionalAttr<StrAttr>:$sym_visibility,
      OptionalAttr<DictArrayAttr>:$arg_attrs,
      OptionalAttr<DictArrayAttr>:$res_attrs);

  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins
        "::llvm::StringRef":$name,
        "::mlir::FunctionType":$type,
        CArg<"::llvm::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs)>
  ];

  let extraClassDeclaration = [{
    /// FunctionOpInterface methods.
    ::llvm::ArrayRef<::mlir::Type> getArgumentTypes() {
      return getFunctionType().getInputs();
    }
    ::llvm::ArrayRef<::mlir::Type> getResultTypes() {
      return getFunctionType().getResults();
    }
    ::mlir::Region *getCallableRegion() {
      return isExternal() ? nullptr : &getBody();
    }
    bool isExternal() { return getBody().empty(); }

    /// Override the default FunctionOpInterface body verifier: a
    /// dataflow.thread body's entry block leads with the
    /// function-signature args, then a `none`-typed `thread_ctrl`
    /// slot, then zero or more `index`-typed `iv_*` slots (per spec
    /// section 5.4.1: `(args_*, thread_ctrl, iv_*)`).
    ::llvm::LogicalResult verifyBody() {
      if (isExternal())
        return ::mlir::success();
      ::mlir::Block &entry = getBody().front();
      ::llvm::ArrayRef<::mlir::Type> inputs = getFunctionType().getInputs();
      const size_t N = inputs.size();
      // Body-carrying threads MUST have the trailing thread_ctrl slot.
      if (entry.getNumArguments() < N + 1)
        return emitOpError("entry block must have at least ")
               << (N + 1)
               << " arguments (function inputs + 1 thread_ctrl slot)";
      // First N entry block arguments must match function_type.inputs.
      for (size_t i = 0; i < N; ++i) {
        if (entry.getArgument(i).getType() != inputs[i])
          return emitOpError("type of entry block argument #")
                 << i << '(' << entry.getArgument(i).getType()
                 << ") must match the corresponding function signature input ("
                 << inputs[i] << ')';
      }
      // Slot N must be the `none`-typed thread_ctrl per spec
      // section 5.4.1.
      if (!::llvm::isa<::mlir::NoneType>(entry.getArgument(N).getType()))
        return emitOpError("entry block argument #")
               << N << " (thread_ctrl) must have type `none`, got "
               << entry.getArgument(N).getType();
      // All remaining slots are grid induction variables of `index`.
      for (size_t i = N + 1, e = entry.getNumArguments(); i < e; ++i) {
        if (!::llvm::isa<::mlir::IndexType>(entry.getArgument(i).getType()))
          return emitOpError("entry block argument #")
                 << i << " (grid iv) must have type `index`, got "
                 << entry.getArgument(i).getType();
      }
      return ::mlir::success();
    }
  }];
}

def Dataflow_ThreadYieldOp : Dataflow_Op<"thread.yield", [
    Terminator,
    ParentOneOf<["::dataflow::ThreadOp"]>,
    Pure
]> {
  let summary = "Terminator for a dataflow.thread body";
  let description = [{
    Accepts an unordered all-of completion frontier of `none` values.
    Tensor-result aggregation from `scf.forall` is materialised into
    explicit destination-buffer writes before thread promotion, so the
    thread definition has no parallel combining region or thread data
    results.
  }];

  let arguments = (ins Variadic<NoneType>:$completionFrontier);

  let assemblyFormat = "($completionFrontier^ `:` type($completionFrontier))? attr-dict";

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins CArg<"::mlir::ValueRange", "{}">:$completionFrontier), [{
      $_state.addOperands(completionFrontier);
    }]>
  ];
}

def Dataflow_ThreadLaunchOp : Dataflow_Op<"thread.launch", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Async launch of a dataflow.thread callable";
  let description = [{
    References a `dataflow.thread` definition by symbol and supplies
    body operands and optional grid upper bounds. Always produces one
    `!dataflow.thread_token` completion handle for all dynamic launch
    instances.

    Grid lower bounds and steps are not modeled; body operands, upper bounds,
    and async dependencies are explicit.
  }];

  let arguments = (ins
      FlatSymbolRefAttr:$callee,
      Variadic<AnyType>:$bodyOperands,
      Variadic<Index>:$gridUpperBounds,
      Variadic<Dataflow_ThreadTokenType>:$asyncDependencies);

  let results = (outs Dataflow_ThreadTokenType:$asyncToken);

  let assemblyFormat = [{
    $callee
    `(` $bodyOperands `)`
    ( `grid` `(` $gridUpperBounds^ `)` )?
    ( `wait` `(` $asyncDependencies^ `)` )?
    attr-dict `:`
    functional-type($bodyOperands, results)
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
        "::mlir::FlatSymbolRefAttr":$callee,
        "::mlir::ValueRange":$bodyOperands,
        "::mlir::ValueRange":$gridUpperBounds,
        "::mlir::ValueRange":$asyncDependencies), [{
      $_state.addOperands(bodyOperands);
      $_state.addOperands(gridUpperBounds);
      $_state.addOperands(asyncDependencies);
      auto &properties = $_state.getOrAddProperties<Properties>();
      properties.operandSegmentSizes = {
          static_cast<int32_t>(bodyOperands.size()),
          static_cast<int32_t>(gridUpperBounds.size()),
          static_cast<int32_t>(asyncDependencies.size())};
      properties.callee = callee;
      $_state.addTypes(::dataflow::ThreadTokenType::get($_builder.getContext()));
    }]>
  ];

  let hasVerifier = 1;
}

def Dataflow_ThreadWaitOp : Dataflow_Op<"thread.wait", [
    NativeOpTrait<"AtLeastNOperands<1>::Impl">
]> {
  let summary = "Ordered wait for one or more thread completion tokens";
  let description = [{
    Consumes an unordered all-of set of `!dataflow.thread_token` values.
    It is an ordered stored-program wait that does not introduce
    memory-barrier or graph-control semantics.
  }];

  let arguments = (ins Variadic<Dataflow_ThreadTokenType>:$asyncDependencies);

  let assemblyFormat = "$asyncDependencies `:` type($asyncDependencies) attr-dict";

  let hasVerifier = 1;
}

def Dataflow_GraphOp : Dataflow_Op<"graph", [
    IsolatedFromAbove,
    HasParent<"::mlir::ModuleOp">,
    SingleBlockImplicitTerminator<"GraphReturnOp">,
    FunctionOpInterface,
    RecursiveMemoryEffects,
    DeclareOpInterfaceMethods<RegionKindInterface>
]> {
  let summary = "Symbol-bearing function-like SpatialCore graph definition";
  let description = [{
    Module-scope, function-like callable holding the SpatialCore body
    of a leaf dataflow graph. It does not itself execute; one or more
    `dataflow.graph.launch` ops materialise launches of it inside the
    body of a `dataflow.thread` definition.

    `function_type` contains only application payload ports. Normalized
    `input_segments` and `result_segments` classify those payloads as value,
    stream, and memory ports. The body's distinguished leading `none` block
    argument is the invocation start protocol endpoint, while launch `done`
    is derived exclusively from `dataflow.graph.return.complete`; neither is
    stored in the function type.

    This is the only canonical graph definition surface.
  }];

  let arguments = (ins
      SymbolNameAttr:$sym_name,
      TypeAttrOf<FunctionType>:$function_type,
      DenseI32ArrayAttr:$input_segments,
      DenseI32ArrayAttr:$result_segments,
      OptionalAttr<StrAttr>:$sym_visibility,
      OptionalAttr<DictArrayAttr>:$arg_attrs,
      OptionalAttr<DictArrayAttr>:$res_attrs);

  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins
        "::llvm::StringRef":$name,
        "::mlir::FunctionType":$type,
        CArg<"::llvm::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs)>
  ];

  let extraClassDeclaration = [{
    /// FunctionOpInterface methods.
    ::llvm::ArrayRef<::mlir::Type> getArgumentTypes() {
      return getFunctionType().getInputs();
    }
    ::llvm::ArrayRef<::mlir::Type> getResultTypes() {
      return getFunctionType().getResults();
    }
    ::mlir::Region *getCallableRegion() {
      return isExternal() ? nullptr : &getBody();
    }
    bool isExternal() { return getBody().empty(); }
    ::mlir::BlockArgument getStart();
    ::llvm::ArrayRef<int32_t> getInputSegmentSizes();
    ::llvm::ArrayRef<int32_t> getResultSegmentSizes();
    GraphPortKind getInputPortKind(unsigned index);
    GraphPortKind getResultPortKind(unsigned index);
    ::llvm::LogicalResult verifyBody() {
      if (isExternal())
        return ::mlir::success();
      ::mlir::Block &entry = getBody().front();
      ::llvm::ArrayRef<::mlir::Type> inputs = getFunctionType().getInputs();
      if (entry.getNumArguments() != inputs.size() + 1)
        return emitOpError("entry block must have one start argument plus ")
               << inputs.size() << " application inputs";
      if (!::llvm::isa<::mlir::NoneType>(entry.getArgument(0).getType()))
        return emitOpError("entry block argument #0 must be start type none");
      for (size_t i = 0, e = inputs.size(); i < e; ++i) {
        if (entry.getArgument(i + 1).getType() != inputs[i])
          return emitOpError("entry block argument #")
                 << (i + 1) << " type "
                 << entry.getArgument(i + 1).getType()
                 << " must match function input type " << inputs[i];
      }
      return ::mlir::success();
    }
  }];
}

def Dataflow_GraphReturnOp : Dataflow_Op<"graph.return", [
    AttrSizedOperandSegments,
    Terminator,
    ParentOneOf<["::dataflow::GraphOp"]>,
    Pure
]> {
  let summary = "Terminator for a dataflow.graph body";
  let description = [{
    Structurally declares the enclosing graph's value, stream, and memory
    outputs together with its mandatory retirement frontier. `complete` is
    an unordered all-of set of one or more `none` values; the launch `done`
    event is derived from that set and is not itself a return operand.

    The compact assembly form `%complete, %values... : none, types...` is
    retained for the common case with one completion witness and no stream
    or memory outputs. Other shapes print all four named segments.
  }];

  let arguments = (ins
      Variadic<AnyType>:$values,
      Variadic<AnyType>:$streams,
      Variadic<AnyType>:$memories,
      Variadic<NoneType>:$complete);

  let hasCustomAssemblyFormat = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
        "::mlir::ValueRange":$values,
        "::mlir::ValueRange":$streams,
        "::mlir::ValueRange":$memories,
        "::mlir::ValueRange":$complete), [{
      $_state.addOperands(values);
      $_state.addOperands(streams);
      $_state.addOperands(memories);
      $_state.addOperands(complete);
      auto &properties = $_state.getOrAddProperties<Properties>();
      properties.operandSegmentSizes = {
          static_cast<int32_t>(values.size()),
          static_cast<int32_t>(streams.size()),
          static_cast<int32_t>(memories.size()),
          static_cast<int32_t>(complete.size())};
    }]>
  ];

  let hasVerifier = 1;
}

def Dataflow_GraphLaunchOp : Dataflow_Op<"graph.launch", [
    AttrSizedOperandSegments,
    AttrSizedResultSegments,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Asynchronous launch of a dataflow.graph callable";
  let description = [{
    References a `dataflow.graph` definition by symbol from inside a
    `dataflow.thread` body. Dependencies, value inputs, stream input channel
    bindings, memory imports, and stream output channel bindings are explicit
    operand segments. Value outputs and memory exports are SSA results; the
    trailing `done` result is the graph retirement event.

    The operation resolves its callee through `SymbolUserOpInterface` but does
    not project callee effects through `MemoryEffectsOpInterface`.
  }];

  let arguments = (ins
      FlatSymbolRefAttr:$callee,
      Variadic<NoneType>:$dependencies,
      Variadic<AnyType>:$valueInputs,
      Variadic<Dataflow_ChannelType>:$streamInputs,
      Variadic<AnyType>:$memoryInputs,
      Variadic<Dataflow_ChannelType>:$streamOutputs);

  let results = (outs
      Variadic<AnyType>:$valueResults,
      Variadic<AnyType>:$memoryResults,
      NoneType:$done);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif // DATAFLOW_OPS_TD
