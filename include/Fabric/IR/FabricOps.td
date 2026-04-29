#ifndef FABRIC_OPS_TD
#define FABRIC_OPS_TD

include "Fabric/IR/FabricDialect.td"
include "Fabric/IR/FabricTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Fabric_MuxOp : Fabric_Op<"mux", [Pure, SameOperandsAndResultType]> {
  let summary = "fabric multiplexer with handshake";
  let description = [{
    Selects one of N (N > 1) inputs onto the single output. All inputs and the
    output share one `!fabric.bits<W>` type. No hardware parameters (the port
    width is fully described by the shared operand/result type).

    Software parameters (either all set or all unset):
      * `sel`        : i32, 0 <= sel < N (forced 0 when `disconnect` is true)
      * `discard`    : bool
      * `disconnect` : bool

    Unset means the op describes pure hardware (not programmed). Programmed
    combinations must satisfy: !(discard && disconnect).
  }];

  let arguments = (ins
      Variadic<Fabric_BitsType>:$inputs,
      OptionalAttr<I32Attr>:$sel,
      OptionalAttr<BoolAttr>:$discard,
      OptionalAttr<BoolAttr>:$disconnect
  );
  let results = (outs Fabric_BitsType:$output);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Fabric_DemuxOp : Fabric_Op<"demux", [Pure, SameOperandsAndResultType]> {
  let summary = "fabric demultiplexer with handshake";
  let description = [{
    Single input, N outputs (N > 1). All input/output ports share one
    `!fabric.bits<W>` type. No hardware parameters (the port width is fully
    described by the shared operand/result type). The op is the input/output
    dual of fabric.mux.

    Software parameters (either all set or all unset):
      * `sel`        : i32, 0 <= sel < N (forced 0 when `disconnect` is true)
      * `discard`    : bool
      * `disconnect` : bool

    Unset means the op describes pure hardware (not programmed). When
    programmed, `discard && disconnect` is forbidden. Untying valid/ready
    on the unselected outputs:

      * pure hardware: input.ready=0, all outputs.valid=0 (= disconnect)
      * sel only    : input <-> outputs[sel] passthrough; other outputs.valid=0
      * discard=T   : input.ready=1 (input drained); all outputs.valid=0
      * disconnect=T: input.ready=0; all outputs.valid=0
  }];

  let arguments = (ins
      Fabric_BitsType:$input,
      OptionalAttr<I32Attr>:$sel,
      OptionalAttr<BoolAttr>:$discard,
      OptionalAttr<BoolAttr>:$disconnect
  );
  let results = (outs Variadic<Fabric_BitsType>:$outputs);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Fabric_FifoOp : Fabric_Op<"fifo", [Pure]> {
  let summary = "fabric FIFO with handshake";
  let description = [{
    Single-input single-output FIFO. Input and output share one fabric type
    (same kind, same widths).

    Hardware parameters (always present, printed in `[...]`):
      * `max_depth`   : i32, must be > 0 (physical FIFO depth)
      * `bypassable`  : bool (whether hardware supports combinational bypass)

    Software parameter (only valid when `bypassable` is true, printed in `{}`):
      * `bypassed`    : bool (when true, the FIFO is configured in bypass mode;
                       data/valid/ready become a combinational passthrough)

    Unset `bypassed` describes pure hardware (not programmed). Setting
    `bypassed` when `bypassable` is false is illegal.
  }];

  let arguments = (ins
      Fabric_AnyType:$input,
      I32Attr:$max_depth,
      BoolAttr:$bypassable,
      OptionalAttr<BoolAttr>:$bypassed
  );
  let results = (outs Fabric_AnyType:$output);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Fabric_OpOp : Fabric_Op<"op", []> {
  let summary =
      "fabric module instance: a configurable hardware area that can implement "
      "one of a set of software ops";
  let description = [{
    Models a piece of fabric hardware (a "module") that can be
    configured to implement any one of the software ops listed in
    `op_list`. Members of `op_list` must either be a single op or all
    belong to one shared hardware-share group. Hardware ports are
    `fabric.bits<N>` only (other fabric types are rejected).

    Attributes:
      * `op_list`    : array of `FlatSymbolRefAttr`, e.g.
                       `[@arith.addi, @arith.subi]`. Printed in `[]`
                       since the list is itself a hardware property.
      * `hw_params`  : `[ {key = ...} ]` - an `ArrayAttr` of length 1
                       wrapping a `DictionaryAttr` whose keys/values
                       describe the hardware-supported set for each
                       configurable parameter. The outer `[]` marks
                       this as a hardware attribute. Optional (some
                       ops have no hardware parameters).
      * `sw_configs` : `{ key = ... }` - a `DictionaryAttr` with
                       runtime-config values selecting concrete
                       behaviour. When `op_list` has more than one
                       member and the op is programmed, `sw_configs`
                       must contain `op_sel` whose string value
                       matches one of the symbols in `op_list`.
                       Follows the fabric all-or-nothing rule:
                       either present (programmed) or absent
                       (pure hardware description).
  }];

  let arguments = (ins
      Variadic<Fabric_BitsType>:$inputs,
      ArrayAttr:$op_list,
      OptionalAttr<ArrayAttr>:$hw_params,
      OptionalAttr<DictionaryAttr>:$sw_configs
  );
  let results = (outs Variadic<Fabric_BitsType>:$outputs);

  let assemblyFormat = [{
    $op_list `(` $inputs `)` attr-dict
      `:` functional-type($inputs, $outputs)
  }];
  let hasVerifier = 1;
}

def Fabric_ModulePortType
    : AnyTypeOf<[Fabric_BitsType, Fabric_BitsTagType, Fabric_TagType,
                 AnyMemRef]>;

def Fabric_ModuleOp : Fabric_Op<"module", [
    IsolatedFromAbove,
    SingleBlockImplicitTerminator<"YieldOp">,
    Symbol,
    DeclareOpInterfaceMethods<RegionKindInterface>
]> {
  let summary =
      "fabric module: top-level container of the fabric dialect";
  let description = [{
    Top-level container for the fabric dialect. Carries a required
    `sym_name`. The body is a single graph-region block whose
    terminator is `fabric.yield`.

    Module inputs are declared as the body's entry-block arguments
    (the op itself has zero SSA operands, like `func.func`). Module
    outputs are declared as result types on the op signature; the
    `fabric.yield` terminator supplies their values.

    Allowed input/output types: `!fabric.bits<W>`,
    `!fabric.bits_tag<W, T>`, `!fabric.tag<T>`, `memref<...>`. No
    other MLIR types are accepted.

    The body whitelist is limited to a fixed set of canonical fabric
    container ops; today that set is `fabric.spatial_pe` and
    `fabric.fifo` (plus the implicit `fabric.yield` terminator).
    Future container ops (`fabric.temporal_pe`, `fabric.spatial_sw`,
    `fabric.instantiate`, ...) will be added to the whitelist when
    they land. See `docs/spec-fabric-module.md` for the full rules.
  }];

  let arguments = (ins
      SymbolNameAttr:$sym_name,
      TypeAttrOf<FunctionType>:$function_type
  );
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Fabric_SpatialPeOp : Fabric_Op<"spatial_pe", [
    IsolatedFromAbove,
    SingleBlock,
    NoTerminator,
    DeclareOpInterfaceMethods<RegionKindInterface>,
    HasParent<"::fabric::ModuleOp">
]> {
  let summary =
      "fabric spatial PE: a container of one or more fabric.fu ops with at "
      "most one architecturally active per configuration";
  let description = [{
    Spatial processing element. Holds one or more `fabric.fu` ops in its body.
    A configuration of the PE selects at most one inner FU as architecturally
    active and routes the PE's external input ports to that FU's input ports
    via the PE-level input-mux fields, and the active FU's output ports back
    onto the PE's external output ports via the PE-level output-demux fields.

    The PE has no terminator: inner FUs are not SSA-wired to the PE results.
    Wiring is described by the PE's runtime configuration record (see the
    spatial_pe spec), not by `fabric.yield`.

    Constraints:
      * `K = numInputs() >= 1` and `L = numOutputs() >= 1`.
      * Every PE operand and every PE result is `!fabric.bits<W>` with the
        same `W`. Mixing widths or non-`bits` fabric types on the PE
        boundary is rejected.
      * The body holds one or more `fabric.fu` ops and nothing else. No
        `fabric.op`, `fabric.mux`, `fabric.demux`, `fabric.fifo`,
        `fabric.spatial_pe`, or `fabric.yield` may appear directly in the
        body.
      * For every inner FU `f`, `f.numInputs() <= K` and
        `f.numOutputs() <= L`. Every FU outer input port and every FU
        outer output port has type `!fabric.bits<W>` matching the PE's
        `W`. Inner FU block-arg types may be narrower
        (`!fabric.bits<F>` with `F <= W`), in which case the FU
        boundary truncates the high `W - F` bits.
      * The op must appear inside a `fabric.module` body.
  }];

  let arguments = (ins Variadic<Fabric_BitsType>:$inputs);
  let results = (outs Variadic<Fabric_BitsType>:$outputs);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Fabric_FuOp : Fabric_Op<"fu", [
    IsolatedFromAbove,
    SingleBlockImplicitTerminator<"YieldOp">,
    DeclareOpInterfaceMethods<RegionKindInterface>
]> {
  let summary =
      "fabric function unit (FU): a graph-region tile of fabric.op / mux / demux";
  let description = [{
    Container for one CGRA-style functional unit. Inputs and outputs
    are `fabric.bits<N>` ports. The body is a single graph-kind block
    (no SSA dominance: cycles / feedback edges are permitted) whose
    only permitted ops are `fabric.op`, `fabric.mux`, `fabric.demux`
    and the `fabric.yield` terminator. The mux / demux ops inside an
    FU let different software configurations realize different
    internal compute graphs over the same hardware: they are *not*
    required to attach to the FU's external inputs / outputs, their
    role is to reshape the in-FU op connectivity. Allowing back-edges
    in the body lets configurable compute ops (e.g.
    `fabric.op[@dataflow.carry]`) and user-facing cyclic
    `dataflow.subgraph`s map onto a single FU.

    The FU input boundary supports high-bit truncation. The textual
    form `(%fa = %src : !fabric.bits<W> to !fabric.bits<F>)` declares
    an outer operand of width `W` and an inner block argument of width
    `F` with `F <= W`; the high `W - F` bits of `%src` are dropped at
    the FU boundary. Without the `to ...` clause the inner type
    defaults to the outer type. Output ports stay strict (yield value
    types must match the FU's outer result types).

    Constraints:
      * The body must contain at least one `fabric.op`.
      * `fabric.fu` cannot be nested.
      * `fabric.fifo` is not permitted in the body.
      * The FU itself carries no `[hw_params]` / `{sw_configs}`;
        all configuration lives on the inner ops.
      * For each input port `i`, the outer operand width must be
        greater than or equal to the inner block-arg width
        (`outer >= inner`).
  }];

  let arguments = (ins Variadic<Fabric_BitsType>:$inputs);
  let results = (outs Variadic<Fabric_BitsType>:$outputs);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Fabric_YieldOp : Fabric_Op<"yield", [
    Terminator,
    Pure
]> {
  let summary = "terminator of a fabric.fu or fabric.module region";
  let description = [{
    Closes the enclosing `fabric.fu` or `fabric.module` body.

    Inside `fabric.fu`, `$values` supplies the values that leave the FU
    on its results; the number and types must match the parent op's
    result count and types exactly.

    Inside `fabric.module`, `$values` supplies the module's declared
    output values. The number and per-value types must match the
    module's declared result list under the width-relaxation rule
    documented in `docs/spec-fabric-module.md` (low-bit alignment, same
    fabric kind; memref must match exactly). The optional per-value
    `to <module-result-type>` clause makes the relaxation explicit:

      fabric.yield %v : !fabric.bits<32> to !fabric.bits<16>
  }];

  // Yield operands are the SSA values being yielded. They must be one of
  // the allowed module port types when this yield closes a fabric.module
  // body, and !fabric.bits<W> when it closes a fabric.fu body. The op
  // accepts the union here; `verify` enforces the per-parent rule.
  let arguments = (ins Variadic<Fabric_ModulePortType>:$values);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, ::mlir::ValueRange{});
    }]>
  ];
}

#endif // FABRIC_OPS_TD
