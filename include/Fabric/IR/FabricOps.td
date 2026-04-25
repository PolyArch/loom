#ifndef FABRIC_OPS_TD
#define FABRIC_OPS_TD

include "Fabric/IR/FabricDialect.td"
include "Fabric/IR/FabricTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Fabric_MuxOp : Fabric_Op<"mux", [Pure, SameOperandsAndResultType]> {
  let summary = "fabric multiplexer with handshake";
  let description = [{
    Selects one of N (N > 1) inputs onto the single output. All inputs and the
    output share one fabric type. No hardware parameters (the port width is
    fully described by the shared operand/result type).

    Software parameters (either all set or all unset):
      * `sel`        : i32, 0 <= sel < N (forced 0 when `disconnect` is true)
      * `discard`    : bool
      * `disconnect` : bool

    Unset means the op describes pure hardware (not programmed). Programmed
    combinations must satisfy: !(discard && disconnect).
  }];

  let arguments = (ins
      Variadic<Fabric_AnyType>:$inputs,
      OptionalAttr<I32Attr>:$sel,
      OptionalAttr<BoolAttr>:$discard,
      OptionalAttr<BoolAttr>:$disconnect
  );
  let results = (outs Fabric_AnyType:$output);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Fabric_DemuxOp : Fabric_Op<"demux", [Pure, SameOperandsAndResultType]> {
  let summary = "fabric demultiplexer with handshake";
  let description = [{
    Single input, N outputs (N > 1). All input/output ports share one fabric
    type. No hardware parameters (the port width is fully described by the
    shared operand/result type). The op is the input/output dual of
    fabric.mux.

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
      Fabric_AnyType:$input,
      OptionalAttr<I32Attr>:$sel,
      OptionalAttr<BoolAttr>:$discard,
      OptionalAttr<BoolAttr>:$disconnect
  );
  let results = (outs Variadic<Fabric_AnyType>:$outputs);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Fabric_FifoOp : Fabric_Op<"fifo", [Pure, SameOperandsAndResultType]> {
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

def Fabric_FuOp : Fabric_Op<"fu", [
    IsolatedFromAbove,
    SingleBlockImplicitTerminator<"YieldOp">
]> {
  let summary =
      "fabric function unit (FU): an SSACFG tile of fabric.op / mux / demux";
  let description = [{
    Container for one CGRA-style functional unit. Inputs and outputs
    are `fabric.bits<N>` ports. The body is a single SSACFG block
    whose only permitted ops are `fabric.op`, `fabric.mux`,
    `fabric.demux` and the `fabric.yield` terminator. The mux / demux
    ops inside an FU let different software configurations realize
    different internal compute graphs over the same hardware: they
    are *not* required to attach to the FU's external inputs /
    outputs, their role is to reshape the in-FU op connectivity.

    Constraints:
      * The body must contain at least one `fabric.op`.
      * `fabric.fu` cannot be nested.
      * `fabric.fifo` is not permitted in the body.
      * The FU itself carries no `[hw_params]` / `{sw_configs}`;
        all configuration lives on the inner ops.
  }];

  let arguments = (ins Variadic<Fabric_BitsType>:$inputs);
  let results = (outs Variadic<Fabric_BitsType>:$outputs);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Fabric_YieldOp : Fabric_Op<"yield", [
    Terminator,
    HasParent<"::fabric::FuOp">,
    Pure
]> {
  let summary = "terminator of a fabric.fu region";
  let description = [{
    Supplies the values that leave the enclosing `fabric.fu` on its
    results. The number and types of `$values` must match the parent
    op's result count and types.
  }];

  let arguments = (ins Variadic<Fabric_BitsType>:$values);

  let assemblyFormat = "($values^ `:` type($values))? attr-dict";
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, ::mlir::ValueRange{});
    }]>
  ];
}

#endif // FABRIC_OPS_TD
