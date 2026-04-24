#ifndef FABRIC_OPS_TD
#define FABRIC_OPS_TD

include "Fabric/IR/FabricDialect.td"
include "Fabric/IR/FabricTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonAttrConstraints.td"
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

#endif // FABRIC_OPS_TD
