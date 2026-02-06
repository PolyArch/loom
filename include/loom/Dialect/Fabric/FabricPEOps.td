//===- FabricPEOps.td - Fabric PE operations -----------------*- tablegen -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_DIALECT_FABRIC_FABRICPEOPS_TD
#define LOOM_DIALECT_FABRIC_FABRICPEOPS_TD

//===----------------------------------------------------------------------===//
// fabric.pe
//
// Named form:
//   fabric.pe @name(%arg0: T0, %arg1: T1) -> (R0)
//     [latency = [...], interval = [...], output_tag = [...]] {
//     // body
//     fabric.yield %result : R0
//   }
//
// Inline form:
//   %out = fabric.pe %in0, %in1
//     [latency = [...], interval = [...]]
//     : (T0, T1) -> (R0) {
//     ^bb0(%a: VT0, %b: VT1):
//       // body
//       fabric.yield %v : VR0
//   }
//===----------------------------------------------------------------------===//

def Fabric_PEOp : Fabric_Op<"pe", [
    SingleBlock,
    SingleBlockImplicitTerminator<"YieldOp">
]> {
  let summary = "Processing element with computation body";
  let description = [{
    Defines a processing element that performs computation on input values.
    For tagged interface, tags are stripped on entry and reattached on exit.
    The body operates on value types only.
  }];

  let arguments = (ins
    Variadic<AnyType>:$inputs,
    OptionalAttr<SymbolNameAttr>:$sym_name,
    OptionalAttr<TypeAttrOf<FunctionType>>:$function_type,
    OptionalAttr<ArrayAttr>:$latency,
    OptionalAttr<ArrayAttr>:$interval,
    OptionalAttr<ArrayAttr>:$output_tag,
    OptionalAttr<AnyAttr>:$constant_value,
    OptionalAttr<I64Attr>:$cont_cond_sel,
    OptionalAttr<I64Attr>:$lqDepth,
    OptionalAttr<I64Attr>:$sqDepth
  );
  let results = (outs Variadic<AnyType>:$outputs);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// fabric.temporal_pe
//
// Spec syntax (always named):
//   fabric.temporal_pe @name(
//     %in0: !dataflow.tagged<T, iJ>, ...
//   ) -> (!dataflow.tagged<T, iJ>, ...)
//     [num_register = R, num_instruction = I, reg_fifo_depth = F,
//      enable_share_operand_buffer = false, operand_buffer_size = S]
//     {instruction_mem = [...]} {
//     // FU body
//     fabric.yield %out0, %out1, ...
//   }
//===----------------------------------------------------------------------===//

def Fabric_TemporalPEOp : Fabric_Op<"temporal_pe", [
    SingleBlock,
    SingleBlockImplicitTerminator<"YieldOp">,
    Symbol
]> {
  let summary = "Time-multiplexed processing element";
  let description = [{
    Defines a temporal processing element that multiplexes multiple FU types
    across instruction slots. All ports must be !dataflow.tagged with the same
    type. The body contains FU definitions using fabric.pe.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    I64Attr:$num_register,
    I64Attr:$num_instruction,
    I64Attr:$reg_fifo_depth,
    DefaultValuedAttr<BoolAttr, "false">:$enable_share_operand_buffer,
    OptionalAttr<I64Attr>:$operand_buffer_size,
    OptionalAttr<ArrayAttr>:$instruction_mem
  );
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif // LOOM_DIALECT_FABRIC_FABRICPEOPS_TD
