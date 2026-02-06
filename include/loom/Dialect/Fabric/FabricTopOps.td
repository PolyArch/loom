//===- FabricTopOps.td - Fabric top-level operations ---------*- tablegen -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_DIALECT_FABRIC_FABRICTOPOPS_TD
#define LOOM_DIALECT_FABRIC_FABRICTOPOPS_TD

//===----------------------------------------------------------------------===//
// fabric.yield
//
// Spec syntax:
//   fabric.yield %v0, %v1 : T0, T1
//===----------------------------------------------------------------------===//

def Fabric_YieldOp : Fabric_Op<"yield", [Terminator]> {
  let summary = "Region terminator for fabric operations";
  let description = [{
    Terminates the body region of a fabric operation. Operand types must match
    the parent operation's result types.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let assemblyFormat = [{
    attr-dict ($operands^ `:` type($operands))?
  }];
}

//===----------------------------------------------------------------------===//
// fabric.module
//
// Spec syntax:
//   fabric.module @name(
//     %mem_in: memref<...>, %native_in: T, %tagged_in: !dataflow.tagged<...>
//   ) -> (memref<...>, T, !dataflow.tagged<...>) {
//     // body
//     fabric.yield %out0, %out1 : T0, T1
//   }
//===----------------------------------------------------------------------===//

def Fabric_ModuleOp : Fabric_Op<"module", [
    IsolatedFromAbove,
    SingleBlock,
    SingleBlockImplicitTerminator<"YieldOp">,
    Symbol
]> {
  let summary = "Top-level accelerator module";
  let description = [{
    Defines a top-level accelerator module containing interconnected fabric
    components. Ports follow strict ordering: memref*, native*, tagged*.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type
  );
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// fabric.instance
//
// Spec syntax:
//   %res0, %res1 = fabric.instance @target(%arg0, %arg1)
//     {sym_name = "u0"} : (T0, T1) -> (R0, R1)
//===----------------------------------------------------------------------===//

def Fabric_InstanceOp : Fabric_Op<"instance", []> {
  let summary = "Instantiate a named fabric component";
  let description = [{
    Instantiates a named fabric component (module, PE, switch, memory, etc.)
    with the given operands and produces the corresponding results.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$module,
    OptionalAttr<StrAttr>:$sym_name,
    Variadic<AnyType>:$operands
  );
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    $module `(` $operands `)` attr-dict `:` functional-type($operands, $results)
  }];
  let hasVerifier = 1;
}

#endif // LOOM_DIALECT_FABRIC_FABRICTOPOPS_TD
