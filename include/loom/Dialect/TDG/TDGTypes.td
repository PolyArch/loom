#ifndef LOOM_DIALECT_TDG_TDGTYPES_TD
#define LOOM_DIALECT_TDG_TDGTYPES_TD

include "loom/Dialect/TDG/TDGDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"

class TDG_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TDG_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// !tdg.contract_ref - Reference to a contract (for annotations)
def TDG_ContractRefType : TDG_Type<"ContractRef", "contract_ref"> {
  let summary = "Reference to a TDG contract";
  let description = [{
    A reference type that points to a specific contract within a TDG graph.
    Used for annotations and cross-references between IR constructs.
  }];
  let parameters = (ins "::mlir::StringAttr":$contractName);
  let assemblyFormat = "`<` $contractName `>`";
}

// !tdg.kernel_ref - Reference to a kernel
def TDG_KernelRefType : TDG_Type<"KernelRef", "kernel_ref"> {
  let summary = "Reference to a TDG kernel";
  let description = [{
    A reference type that points to a specific kernel within a TDG graph.
    Used for annotations and cross-references between IR constructs.
  }];
  let parameters = (ins "::mlir::StringAttr":$kernelName);
  let assemblyFormat = "`<` $kernelName `>`";
}

#endif
