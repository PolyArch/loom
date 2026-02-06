//===- FabricOps.td - Fabric operations definition -----------*- tablegen -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_DIALECT_FABRIC_FABRICOPS_TD
#define LOOM_DIALECT_FABRIC_FABRICOPS_TD

include "loom/Dialect/Fabric/FabricDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Fabric_Op<string mnemonic, list<Trait> traits = []>
    : Op<Fabric_Dialect, mnemonic, traits>;

include "loom/Dialect/Fabric/FabricTagOps.td"
include "loom/Dialect/Fabric/FabricSwitchOps.td"
include "loom/Dialect/Fabric/FabricTopOps.td"
include "loom/Dialect/Fabric/FabricPEOps.td"
include "loom/Dialect/Fabric/FabricMemOps.td"

#endif // LOOM_DIALECT_FABRIC_FABRICOPS_TD
