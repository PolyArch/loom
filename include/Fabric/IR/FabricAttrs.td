#ifndef FABRIC_ATTRS_TD
#define FABRIC_ATTRS_TD

include "Fabric/IR/FabricDialect.td"
include "Fabric/IR/FabricEnums.td"
include "mlir/IR/AttrTypeBase.td"

class Fabric_Attr<string name, string attrMnemonic,
                  list<Trait> traits = []>
    : AttrDef<Fabric_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def Fabric_MemoryServiceContractAttr
    : Fabric_Attr<"MemoryServiceContract", "memory_service_contract"> {
  let summary = "closed typed local memory service behavior";
  let parameters = (ins
      EnumParameter<Fabric_MemoryServiceBehaviorAttr>:$behavior);
  let assemblyFormat = "`<` struct(params) `>`";
}

def Fabric_LocalMemoryServiceAttr
    : Fabric_Attr<"LocalMemoryService", "local_memory_service"> {
  let summary = "local physical memory service architecture contract";
  let parameters = (ins
      "uint64_t":$capacity_bytes,
      "::fabric::MemoryServiceContractAttr":$service_contract);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def Fabric_MemoryEngineAttr
    : Fabric_Attr<"MemoryEngine", "memory_engine"> {
  let summary = "memory operation engine occurrence contract";
  let parameters = (ins EnumParameter<Fabric_ScheduleAttr>:$schedule);
  let assemblyFormat = "`<` struct(params) `>`";
}

def Fabric_MemoryContractAttr
    : Fabric_Attr<"MemoryContract", "memory_contract"> {
  let summary = "typed fabric.mem occurrence contract";
  let parameters = (ins
      OptionalParameter<"::fabric::MemoryEngineAttr">:$engine,
      OptionalParameter<"::fabric::LocalMemoryServiceAttr">:$local_service,
      "::mlir::DenseI32ArrayAttr":$manager_endpoints,
      "::mlir::DenseI32ArrayAttr":$subordinate_endpoints);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

#endif // FABRIC_ATTRS_TD
