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

def Fabric_EntityIdAttr : Fabric_Attr<"EntityId", "entity_id"> {
  let summary = "materialized canonical Fabric entity identifier";
  let description = [{
    The artifact-global unsigned 64-bit EntityId assigned by the Fabric
    finalizer. It is a derived projection of canonical structure, never an
    authoring input. Finalization strips any supplied value before labeling,
    and strict import recomputes and verifies it.
  }];
  let parameters = (ins "uint64_t":$id);
  let assemblyFormat = "`<` $id `>`";
}

def Fabric_FuCapabilityDomainAttr
    : Fabric_Attr<"FuCapabilityDomain", "fu_capability_domain"> {
  let summary = "canonical correlated fabric.fu capability-template domain";
  let parameters = (ins "::mlir::DenseI8ArrayAttr":$record);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def Fabric_SwitchFixedPriorityAttr
    : Fabric_Attr<"SwitchFixedPriority", "switch_fixed_priority"> {
  let summary = "fixed-priority temporal-switch requester order";
  let parameters = (ins "::mlir::DenseI64ArrayAttr":$requester_order);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def Fabric_SwitchRoundRobinAttr
    : Fabric_Attr<"SwitchRoundRobin", "switch_round_robin"> {
  let summary = "round-robin temporal-switch requester cycle";
  let parameters = (ins
      "::mlir::DenseI64ArrayAttr":$requester_cycle,
      "uint64_t":$reset_requester);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def Fabric_MemoryServiceContractAttr
    : Fabric_Attr<"MemoryServiceContract", "memory_service_contract"> {
  let summary = "canonical typed memory service contract record";
  let parameters = (ins
      "::mlir::DenseI8ArrayAttr":$record);
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

def Fabric_MemoryResidentContextsAttr
    : Fabric_Attr<"MemoryResidentContexts", "memory_resident_contexts"> {
  let summary = "bounded temporal memory operation-table row domain";
  let parameters = (ins "uint64_t":$count);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def Fabric_MemoryConnectivityContractAttr
    : Fabric_Attr<"MemoryConnectivityContract",
                  "memory_connectivity_contract"> {
  let summary = "canonical fixed memory dispatch and internal connectivity";
  let parameters = (ins "::mlir::DenseI8ArrayAttr":$record);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def Fabric_MemoryEngineAttr
    : Fabric_Attr<"MemoryEngine", "memory_engine"> {
  let summary = "memory operation engine occurrence contract";
  let description = [{
    `operation_issue_depth` is the number of firings one bound memory actor
    may hold outstanding before the oldest of them retires. Retirement stays
    in issue order, so depth one is the serialized engine and a larger depth
    is the engine's own memory-level parallelism, independent of any cache
    outstanding-miss capacity or service outstanding guarantee. The canonical
    text elides the depth exactly when it is the serialized depth.
  }];
  let parameters = (ins
      EnumParameter<Fabric_ScheduleAttr>:$schedule,
      OptionalParameter<"::fabric::MemoryResidentContextsAttr">:
          $resident_contexts,
      DefaultValuedParameter<
          "uint64_t", "::fabric::serializedMemoryOperationIssueDepth">:
          $operation_issue_depth);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def Fabric_MemoryContractAttr
    : Fabric_Attr<"MemoryContract", "memory_contract"> {
  let summary = "typed fabric.mem occurrence contract";
  let parameters = (ins
      OptionalParameter<"::fabric::MemoryEngineAttr">:$engine,
      OptionalParameter<"::fabric::LocalMemoryServiceAttr">:$local_service,
      OptionalParameter<"::fabric::MemoryConnectivityContractAttr">:
          $connectivity,
      "::mlir::DenseI32ArrayAttr":$manager_endpoints,
      "::mlir::DenseI32ArrayAttr":$subordinate_endpoints);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

#endif // FABRIC_ATTRS_TD
