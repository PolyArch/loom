#ifndef DATAFLOW_ATTRS_TD
#define DATAFLOW_ATTRS_TD

include "Dataflow/IR/DataflowDialect.td"
include "Dataflow/IR/DataflowEnums.td"
include "mlir/IR/AttrTypeBase.td"

class Dataflow_Attr<string name, string attrMnemonic,
                    list<Trait> traits = []>
    : AttrDef<Dataflow_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// Memory Access Contracts
//
// Every canonical memory actor owns exactly one applicable aggregate contract.
// The aggregate is the only persistent owner of its fields: there are no
// independent ordering, scope, atomicity, read-modify-write kind, weakness,
// volatility, MMIO, or coherence attributes that can disagree with it.
//===----------------------------------------------------------------------===//

def Dataflow_Ordering : EnumParameter<Dataflow_AtomicOrdering>;

def Dataflow_OptionalGranularity : OptionalParameter<
    "::std::optional<::dataflow::VectorAtomicGranularity>",
    "vector atomic object granularity, absent for a scalar atomic access">;

def Dataflow_Flag : DefaultValuedParameter<"bool", "false">;

def Dataflow_SyncScopeRefAttr : Dataflow_Attr<"SyncScopeRef", "sync_scope"> {
  let summary = "closed synchronization scope reference";
  let description = [{
    The closed sum `System | SingleThread | TargetScopeKey`. A target scope is
    a typed target-namespaced key, never a free unscoped string; the
    compiler-target contract uniquely owns its meaning. A target-specific
    source scope that cannot be resolved to such a key cannot enter a
    SpatialCore graph.
  }];

  let parameters = (ins EnumParameter<Dataflow_SyncScopeKind>:$kind,
                        OptionalParameter<"::mlir::StringAttr">:$target_namespace,
                        OptionalParameter<"::mlir::StringAttr">:$target_key);
  let assemblyFormat = "`<` $kind (`,` $target_namespace^ `,` $target_key)? `>`";
  let genVerifyDecl = 1;

  let builders = [
    AttrBuilder<(ins "::dataflow::SyncScopeKind":$kind), [{
      return $_get($_ctxt, kind, ::mlir::StringAttr(), ::mlir::StringAttr());
    }]>
  ];
}

def Dataflow_AtomicAccessContractAttr
    : Dataflow_Attr<"AtomicAccessContract", "atomic_access"> {
  let summary = "the common atomic-access fields as one closed typed value";
  let description = [{
    The single owner of an atomic access's ordering, synchronization scope,
    optional vector atomic granularity, and volatility. Every actor that
    performs an atomic access nests exactly one of these; none of its fields
    is repeated by an enclosing aggregate.

    A scalar atomic access omits the granularity because both vector cases
    degenerate to one atomic object.
  }];

  let parameters = (ins Dataflow_Ordering:$ordering,
                        "::dataflow::SyncScopeRefAttr":$sync_scope,
                        Dataflow_OptionalGranularity:$vector_granularity,
                        Dataflow_Flag:$is_volatile);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def Dataflow_PlainAccessContractAttr
    : Dataflow_Attr<"PlainAccessContract", "plain_access"> {
  let summary = "the plain arm of a memory access contract";
  let description = [{
    A plain access carries no ordering, scope, or atomic granularity. Volatile
    is an observability contract, not synchronization: it neither creates a
    synchronizes-with relation nor makes the access atomic.
  }];

  let parameters = (ins "bool":$is_volatile);
  let assemblyFormat = "`<` struct(params) `>`";
}

// MemoryAccessContract = Plain { volatile } | Atomic(AtomicAccessContract).
// The two arms are distinct attributes so that no instance can carry two
// owners of the same field.
def Dataflow_MemoryAccessContract : AnyAttrOf<[
    Dataflow_PlainAccessContractAttr, Dataflow_AtomicAccessContractAttr],
    "plain or atomic memory access contract">;

def Dataflow_AtomicRmwContractAttr
    : Dataflow_Attr<"AtomicRmwContract", "rmw_contract"> {
  let summary = "contract of one atomic read-modify-write actor";
  let description = [{
    One enumerated read-modify-write action together with the one
    `AtomicAccessContract` it performs. Canonical Dataflow carries no generic
    atomic-region body: a region normalizes to this actor only when it is
    proven equivalent to one enumerated action.
  }];

  let parameters = (ins EnumParameter<Dataflow_AtomicRmwKind>:$kind,
                        "::dataflow::AtomicAccessContractAttr":$access);
  let assemblyFormat = "`<` struct(params) `>`";
}

def Dataflow_CompareExchangeContractAttr
    : Dataflow_Attr<"CompareExchangeContract", "cmpxchg_contract"> {
  let summary = "contract of one compare-exchange actor";
  let description = [{
    Preserves distinct success and failure orderings and strong versus weak
    behavior. A failed compare-exchange performs no write; a weak
    compare-exchange may fail spuriously.
  }];

  let parameters = (ins Dataflow_Ordering:$success_ordering,
                        Dataflow_Ordering:$failure_ordering,
                        "::dataflow::SyncScopeRefAttr":$sync_scope,
                        Dataflow_OptionalGranularity:$vector_granularity,
                        Dataflow_Flag:$weak,
                        Dataflow_Flag:$is_volatile);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def Dataflow_FenceContractAttr
    : Dataflow_Attr<"FenceContract", "fence_contract"> {
  let summary = "contract of one fence actor";
  let description = [{
    A fence publishes or consumes visibility under one ordering and scope. It
    addresses no memory and therefore has no access shape or granularity.
  }];

  let parameters = (ins Dataflow_Ordering:$ordering,
                        "::dataflow::SyncScopeRefAttr":$sync_scope);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

#endif // DATAFLOW_ATTRS_TD
