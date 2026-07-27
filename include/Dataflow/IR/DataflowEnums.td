#ifndef DATAFLOW_ENUMS_TD
#define DATAFLOW_ENUMS_TD

include "Dataflow/IR/DataflowDialect.td"
include "mlir/IR/EnumAttr.td"

// The closed logical-domain discriminator of one dataflow.thread definition.
// Coordinate rank remains derived from the definition's entry-block suffix;
// it is deliberately absent from this enum and its enclosing attribute.
def Dataflow_ThreadDomainDenseRectangular
    : I32EnumAttrCase<"DenseRectangular", 0, "dense">;
def Dataflow_ThreadDomainDynamicWork
    : I32EnumAttrCase<"DynamicWork", 1, "dynamic_work">;

def Dataflow_ThreadDomainKind : I32EnumAttr<
    "ThreadDomainKind", "logical thread domain discriminator",
    [Dataflow_ThreadDomainDenseRectangular,
     Dataflow_ThreadDomainDynamicWork]> {
  let cppNamespace = "::dataflow";
  let genSpecializedAttr = 0;
}

def Dataflow_StreamStepAdd : I32EnumAttrCase<"Add", 0, "add">;
def Dataflow_StreamStepSub : I32EnumAttrCase<"Sub", 1, "sub">;
def Dataflow_StreamStepMul : I32EnumAttrCase<"Mul", 2, "mul">;
def Dataflow_StreamStepSDiv : I32EnumAttrCase<"SDiv", 3, "sdiv">;
def Dataflow_StreamStepUDiv : I32EnumAttrCase<"UDiv", 4, "udiv">;
def Dataflow_StreamStepShL : I32EnumAttrCase<"ShL", 5, "shl">;
def Dataflow_StreamStepAShr : I32EnumAttrCase<"AShr", 6, "ashr">;
def Dataflow_StreamStepLShr : I32EnumAttrCase<"LShr", 7, "lshr">;

def Dataflow_StreamStepKindAttr : I32EnumAttr<
    "StreamStepKind", "integer recurrence update operation",
    [Dataflow_StreamStepAdd, Dataflow_StreamStepSub,
     Dataflow_StreamStepMul, Dataflow_StreamStepSDiv,
     Dataflow_StreamStepUDiv, Dataflow_StreamStepShL,
     Dataflow_StreamStepAShr, Dataflow_StreamStepLShr]> {
  let cppNamespace = "::dataflow";
}

//===----------------------------------------------------------------------===//
// Memory Consistency Vocabulary
//
// These enumerations reuse the LLVM semantic vocabulary by case name. Their
// numeric values are an implementation detail of this dialect: translation to
// or from any other ordering, scope, or read-modify-write vocabulary must go
// through the case names, never through a numeric coincidence.
//===----------------------------------------------------------------------===//

// The closed set of software atomic orderings. There is deliberately no
// "not atomic" case: a plain access is expressed by a plain
// MemoryAccessContract, never by an ordering value.
def Dataflow_OrderingUnordered : I32EnumAttrCase<"Unordered", 0, "unordered">;
def Dataflow_OrderingMonotonic : I32EnumAttrCase<"Monotonic", 1, "monotonic">;
def Dataflow_OrderingAcquire : I32EnumAttrCase<"Acquire", 2, "acquire">;
def Dataflow_OrderingRelease : I32EnumAttrCase<"Release", 3, "release">;
def Dataflow_OrderingAcqRel : I32EnumAttrCase<"AcqRel", 4, "acq_rel">;
def Dataflow_OrderingSeqCst : I32EnumAttrCase<"SeqCst", 5, "seq_cst">;

def Dataflow_AtomicOrdering : I32EnumAttr<
    "AtomicOrdering", "software atomic ordering",
    [Dataflow_OrderingUnordered, Dataflow_OrderingMonotonic,
     Dataflow_OrderingAcquire, Dataflow_OrderingRelease,
     Dataflow_OrderingAcqRel, Dataflow_OrderingSeqCst]> {
  let cppNamespace = "::dataflow";
  let genSpecializedAttr = 0;
}

// The discriminator of the closed SyncScopeRef sum. A target scope carries a
// typed target-namespaced key rather than a free unscoped string.
def Dataflow_SyncScopeSystem : I32EnumAttrCase<"System", 0, "system">;
def Dataflow_SyncScopeSingleThread
    : I32EnumAttrCase<"SingleThread", 1, "single_thread">;
def Dataflow_SyncScopeTarget : I32EnumAttrCase<"Target", 2, "target">;

def Dataflow_SyncScopeKind : I32EnumAttr<
    "SyncScopeKind", "synchronization scope discriminator",
    [Dataflow_SyncScopeSystem, Dataflow_SyncScopeSingleThread,
     Dataflow_SyncScopeTarget]> {
  let cppNamespace = "::dataflow";
  let genSpecializedAttr = 0;
}

// Which objects one vector atomic access treats as atomic.
def Dataflow_GranularityWholePayload
    : I32EnumAttrCase<"WholePayload", 0, "whole_payload">;
def Dataflow_GranularityPerLane : I32EnumAttrCase<"PerLane", 1, "per_lane">;

def Dataflow_VectorAtomicGranularity : I32EnumAttr<
    "VectorAtomicGranularity", "vector atomic object granularity",
    [Dataflow_GranularityWholePayload, Dataflow_GranularityPerLane]> {
  let cppNamespace = "::dataflow";
  let genSpecializedAttr = 0;
}

// The closed set of read-modify-write actions, with exact LLVM `atomicrmw`
// semantics.
def Dataflow_RmwXchg : I32EnumAttrCase<"Xchg", 0, "xchg">;
def Dataflow_RmwAdd : I32EnumAttrCase<"Add", 1, "add">;
def Dataflow_RmwSub : I32EnumAttrCase<"Sub", 2, "sub">;
def Dataflow_RmwAnd : I32EnumAttrCase<"And", 3, "and">;
def Dataflow_RmwNand : I32EnumAttrCase<"Nand", 4, "nand">;
def Dataflow_RmwOr : I32EnumAttrCase<"Or", 5, "or">;
def Dataflow_RmwXor : I32EnumAttrCase<"Xor", 6, "xor">;
def Dataflow_RmwMax : I32EnumAttrCase<"Max", 7, "max">;
def Dataflow_RmwMin : I32EnumAttrCase<"Min", 8, "min">;
def Dataflow_RmwUMax : I32EnumAttrCase<"UMax", 9, "umax">;
def Dataflow_RmwUMin : I32EnumAttrCase<"UMin", 10, "umin">;
def Dataflow_RmwFAdd : I32EnumAttrCase<"FAdd", 11, "fadd">;
def Dataflow_RmwFSub : I32EnumAttrCase<"FSub", 12, "fsub">;
def Dataflow_RmwFMax : I32EnumAttrCase<"FMax", 13, "fmax">;
def Dataflow_RmwFMin : I32EnumAttrCase<"FMin", 14, "fmin">;
def Dataflow_RmwUIncWrap : I32EnumAttrCase<"UIncWrap", 15, "uinc_wrap">;
def Dataflow_RmwUDecWrap : I32EnumAttrCase<"UDecWrap", 16, "udec_wrap">;
def Dataflow_RmwUSubCond : I32EnumAttrCase<"USubCond", 17, "usub_cond">;
def Dataflow_RmwUSubSat : I32EnumAttrCase<"USubSat", 18, "usub_sat">;
def Dataflow_RmwFMaximum : I32EnumAttrCase<"FMaximum", 19, "fmaximum">;
def Dataflow_RmwFMinimum : I32EnumAttrCase<"FMinimum", 20, "fminimum">;
def Dataflow_RmwFMaximumNum : I32EnumAttrCase<"FMaximumNum", 21, "fmaximumnum">;
def Dataflow_RmwFMinimumNum : I32EnumAttrCase<"FMinimumNum", 22, "fminimumnum">;

def Dataflow_AtomicRmwKind : I32EnumAttr<
    "AtomicRmwKind", "atomic read-modify-write action",
    [Dataflow_RmwXchg, Dataflow_RmwAdd, Dataflow_RmwSub, Dataflow_RmwAnd,
     Dataflow_RmwNand, Dataflow_RmwOr, Dataflow_RmwXor, Dataflow_RmwMax,
     Dataflow_RmwMin, Dataflow_RmwUMax, Dataflow_RmwUMin, Dataflow_RmwFAdd,
     Dataflow_RmwFSub, Dataflow_RmwFMax, Dataflow_RmwFMin,
     Dataflow_RmwUIncWrap, Dataflow_RmwUDecWrap, Dataflow_RmwUSubCond,
     Dataflow_RmwUSubSat, Dataflow_RmwFMaximum, Dataflow_RmwFMinimum,
     Dataflow_RmwFMaximumNum, Dataflow_RmwFMinimumNum]> {
  let cppNamespace = "::dataflow";
  let genSpecializedAttr = 0;
}

#endif // DATAFLOW_ENUMS_TD
