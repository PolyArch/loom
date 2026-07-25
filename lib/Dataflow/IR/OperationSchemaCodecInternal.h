#ifndef LOOM_DATAFLOW_IR_OPERATION_SCHEMA_CODEC_INTERNAL_H
#define LOOM_DATAFLOW_IR_OPERATION_SCHEMA_CODEC_INTERNAL_H

#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Dataflow/IR/OperationSchema.h"

#include "llvm/Support/Error.h"

#include <cstdint>

namespace dataflow::detail {

llvm::Expected<std::uint32_t>
serviceValueRoleWireTag(semantics::ServiceValueRole role);
llvm::Expected<semantics::ServiceValueRole>
serviceValueRoleFromWireTag(std::uint32_t wireTag);

llvm::Expected<std::uint32_t>
memoryAccessFormWireTag(semantics::MemoryAccessForm form);
llvm::Expected<semantics::MemoryAccessForm>
memoryAccessFormFromWireTag(std::uint32_t wireTag);

llvm::Expected<std::uint32_t>
memoryMaskFormWireTag(semantics::MemoryMaskForm form);
llvm::Expected<semantics::MemoryMaskForm>
memoryMaskFormFromWireTag(std::uint32_t wireTag);

llvm::Expected<std::uint32_t> atomicOrderingWireTag(AtomicOrdering ordering);
llvm::Expected<AtomicOrdering> atomicOrderingFromWireTag(std::uint32_t wireTag);

llvm::Expected<std::uint32_t> syncScopeKindWireTag(SyncScopeKind kind);
llvm::Expected<SyncScopeKind> syncScopeKindFromWireTag(std::uint32_t wireTag);

llvm::Expected<std::uint32_t>
vectorAtomicGranularityWireTag(VectorAtomicGranularity granularity);
llvm::Expected<VectorAtomicGranularity>
vectorAtomicGranularityFromWireTag(std::uint32_t wireTag);

llvm::Expected<std::uint32_t> atomicRmwKindWireTag(AtomicRmwKind kind);
llvm::Expected<AtomicRmwKind> atomicRmwKindFromWireTag(std::uint32_t wireTag);

} // namespace dataflow::detail

#endif // LOOM_DATAFLOW_IR_OPERATION_SCHEMA_CODEC_INTERNAL_H
