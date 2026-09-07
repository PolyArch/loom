#ifndef LOOM_LIB_HARDWARE_RTL_HIERARCHY_MEMORYDIAGNOSTICS_H
#define LOOM_LIB_HARDWARE_RTL_HIERARCHY_MEMORYDIAGNOSTICS_H

#include "Support.h"
#include "Dataflow/IR/DataflowServiceSchema.h"

namespace loom::hardware::rtl::hierarchy {

/// Borrowed SSA values from the memory semantic owner; no new state equations.
struct MemoryOperandDiagnostics final {
  ::dataflow::semantics::ServiceValueRole role;
  mlir::Value present;
  mlir::Value valid;
  mlir::Value internal;
  mlir::Value endpoint;
  mlir::Value tag;
  mlir::Value occupied;
  mlir::Value enqueue;
  mlir::Value dequeue;
  mlir::Value data;
  mlir::Value incoming;
};

struct MemoryRowDiagnostics final {
  mlir::Value context;
  mlir::Value active;
  mlir::Value write;
  mlir::Value requestValid;
  mlir::Value issued;
  mlir::Value response;
  mlir::Value released;
  mlir::Value occupied;
  mlir::Value completed;
  mlir::Value selected;
  mlir::Value address;
  mlir::Value data;
  mlir::Value result;
  mlir::Value resultNext;
  std::vector<MemoryOperandDiagnostics> operands;
};

/// Emits simulation-only observation of existing owner signals. The common
/// runtime verbosity controls presentation; artifact generation is unconditional.
void emitMemoryDiagnostics(mlir::OpBuilder &builder, mlir::Location location,
                           circt::hw::HWModulePortAccessor &accessor,
                           mlir::Value contextBase,
                           llvm::ArrayRef<EndpointPlan> endpoints,
                           llvm::ArrayRef<MemoryRowDiagnostics> rows);

} // namespace loom::hardware::rtl::hierarchy

#endif
