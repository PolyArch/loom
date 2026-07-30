#ifndef LOOM_SIMULATOR_DETERMINISTIC_TRANSCENDENTAL_H
#define LOOM_SIMULATOR_DETERMINISTIC_TRANSCENDENTAL_H

#include "Dataflow/IR/OperationSchema.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Error.h"

namespace loom::sim::detail {

llvm::Expected<llvm::APFloat>
evaluateDeterministicUnaryMath(dataflow::OperationSchemaId schema,
                               const llvm::APFloat &operand);

} // namespace loom::sim::detail

#endif // LOOM_SIMULATOR_DETERMINISTIC_TRANSCENDENTAL_H
