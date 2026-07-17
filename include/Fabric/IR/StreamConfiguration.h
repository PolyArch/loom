#ifndef FABRIC_IR_STREAMCONFIGURATION_H
#define FABRIC_IR_STREAMCONFIGURATION_H

#include "Dataflow/IR/DataflowEnums.h"
#include "Fabric/IR/FabricOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <string>

namespace fabric {

struct StreamConfiguration {
  dataflow::StreamStepKind stepKind;
  llvm::SmallVector<mlir::arith::CmpIPredicate, 10> predicates;
  std::optional<mlir::arith::CmpIPredicate> selectedPredicate;

  bool supports(mlir::arith::CmpIPredicate predicate) const;
};

mlir::FailureOr<StreamConfiguration>
parseStreamConfiguration(OpOp op, std::string &error);

} // namespace fabric

#endif // FABRIC_IR_STREAMCONFIGURATION_H
