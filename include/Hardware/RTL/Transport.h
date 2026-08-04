#ifndef LOOM_HARDWARE_RTL_TRANSPORT_H
#define LOOM_HARDWARE_RTL_TRANSPORT_H

#include "Fabric/IR/BoundaryDataPath.h"

#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Error.h"

#include <optional>

namespace mlir {
class OpBuilder;
}

namespace loom::hardware::rtl {

/// The forward half of a Fabric ready/valid connection. Ready travels in the
/// opposite direction and is connected unchanged by the structural owner.
struct ForwardTransportSignals final {
  mlir::Value valid;
  std::optional<mlir::Value> payload;
  std::optional<mlir::Value> tag;
};

llvm::Expected<ForwardTransportSignals>
adaptForwardTransportSignals(mlir::OpBuilder &builder, mlir::Location location,
                             ::fabric::DataPathType sourceType,
                             ::fabric::DataPathType destinationType,
                             ForwardTransportSignals sourceSignals);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_TRANSPORT_H
