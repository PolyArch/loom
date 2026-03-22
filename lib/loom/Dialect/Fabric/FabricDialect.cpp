#include "loom/Dialect/Fabric/FabricDialect.h"
#include "loom/Dialect/Fabric/FabricOps.h"
#include "loom/Dialect/Fabric/FabricTypes.h"
#include "loom/Dialect/Dataflow/DataflowDialect.h"
#include "circt/Dialect/Handshake/HandshakeInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace loom::fabric;

#include "loom/Dialect/Fabric/FabricDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "loom/Dialect/Fabric/FabricTypes.cpp.inc"

namespace {
// Allow handshake ops inside fabric.function_unit by implementing the
// FineGrainedDataflowRegionOpInterface marker interface.
struct FUDataflowModel
    : public circt::handshake::FineGrainedDataflowRegionOpInterface::
          ExternalModel<FUDataflowModel, FunctionUnitOp> {};
} // namespace

void FabricDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "loom/Dialect/Fabric/FabricOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "loom/Dialect/Fabric/FabricTypes.cpp.inc"
      >();

  FunctionUnitOp::attachInterface<FUDataflowModel>(*getContext());
}
