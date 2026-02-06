//===-- FabricDialect.cpp - Fabric dialect registration ---------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Dialect/Fabric/FabricDialect.h"
#include "loom/Dialect/Fabric/FabricOps.h"
#include "loom/Dialect/Dataflow/DataflowDialect.h"

#include "circt/Dialect/Handshake/HandshakeInterfaces.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace loom::fabric;

#include "FabricDialect.cpp.inc"

#define GET_OP_CLASSES
#include "FabricOps.cpp.inc"

namespace {
struct PEOpDataflowModel
    : public circt::handshake::FineGrainedDataflowRegionOpInterface::
          ExternalModel<PEOpDataflowModel, PEOp> {};
} // namespace

void FabricDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "FabricOps.cpp.inc"
      >();
  PEOp::attachInterface<PEOpDataflowModel>(*getContext());
}
