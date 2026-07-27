#include "Dataflow/IR/DataflowDialect.h"

#include "Dataflow/IR/DataflowAttrs.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace dataflow;

#include "Dataflow/IR/DataflowDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Dataflow/IR/DataflowTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Dataflow/IR/DataflowAttrs.cpp.inc"

LogicalResult
ThreadDomainAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                         ThreadDomainKind kind,
                         std::optional<uint64_t> workItemArgOrdinal) {
  switch (kind) {
  case ThreadDomainKind::DenseRectangular:
    if (workItemArgOrdinal)
      return emitError()
             << "dense thread domain must not carry a work-item argument "
                "ordinal";
    return success();
  case ThreadDomainKind::DynamicWork:
    if (!workItemArgOrdinal)
      return emitError()
             << "dynamic-work thread domain requires a work-item argument "
                "ordinal";
    return success();
  }
  return emitError() << "unknown thread domain kind";
}

void DataflowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dataflow/IR/DataflowOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dataflow/IR/DataflowTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dataflow/IR/DataflowAttrs.cpp.inc"
      >();

  attachCanonicalDataflowActorInterfaces(*getContext());
}
