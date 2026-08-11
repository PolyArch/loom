#include "Fabric/IR/FabricOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <cstdint>

using namespace mlir;
using namespace fabric;

namespace {

LogicalResult verifyClosedAttributes(Operation *operation) {
  for (NamedAttribute attribute : operation->getDiscardableAttrs())
    return operation->emitOpError("has non-canonical discardable attribute '")
           << attribute.getName() << "'";
  return success();
}

LogicalResult verifyDenseOrdinal(Operation *operation, std::int64_t ordinal,
                                 std::uint64_t expected) {
  if (ordinal < 0 || static_cast<std::uint64_t>(ordinal) != expected)
    return operation->emitOpError("has non-canonical ordinal ")
           << ordinal << "; expected " << expected;
  return success();
}

LogicalResult verifyStrictSet(Operation *operation, ArrayRef<std::int64_t> set,
                              StringRef field) {
  if (llvm::any_of(set, [](std::int64_t value) { return value < 0; }))
    return operation->emitOpError("has a negative ") << field << " ordinal";
  if (!llvm::is_sorted(set) || std::adjacent_find(set.begin(), set.end()) !=
                                    set.end())
    return operation->emitOpError("requires a strictly ordered ") << field;
  return success();
}

} // namespace

LogicalResult InterconnectImplementationOp::verify() {
  if (failed(verifyClosedAttributes(getOperation())))
    return failure();
  if (!llvm::hasSingleElement(getImplementation()) ||
      !llvm::hasSingleElement(getRefinements()))
    return emitOpError("requires one block in each declarative region");
  if (!getImplementation().front().getArguments().empty() ||
      !getRefinements().front().getArguments().empty())
    return emitOpError("declarative regions must not have block arguments");

  switch (getProtocolSchema()) {
  case InterconnectProtocolSchema::Gem5EventTransportV1:
    break;
  }

  std::uint32_t lastKind = 0;
  bool first = true;
  std::uint64_t endpointCount = 0;
  std::uint64_t resourceCount = 0;
  std::uint64_t transferCount = 0;
  std::uint64_t configurationCount = 0;
  for (Operation &operation : getImplementation().front()) {
    std::uint32_t kind = 0;
    std::int64_t ordinal = 0;
    std::uint64_t *expected = nullptr;
    if (auto endpoint =
            dyn_cast<InterconnectGem5EventEndpointOp>(&operation)) {
      kind = 0;
      ordinal = endpoint.getOrdinal();
      expected = &endpointCount;
    } else if (auto resource =
                   dyn_cast<InterconnectGem5EventResourceOp>(&operation)) {
      kind = 1;
      ordinal = resource.getOrdinal();
      expected = &resourceCount;
    } else if (auto transfer =
                   dyn_cast<InterconnectGem5EventTransferOp>(&operation)) {
      kind = 2;
      ordinal = transfer.getOrdinal();
      expected = &transferCount;
    } else if (auto configuration =
                   dyn_cast<InterconnectGem5EventConfigurationFieldOp>(
                       &operation)) {
      kind = 3;
      ordinal = configuration.getOrdinal();
      expected = &configurationCount;
    } else {
      return operation.emitOpError(
          "is not in the selected interconnect protocol body catalog");
    }
    if (!first && kind < lastKind)
      return operation.emitOpError(
          "is not in canonical protocol body kind order");
    first = false;
    lastKind = kind;
    if (failed(verifyDenseOrdinal(&operation, ordinal, *expected)))
      return failure();
    ++*expected;
  }

  for (Operation &operation : getRefinements().front())
    if (!isa<InterconnectRefinementOp>(operation))
      return operation.emitOpError(
          "is not a fabric.interconnect.refinement record");
  return success();
}

LogicalResult InterconnectGem5EventEndpointOp::verify() {
  if (failed(verifyClosedAttributes(getOperation())))
    return failure();
  if (getOrdinal() < 0)
    return emitOpError("requires a nonnegative ordinal");
  switch (getDirection()) {
  case InterconnectEndpointDirection::Ingress:
  case InterconnectEndpointDirection::Egress:
    return success();
  }
  llvm_unreachable("closed endpoint direction");
}

LogicalResult InterconnectGem5EventResourceOp::verify() {
  if (failed(verifyClosedAttributes(getOperation())))
    return failure();
  return getOrdinal() < 0 ? emitOpError("requires a nonnegative ordinal")
                          : success();
}

LogicalResult InterconnectGem5EventTransferOp::verify() {
  if (failed(verifyClosedAttributes(getOperation())))
    return failure();
  if (getOrdinal() < 0 || getIngress() < 0)
    return emitOpError("requires nonnegative local references");
  if (getEgresses().empty())
    return emitOpError("requires at least one egress");
  if (failed(verifyStrictSet(getOperation(), getEgresses(), "egress set")) ||
      failed(verifyStrictSet(getOperation(), getResources(), "resource set")))
    return failure();
  return success();
}

LogicalResult InterconnectGem5EventConfigurationFieldOp::verify() {
  if (failed(verifyClosedAttributes(getOperation())))
    return failure();
  return getOrdinal() < 0 ? emitOpError("requires a nonnegative ordinal")
                          : success();
}

LogicalResult InterconnectRefinementOp::verify() {
  if (failed(verifyClosedAttributes(getOperation())))
    return failure();
  if (getArchitectureRef().empty())
    return emitOpError("requires a nonempty architecture reference");
  const ArrayRef<std::int64_t> protocolRefs = getProtocolRefs();
  if (protocolRefs.empty())
    return emitOpError("requires at least one protocol reference");
  if (llvm::any_of(protocolRefs,
                   [](std::int64_t value) { return value < 0; }))
    return emitOpError("has a negative protocol reference");

  switch (getKind()) {
  case InterconnectRefinementKind::Endpoint:
  case InterconnectRefinementKind::Configuration:
    if (protocolRefs.size() != 1)
      return emitOpError("requires exactly one protocol reference");
    return success();
  case InterconnectRefinementKind::ResourceState:
    return verifyStrictSet(getOperation(), protocolRefs,
                           "protocol resource set");
  case InterconnectRefinementKind::TransferPattern:
    return success();
  }
  llvm_unreachable("closed interconnect refinement kind");
}
