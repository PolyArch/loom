#include "ADG/Builder.h"

#include "../Fabric/Artifact/FabricModuleDomainNormalization.h"
#include "BuilderInternal.h"

#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Fabric/IR/FabricCanonicalEntity.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MemoryBuffer.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace loom::adg {
namespace detail {
llvm::Expected<mlir::ModuleOp>
loadCanonicalFabricModule(const loom::fabric::FinalizedFabricRoot &parent,
                          DesignState &state,
                          loom::fabric::FabricRootKind expectedKind) {
  auto decoded = loom::fabric::decodeFabricArtifactEnvelope(
      parent.canonicalBytes().bytes());
  if (!decoded)
    return decoded.takeError();
  if (decoded->rootKind != expectedKind)
    return detail::invalid("finalized parent has the wrong Fabric root kind");

  llvm::StringRef byteString(
      reinterpret_cast<const char *>(decoded->canonicalMlirBytecode.data()),
      decoded->canonicalMlirBytecode.size());
  llvm::MemoryBufferRef buffer(byteString, "<derived-fabric>");
  mlir::ParserConfig parserConfig(&state.context);
  mlir::Block topLevel;
  if (mlir::failed(mlir::readBytecodeFile(buffer, &topLevel, parserConfig)))
    return detail::invalid(
        "finalized parent bytecode cannot form a Builder draft");
  if (!llvm::hasSingleElement(topLevel))
    return detail::invalid("finalized parent has multiple builtin roots");
  auto module = mlir::dyn_cast<mlir::ModuleOp>(&topLevel.front());
  if (!module || mlir::failed(mlir::verify(module)))
    return detail::invalid(
        "finalized parent is not valid canonical Fabric bytecode");
  module->remove();
  state.draft = mlir::OwningOpRef<mlir::ModuleOp>(module);
  return module;
}
} // namespace detail

namespace {

using detail::invalid;

template <typename RootOp>
llvm::Expected<RootOp> singleFabricRoot(mlir::ModuleOp module) {
  if (!llvm::hasSingleElement(module.getBody()->getOperations()))
    return invalid("finalized parent does not contain exactly one Fabric root");
  auto root = mlir::dyn_cast<RootOp>(&module.getBody()->front());
  if (!root)
    return invalid("finalized parent contains the wrong Fabric root operation");
  return root;
}

bool operationHasKind(mlir::Operation *operation,
                      loom::fabric::FabricEntityKind kind) {
  using loom::fabric::FabricEntityKind;
  switch (kind) {
  case FabricEntityKind::FabricModuleTemplate:
    return mlir::isa<::fabric::ModuleOp>(operation);
  case FabricEntityKind::FabricPeOccurrence:
    return mlir::isa<::fabric::PeOp>(operation);
  case FabricEntityKind::FabricFuTemplate:
    return false;
  case FabricEntityKind::FabricFuOccurrence:
    return mlir::isa<::fabric::FuOp>(operation);
  case FabricEntityKind::FabricMemoryOccurrence:
    return mlir::isa<::fabric::MemOp>(operation);
  case FabricEntityKind::FabricSwitchOccurrence:
    return mlir::isa<::fabric::SwitchOp>(operation);
  case FabricEntityKind::FabricFifoOccurrence:
    return mlir::isa<::fabric::FifoOp>(operation);
  case FabricEntityKind::FabricBoundaryOccurrence:
    return mlir::isa<::fabric::BoundaryOp>(operation);
  case FabricEntityKind::HostCoreOccurrence:
    return mlir::isa<::fabric::SystemHostCoreOp>(operation);
  case FabricEntityKind::AccCoreOccurrence:
    return mlir::isa<::fabric::SystemAccCoreOp>(operation);
  case FabricEntityKind::SystemMemoryService:
    return mlir::isa<::fabric::SystemMemoryServiceOp>(operation);
  case FabricEntityKind::SystemServiceEndpoint:
    return mlir::isa<::fabric::SystemServiceEndpointOp>(operation);
  case FabricEntityKind::SystemServiceTransform:
    return mlir::isa<::fabric::SystemServiceTransformOp>(operation);
  case FabricEntityKind::SystemTransportResource:
    return mlir::isa<::fabric::SystemTransportResourceOp>(operation);
  case FabricEntityKind::HardwareDomain:
    return mlir::isa<::fabric::SystemHardwareDomainOp>(operation);
  case FabricEntityKind::ExternalBoundary:
    return mlir::isa<::fabric::SystemExternalBoundaryOp>(operation);
  case FabricEntityKind::FabricMemoryEngineTemplate:
    return false;
  }
  return false;
}

template <typename Ref> constexpr loom::fabric::FabricEntityKind entityKind() {
  if constexpr (std::is_same_v<Ref, loom::fabric::FabricPeOccurrenceRef>)
    return loom::fabric::FabricEntityKind::FabricPeOccurrence;
  if constexpr (std::is_same_v<Ref, loom::fabric::FabricFuOccurrenceRef>)
    return loom::fabric::FabricEntityKind::FabricFuOccurrence;
  if constexpr (std::is_same_v<Ref, loom::fabric::FabricMemoryOccurrenceRef>)
    return loom::fabric::FabricEntityKind::FabricMemoryOccurrence;
  if constexpr (std::is_same_v<Ref, loom::fabric::FabricSwitchOccurrenceRef>)
    return loom::fabric::FabricEntityKind::FabricSwitchOccurrence;
  if constexpr (std::is_same_v<Ref, loom::fabric::FabricFifoOccurrenceRef>)
    return loom::fabric::FabricEntityKind::FabricFifoOccurrence;
  if constexpr (std::is_same_v<Ref, loom::fabric::FabricBoundaryOccurrenceRef>)
    return loom::fabric::FabricEntityKind::FabricBoundaryOccurrence;
  llvm_unreachable("unsupported typed Fabric entity reference");
}

llvm::Expected<mlir::Operation *>
findModuleEntity(::fabric::ModuleOp root, loom::fabric::FabricEntityKind kind,
                 loom::fabric::FabricEntityId id) {
  mlir::Operation *result = nullptr;
  root->walk([&](mlir::Operation *operation) {
    auto attribute = operation->getAttrOfType<::fabric::EntityIdAttr>(
        ::fabric::kEntityIdAttrName);
    if (attribute && attribute.getId() == id)
      result = operation;
  });
  if (!result || !operationHasKind(result, kind))
    return invalid("typed Fabric reference does not resolve in the parent");
  return result;
}

llvm::Expected<mlir::Operation *>
occurrenceOperation(::fabric::ModuleOp root,
                    const loom::fabric::FabricModulePhysicalOwnerRef &owner) {
  return std::visit(
      [&](const auto &reference) -> llvm::Expected<mlir::Operation *> {
        using Reference = std::decay_t<decltype(reference)>;
        if constexpr (
            std::is_same_v<Reference, loom::fabric::FabricPeOccurrenceRef> ||
            std::is_same_v<Reference, loom::fabric::FabricFuOccurrenceRef> ||
            std::is_same_v<Reference,
                           loom::fabric::FabricMemoryOccurrenceRef> ||
            std::is_same_v<Reference,
                           loom::fabric::FabricSwitchOccurrenceRef> ||
            std::is_same_v<Reference, loom::fabric::FabricFifoOccurrenceRef> ||
            std::is_same_v<Reference,
                           loom::fabric::FabricBoundaryOccurrenceRef>) {
          return findModuleEntity(root, entityKind<Reference>(),
                                  reference.id());
        }
        return invalid("physical-owner selector is not an occurrence");
      },
      owner.payload());
}

template <typename Ref>
llvm::Expected<mlir::Operation *>
moduleOccurrence(::fabric::ModuleOp root,
                 const loom::fabric::FabricArtifactView &view,
                 const Ref &reference) {
  if (llvm::Error error = loom::fabric::validateFabricRef(view, reference))
    return std::move(error);
  return findModuleEntity(root, entityKind<Ref>(), reference.id());
}

llvm::Expected<mlir::Operation *> endpointOwnerOperation(
    ::fabric::ModuleOp root, const loom::fabric::FabricArtifactView &view,
    const loom::fabric::FabricTransportEndpointOwnerRef &owner) {
  if (llvm::Error error = loom::fabric::validateFabricRef(view, owner))
    return std::move(error);
  return std::visit(
      [&](const auto &reference) -> llvm::Expected<mlir::Operation *> {
        using Reference = std::decay_t<decltype(reference)>;
        if constexpr (
            std::is_same_v<Reference, loom::fabric::FabricPeOccurrenceRef> ||
            std::is_same_v<Reference, loom::fabric::FabricFuOccurrenceRef> ||
            std::is_same_v<Reference,
                           loom::fabric::FabricMemoryOccurrenceRef> ||
            std::is_same_v<Reference,
                           loom::fabric::FabricSwitchOccurrenceRef> ||
            std::is_same_v<Reference, loom::fabric::FabricFifoOccurrenceRef> ||
            std::is_same_v<Reference,
                           loom::fabric::FabricBoundaryOccurrenceRef>) {
          return findModuleEntity(root, entityKind<Reference>(),
                                  reference.id());
        }
        return invalid("transport endpoint is not local to a Module draft");
      },
      owner.payload);
}

struct ResolvedTransportEndpoint final {
  mlir::Operation *owner = nullptr;
  std::optional<mlir::Value> source;
  mlir::OpOperand *destination = nullptr;
};

llvm::Expected<ResolvedTransportEndpoint> resolveTransportEndpoint(
    ::fabric::ModuleOp root, const loom::fabric::FabricArtifactView &view,
    const loom::fabric::FabricTransportEndpointRef &endpoint) {
  if (llvm::Error error = loom::fabric::validateFabricRef(view, endpoint))
    return std::move(error);
  auto operation = endpointOwnerOperation(root, view, endpoint.owner);
  if (!operation)
    return operation.takeError();
  auto direction = view.transportEndpointDirection(endpoint);
  if (!direction)
    return invalid("transport endpoint has no direction");
  const std::uint64_t inputCount = (*operation)->getNumOperands();
  if (*direction == loom::fabric::FabricPortDirection::Input) {
    if (endpoint.ordinal >= inputCount)
      return invalid("transport input ordinal disagrees with the draft");
    return ResolvedTransportEndpoint{
        *operation, std::nullopt,
        &(*operation)->getOpOperand(static_cast<unsigned>(endpoint.ordinal))};
  }
  if (endpoint.ordinal < inputCount ||
      endpoint.ordinal - inputCount >= (*operation)->getNumResults())
    return invalid("transport output ordinal disagrees with the draft");
  return ResolvedTransportEndpoint{
      *operation,
      (*operation)
          ->getResult(static_cast<unsigned>(endpoint.ordinal - inputCount)),
      nullptr};
}

std::vector<mlir::Operation *> operationSubtree(mlir::Operation *operation) {
  std::vector<mlir::Operation *> operations;
  operation->walk(
      [&](mlir::Operation *nested) { operations.push_back(nested); });
  return operations;
}

bool referencesAnyResult(llvm::ArrayRef<mlir::Value> values,
                         mlir::Operation *operation) {
  for (mlir::Value value : values)
    if (value.getDefiningOp() == operation)
      return true;
  return false;
}

llvm::Error replaceOperationFromPrototype(detail::SpatialRootState &root,
                                          mlir::Operation *target,
                                          mlir::Operation *prototype) {
  if (target->getNumOperands() != prototype->getNumOperands() ||
      target->getNumResults() != prototype->getNumResults())
    return invalid("replacement prototype has a different port inventory");
  for (auto [targetOperand, prototypeOperand] :
       llvm::zip(target->getOperands(), prototype->getOperands()))
    if (targetOperand.getType() != prototypeOperand.getType())
      return invalid("replacement prototype has a different input type");
  for (auto [targetResult, prototypeResult] :
       llvm::zip(target->getResults(), prototype->getResults()))
    if (targetResult.getType() != prototypeResult.getType())
      return invalid("replacement prototype has a different output type");

  mlir::IRMapping mapping;
  for (auto [prototypeOperand, targetOperand] :
       llvm::zip(prototype->getOperands(), target->getOperands()))
    mapping.map(prototypeOperand, targetOperand);
  mlir::Operation *replacement = prototype->clone(mapping);
  target->getBlock()->getOperations().insert(target->getIterator(),
                                             replacement);
  if (llvm::Error error =
          root.domainRelation.replicateMappedOperations(mapping)) {
    replacement->erase();
    return error;
  }

  for (auto [oldResult, newResult] :
       llvm::zip(target->getResults(), replacement->getResults())) {
    oldResult.replaceAllUsesWith(newResult);
    for (mlir::Value &output : root.derivedOutputs)
      if (output == oldResult)
        output = newResult;
  }
  std::vector<mlir::Operation *> erased = operationSubtree(target);
  if (llvm::Error error = root.domainRelation.eraseOperations(erased)) {
    replacement->erase();
    return error;
  }
  target->erase();
  return llvm::Error::success();
}

llvm::Expected<detail::SpatialRootState *>
derivedSpatialRoot(const std::shared_ptr<detail::DesignState> &state,
                   std::size_t rootOrdinal) {
  if (rootOrdinal >= state->spatialRoots.size())
    return invalid("SpatialCore handle has an invalid owner ordinal");
  detail::SpatialRootState &root = state->spatialRoots[rootOrdinal];
  if (root.closed)
    return invalid("SpatialCore is already closed");
  if (!root.derivedParent)
    return invalid("operation requires a derived SpatialCore draft");
  return &root;
}

llvm::Error replaceConnections(
    detail::SpatialRootState &root,
    llvm::ArrayRef<loom::fabric::FabricPointConnectionPayload> connections) {
  if (connections.empty())
    return invalid("parallel connection replacement is empty");
  std::set<std::vector<std::uint8_t>> destinations;
  struct Replacement {
    mlir::OpOperand *destination;
    mlir::Value source;
  };
  std::vector<Replacement> replacements;
  replacements.reserve(connections.size());
  for (const loom::fabric::FabricPointConnectionPayload &connection :
       connections) {
    std::vector<std::uint8_t> key =
        loom::fabric::canonicalFabricBytes(connection.destination);
    if (!destinations.insert(std::move(key)).second)
      return invalid("parallel connection replacement repeats a destination");
    bool parentDestination = false;
    for (const loom::fabric::FabricPointConnectionPayload &existing :
         root.derivedParent->pointConnections())
      if (existing.destination == connection.destination) {
        parentDestination = true;
        break;
      }
    if (!parentDestination)
      return invalid("connection destination is not present in the parent");
    auto destination = resolveTransportEndpoint(
        root.operation, *root.derivedParent, connection.destination);
    if (!destination)
      return destination.takeError();
    auto source = resolveTransportEndpoint(root.operation, *root.derivedParent,
                                           connection.source);
    if (!source)
      return source.takeError();
    if (!destination->destination || !source->source)
      return invalid("point connection directions are not output-to-input");
    if (destination->destination->get().getType() != source->source->getType())
      return invalid("point connection replacement changes the port type");
    replacements.push_back({destination->destination, *source->source});
  }
  for (const Replacement &replacement : replacements)
    replacement.destination->set(replacement.source);
  return llvm::Error::success();
}

} // namespace

llvm::Expected<SpatialCoreBuilder> DesignBuilder::deriveSpatialCore(
    const loom::fabric::FinalizedFabricRoot &parent) {
  if (!state_ || state_->consumed)
    return invalid("DesignBuilder is already consumed");
  if (!state_->spatialRoots.empty() || !state_->systemRoots.empty())
    return invalid("derived Fabric draft requires an empty DesignBuilder");
  if (parent.view().rootKind() != loom::fabric::FabricRootKind::Module)
    return invalid("deriveSpatialCore requires a finalized Module parent");

  auto module = detail::loadCanonicalFabricModule(
      parent, *state_, loom::fabric::FabricRootKind::Module);
  if (!module)
    return module.takeError();
  auto root = singleFabricRoot<::fabric::ModuleOp>(*module);
  if (!root)
    return root.takeError();
  auto relation =
      loom::fabric::detail::recoverFabricModuleDomainAuthoring(*root);
  if (!relation)
    return relation.takeError();
  auto yield = mlir::dyn_cast<::fabric::YieldOp>(
      root->getBody().front().getTerminator());
  if (!yield)
    return invalid("finalized Module has no output terminator");

  std::vector<mlir::Value> outputs(yield.getValues().begin(),
                                   yield.getValues().end());
  std::vector<mlir::Type> resultTypes(root->getFunctionType().getResults());
  yield.erase();
  state_->labels.insert(root->getSymName());
  state_->spatialRoots.push_back(
      detail::SpatialRootState{*root,
                               root->getSymName().str(),
                               std::move(resultTypes),
                               {},
                               std::move(*relation),
                               false,
                               parent.view(),
                               std::move(outputs)});
  return SpatialCoreBuilder(state_, 0);
}

llvm::Error SpatialCoreBuilder::cloneOccurrence(
    const loom::fabric::FabricModulePhysicalOwnerRef &prototype) {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto root = derivedSpatialRoot(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  auto operation = occurrenceOperation((*root)->operation, prototype);
  if (!operation)
    return operation.takeError();

  mlir::IRMapping mapping;
  mlir::Operation *clone = (*operation)->clone(mapping);
  (*operation)
      ->getBlock()
      ->getOperations()
      .insert(std::next((*operation)->getIterator()), clone);
  if (llvm::Error error =
          (*root)->domainRelation.replicateMappedOperations(mapping)) {
    clone->erase();
    return error;
  }
  return llvm::Error::success();
}

llvm::Error SpatialCoreBuilder::eraseOccurrence(
    const loom::fabric::FabricModulePhysicalOwnerRef &target) {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto root = derivedSpatialRoot(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  auto operation = occurrenceOperation((*root)->operation, target);
  if (!operation)
    return operation.takeError();
  for (mlir::Value result : (*operation)->getResults())
    if (!result.use_empty())
      return invalid("removed occurrence still has a transport consumer");
  if (referencesAnyResult((*root)->derivedOutputs, *operation))
    return invalid("removed occurrence still supplies a Module output");
  if (auto fu = mlir::dyn_cast<::fabric::FuOp>(*operation)) {
    std::size_t siblings = 0;
    for (mlir::Operation &candidate : fu->getBlock()->getOperations())
      siblings += mlir::isa<::fabric::FuOp>(candidate);
    if (siblings <= 1)
      return invalid("PE must retain at least one FU occurrence");
  }
  std::vector<mlir::Operation *> erased = operationSubtree(*operation);
  if (llvm::Error error = (*root)->domainRelation.eraseOperations(erased))
    return error;
  (*operation)->erase();
  return llvm::Error::success();
}

llvm::Error SpatialCoreBuilder::replacePointConnection(
    const loom::fabric::FabricTransportEndpointRef &destination,
    const loom::fabric::FabricTransportEndpointRef &source) {
  return replaceParallelConnections(
      {loom::fabric::FabricPointConnectionPayload{source, destination}});
}

llvm::Error SpatialCoreBuilder::replaceParallelConnections(
    llvm::ArrayRef<loom::fabric::FabricPointConnectionPayload> connections) {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto root = derivedSpatialRoot(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  return replaceConnections(**root, connections);
}

llvm::Error SpatialCoreBuilder::changeBoundaryInventory(
    std::size_t inputCount,
    llvm::ArrayRef<loom::fabric::FabricTransportEndpointRef> outputSources) {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto root = derivedSpatialRoot(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  mlir::Block &body = (*root)->operation.getBody().front();
  const std::size_t oldInputCount = body.getNumArguments();
  const std::size_t oldOutputCount = (*root)->resultTypes.size();
  if (inputCount > oldInputCount || outputSources.size() > oldOutputCount)
    return invalid("derived boundary growth requires explicit domain rows");

  llvm::SmallVector<mlir::Type, 8> inputTypes;
  inputTypes.reserve(inputCount);
  for (std::size_t ordinal = 0; ordinal < inputCount; ++ordinal)
    inputTypes.push_back(body.getArgument(ordinal).getType());
  for (std::size_t ordinal = oldInputCount; ordinal > inputCount; --ordinal) {
    mlir::BlockArgument argument = body.getArgument(ordinal - 1);
    if (!argument.use_empty())
      return invalid("removed Module input still has a consumer");
    body.eraseArgument(static_cast<unsigned>(ordinal - 1));
  }

  std::vector<mlir::Value> outputs;
  llvm::SmallVector<mlir::Type, 8> outputTypes;
  outputs.reserve(outputSources.size());
  outputTypes.reserve(outputSources.size());
  for (const auto &reference : outputSources) {
    auto endpoint = resolveTransportEndpoint(
        (*root)->operation, *(*root)->derivedParent, reference);
    if (!endpoint)
      return endpoint.takeError();
    if (!endpoint->source)
      return invalid("Module output source is not a transport output");
    if (!endpoint->source->use_empty())
      return invalid("Module output source already has a consumer");
    outputs.push_back(*endpoint->source);
    outputTypes.push_back(endpoint->source->getType());
  }

  if (llvm::Error error = (*root)->domainRelation.truncateBoundaryMembers(
          loom::fabric::FabricPortDirection::Input, oldInputCount, inputCount))
    return error;
  if (llvm::Error error = (*root)->domainRelation.truncateBoundaryMembers(
          loom::fabric::FabricPortDirection::Output, oldOutputCount,
          outputSources.size()))
    return error;
  (*root)->operation.setFunctionType(
      mlir::FunctionType::get(&(*state)->context, inputTypes, outputTypes));
  (*root)->resultTypes.assign(outputTypes.begin(), outputTypes.end());
  (*root)->derivedOutputs = std::move(outputs);
  return llvm::Error::success();
}

llvm::Error SpatialCoreBuilder::replacePeKind(
    loom::fabric::FabricPeOccurrenceRef target,
    loom::fabric::FabricPeOccurrenceRef prototype) {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto root = derivedSpatialRoot(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  auto targetOp =
      moduleOccurrence((*root)->operation, *(*root)->derivedParent, target);
  if (!targetOp)
    return targetOp.takeError();
  auto prototypeOp =
      moduleOccurrence((*root)->operation, *(*root)->derivedParent, prototype);
  if (!prototypeOp)
    return prototypeOp.takeError();
  return replaceOperationFromPrototype(**root, *targetOp, *prototypeOp);
}

llvm::Error SpatialCoreBuilder::resizeInstructionStore(
    loom::fabric::FabricPeOccurrenceRef target,
    std::uint32_t instructionCapacity) {
  if (instructionCapacity == 0 ||
      instructionCapacity >
          static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))
    return invalid("instruction capacity must fit positive i32");
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto root = derivedSpatialRoot(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  auto operation =
      moduleOccurrence((*root)->operation, *(*root)->derivedParent, target);
  if (!operation)
    return operation.takeError();
  auto pe = mlir::cast<::fabric::PeOp>(*operation);
  if (pe.getSchedule() != ::fabric::Schedule::Temporal ||
      !pe.getNumInstruction())
    return invalid("instruction-store resize requires a temporal PE");
  const std::uint32_t oldCapacity = *pe.getNumInstruction();
  if (oldCapacity == instructionCapacity)
    return invalid("instruction-store resize is a no-op");
  if (llvm::Error error = (*root)->domainRelation.resizeInternalMembers(
          pe.getOperation(),
          ::fabric::ModuleDomainAuthoringRelation::InternalMemberRole::
              InstructionContext,
          oldCapacity, instructionCapacity))
    return error;
  pe.setNumInstruction(instructionCapacity);
  return llvm::Error::success();
}

llvm::Error SpatialCoreBuilder::changeTemporalOperandBufferMode(
    loom::fabric::FabricPeOccurrenceRef target,
    ::fabric::OperandBufferMode mode) {
  if (!::fabric::symbolizeOperandBufferMode(static_cast<std::uint32_t>(mode)))
    return invalid("operand-buffer mode is outside the closed Fabric domain");
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto root = derivedSpatialRoot(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  auto operation =
      moduleOccurrence((*root)->operation, *(*root)->derivedParent, target);
  if (!operation)
    return operation.takeError();
  auto pe = mlir::cast<::fabric::PeOp>(*operation);
  const auto current = pe.getOperandBufferMode();
  if (pe.getSchedule() != ::fabric::Schedule::Temporal || !current)
    return invalid("operand-buffer mode change requires a temporal PE");
  if (*current == mode)
    return invalid("operand-buffer mode change is a no-op");
  pe.setOperandBufferMode(mode);
  return llvm::Error::success();
}

llvm::Error SpatialCoreBuilder::resizeTemporalOperandBuffer(
    loom::fabric::FabricPeOccurrenceRef target,
    std::uint32_t entriesPerAllocationUnit) {
  if (entriesPerAllocationUnit == 0 ||
      entriesPerAllocationUnit >
          static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))
    return invalid("operand-buffer entries must fit positive i32");
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto root = derivedSpatialRoot(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  auto operation =
      moduleOccurrence((*root)->operation, *(*root)->derivedParent, target);
  if (!operation)
    return operation.takeError();
  auto pe = mlir::cast<::fabric::PeOp>(*operation);
  const auto current = pe.getOperandBufferSize();
  if (pe.getSchedule() != ::fabric::Schedule::Temporal || !current)
    return invalid("operand-buffer resize requires a temporal PE");
  if (*current == entriesPerAllocationUnit)
    return invalid("operand-buffer resize is a no-op");
  pe.setOperandBufferSize(entriesPerAllocationUnit);
  return llvm::Error::success();
}

llvm::Error SpatialCoreBuilder::replaceFuInventory(
    loom::fabric::FabricPeOccurrenceRef target,
    llvm::ArrayRef<loom::fabric::FabricFuOccurrenceRef> prototypes) {
  if (prototypes.empty())
    return invalid("FU inventory replacement is empty");
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto root = derivedSpatialRoot(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  auto targetOperation =
      moduleOccurrence((*root)->operation, *(*root)->derivedParent, target);
  if (!targetOperation)
    return targetOperation.takeError();
  auto targetPe = mlir::cast<::fabric::PeOp>(*targetOperation);

  std::vector<::fabric::FuOp> prototypeOps;
  std::vector<::fabric::PeOp> prototypePes;
  prototypeOps.reserve(prototypes.size());
  prototypePes.reserve(prototypes.size());
  for (const auto &prototype : prototypes) {
    auto operation = moduleOccurrence((*root)->operation,
                                      *(*root)->derivedParent, prototype);
    if (!operation)
      return operation.takeError();
    auto fu = mlir::cast<::fabric::FuOp>(*operation);
    auto prototypePe = mlir::dyn_cast<::fabric::PeOp>(fu->getParentOp());
    if (!prototypePe)
      return invalid("FU inventory prototype has no parent PE");
    mlir::Block &targetInputs = targetPe.getBody().front();
    mlir::Block &prototypeInputs = prototypePe.getBody().front();
    if (targetInputs.getNumArguments() != prototypeInputs.getNumArguments())
      return invalid("FU inventory prototype PE has a different input count");
    for (auto [targetInput, prototypeInput] :
         llvm::zip(targetInputs.getArguments(), prototypeInputs.getArguments()))
      if (targetInput.getType() != prototypeInput.getType())
        return invalid("FU inventory prototype PE has a different input type");
    prototypeOps.push_back(fu);
    prototypePes.push_back(prototypePe);
  }

  std::vector<mlir::Operation *> oldFus;
  for (mlir::Operation &operation : targetPe.getBody().front())
    if (mlir::isa<::fabric::FuOp>(operation))
      oldFus.push_back(&operation);
  if (oldFus.empty())
    return invalid("target PE has no FU inventory");

  mlir::Operation *insertion = oldFus.front();
  for (auto [prototype, prototypePe] : llvm::zip(prototypeOps, prototypePes)) {
    mlir::IRMapping mapping;
    for (auto [prototypeInput, targetInput] :
         llvm::zip(prototypePe.getBody().front().getArguments(),
                   targetPe.getBody().front().getArguments()))
      mapping.map(prototypeInput, targetInput);
    mlir::Operation *clone = prototype->clone(mapping);
    insertion->getBlock()->getOperations().insert(insertion->getIterator(),
                                                  clone);
    if (llvm::Error error =
            (*root)->domainRelation.replicateMappedOperations(mapping))
      return error;
  }
  std::vector<mlir::Operation *> erased;
  for (mlir::Operation *fu : oldFus) {
    std::vector<mlir::Operation *> subtree = operationSubtree(fu);
    erased.insert(erased.end(), subtree.begin(), subtree.end());
  }
  if (llvm::Error error = (*root)->domainRelation.eraseOperations(erased))
    return error;
  for (mlir::Operation *fu : llvm::reverse(oldFus))
    fu->erase();
  return llvm::Error::success();
}

llvm::Error SpatialCoreBuilder::replaceFuCapability(
    loom::fabric::FabricFuOccurrenceRef target,
    loom::fabric::FabricFuOccurrenceRef prototype) {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto root = derivedSpatialRoot(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  auto targetOp =
      moduleOccurrence((*root)->operation, *(*root)->derivedParent, target);
  if (!targetOp)
    return targetOp.takeError();
  auto prototypeOp =
      moduleOccurrence((*root)->operation, *(*root)->derivedParent, prototype);
  if (!prototypeOp)
    return prototypeOp.takeError();
  return replaceOperationFromPrototype(**root, *targetOp, *prototypeOp);
}

llvm::Error SpatialCoreBuilder::replaceSwitchModeOrScheduleCapacity(
    loom::fabric::FabricSwitchOccurrenceRef target,
    loom::fabric::FabricSwitchOccurrenceRef prototype) {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto root = derivedSpatialRoot(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  auto targetOperation =
      moduleOccurrence((*root)->operation, *(*root)->derivedParent, target);
  if (!targetOperation)
    return targetOperation.takeError();
  auto prototypeOperation =
      moduleOccurrence((*root)->operation, *(*root)->derivedParent, prototype);
  if (!prototypeOperation)
    return prototypeOperation.takeError();
  auto targetSwitch = mlir::cast<::fabric::SwitchOp>(*targetOperation);
  auto prototypeSwitch = mlir::cast<::fabric::SwitchOp>(*prototypeOperation);
  if (targetSwitch.getNumOperands() != prototypeSwitch.getNumOperands() ||
      targetSwitch.getNumResults() != prototypeSwitch.getNumResults())
    return invalid("switch prototype has a different port inventory");
  targetSwitch.setSchedule(prototypeSwitch.getSchedule());
  if (auto parameters = prototypeSwitch.getHwParamsAttr())
    targetSwitch.setHwParamsAttr(parameters);
  else
    targetSwitch.removeHwParamsAttr();
  return llvm::Error::success();
}

llvm::Error SpatialCoreBuilder::resizeSwitchRouteTable(
    loom::fabric::FabricSwitchOccurrenceRef target, std::uint32_t entries) {
  if (entries == 0)
    return invalid("switch route-table capacity must be positive");
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto root = derivedSpatialRoot(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  auto operation =
      moduleOccurrence((*root)->operation, *(*root)->derivedParent, target);
  if (!operation)
    return operation.takeError();
  auto sw = mlir::cast<::fabric::SwitchOp>(*operation);
  if (sw.getSchedule() != ::fabric::Schedule::Temporal)
    return invalid("route-table resize requires a Temporal switch");
  if (auto configuration = sw.getSwConfigsAttr();
      configuration && configuration.get("route_table"))
    return invalid("cannot resize a programmed switch route table");
  auto parameters = sw.getHwParamsAttr();
  auto dictionary = parameters && parameters.size() == 1
                        ? mlir::dyn_cast<mlir::DictionaryAttr>(parameters[0])
                        : mlir::DictionaryAttr();
  auto current = dictionary ? mlir::dyn_cast_or_null<mlir::IntegerAttr>(
                                  dictionary.get("route_table_size"))
                            : mlir::IntegerAttr();
  if (!current || current.getInt() <= 0)
    return invalid("Temporal switch has no route-table capacity");
  if (static_cast<std::uint64_t>(current.getInt()) == entries)
    return invalid("switch route-table resize is a no-op");
  mlir::NamedAttrList updated(dictionary.getValue());
  updated.set("route_table_size",
              mlir::IntegerAttr::get(
                  mlir::IntegerType::get(sw.getContext(), 32), entries));
  sw.setHwParamsAttr(mlir::ArrayAttr::get(
      sw.getContext(), {updated.getDictionary(sw.getContext())}));
  return llvm::Error::success();
}

llvm::Error
SpatialCoreBuilder::resizeMemory(loom::fabric::FabricMemoryOccurrenceRef target,
                                 std::uint64_t capacityBytes) {
  if (capacityBytes == 0)
    return invalid("memory capacity must be positive");
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto root = derivedSpatialRoot(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  auto operation =
      moduleOccurrence((*root)->operation, *(*root)->derivedParent, target);
  if (!operation)
    return operation.takeError();
  auto memory = mlir::cast<::fabric::MemOp>(*operation);
  ::fabric::MemoryContractAttr contract = memory.getMemoryContract();
  ::fabric::LocalMemoryServiceAttr local = contract.getLocalService();
  if (!local)
    return invalid("memory resize requires a Local Memory Service");
  if (local.getCapacityBytes() == capacityBytes)
    return invalid("memory resize is a no-op");
  auto resized = ::fabric::LocalMemoryServiceAttr::get(
      &(*state)->context, capacityBytes, local.getServiceContract());
  memory.setMemoryContractAttr(::fabric::MemoryContractAttr::get(
      &(*state)->context, contract.getEngine(), resized,
      contract.getConnectivity(), contract.getManagerEndpoints(),
      contract.getSubordinateEndpoints()));
  return llvm::Error::success();
}

llvm::Error SpatialCoreBuilder::replaceMemoryOperationTable(
    loom::fabric::FabricMemoryOccurrenceRef target,
    loom::fabric::FabricMemoryOccurrenceRef prototype) {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto root = derivedSpatialRoot(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  auto targetOperation =
      moduleOccurrence((*root)->operation, *(*root)->derivedParent, target);
  if (!targetOperation)
    return targetOperation.takeError();
  auto prototypeOperation =
      moduleOccurrence((*root)->operation, *(*root)->derivedParent, prototype);
  if (!prototypeOperation)
    return prototypeOperation.takeError();
  auto targetMemory = mlir::cast<::fabric::MemOp>(*targetOperation);
  auto prototypeMemory = mlir::cast<::fabric::MemOp>(*prototypeOperation);
  if (targetMemory.getNumOperands() != prototypeMemory.getNumOperands() ||
      targetMemory.getNumResults() != prototypeMemory.getNumResults())
    return invalid("memory table prototype has a different port inventory");
  ::fabric::MemoryContractAttr targetContract =
      targetMemory.getMemoryContract();
  ::fabric::MemoryContractAttr prototypeContract =
      prototypeMemory.getMemoryContract();
  targetMemory.setMemoryContractAttr(::fabric::MemoryContractAttr::get(
      &(*state)->context, prototypeContract.getEngine(),
      targetContract.getLocalService(), prototypeContract.getConnectivity(),
      targetContract.getManagerEndpoints(),
      targetContract.getSubordinateEndpoints()));
  if (auto ports = prototypeMemory.getMemoryOperationPortsAttr())
    targetMemory.setMemoryOperationPortsAttr(ports);
  else
    targetMemory.removeMemoryOperationPortsAttr();
  if (auto parameters = prototypeMemory.getHwParamsAttr())
    targetMemory.setHwParamsAttr(parameters);
  else
    targetMemory.removeHwParamsAttr();
  return llvm::Error::success();
}

llvm::Error
SpatialCoreBuilder::resizeFifo(loom::fabric::FabricFifoOccurrenceRef target,
                               std::uint32_t depth) {
  if (depth == 0 || depth > static_cast<std::uint32_t>(
                                std::numeric_limits<std::int32_t>::max()))
    return invalid("FIFO depth must fit positive i32");
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto root = derivedSpatialRoot(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  auto operation =
      moduleOccurrence((*root)->operation, *(*root)->derivedParent, target);
  if (!operation)
    return operation.takeError();
  auto fifo = mlir::cast<::fabric::FifoOp>(*operation);
  if (fifo.getMaxDepth() == depth)
    return invalid("FIFO resize is a no-op");
  fifo.setMaxDepth(depth);
  return llvm::Error::success();
}

llvm::Error SpatialCoreBuilder::changeFifoBypassCapability(
    loom::fabric::FabricFifoOccurrenceRef target, bool bypassable) {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto root = derivedSpatialRoot(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  auto operation =
      moduleOccurrence((*root)->operation, *(*root)->derivedParent, target);
  if (!operation)
    return operation.takeError();
  auto fifo = mlir::cast<::fabric::FifoOp>(*operation);
  if (fifo.getBypassable() == bypassable)
    return invalid("FIFO bypass-capability change is a no-op");
  fifo.setBypassable(bypassable);
  if (!bypassable)
    fifo.setBypassed(std::nullopt);
  return llvm::Error::success();
}

llvm::Error SpatialCoreBuilder::closeDerived() {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto root = derivedSpatialRoot(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  std::vector<SpatialValue> outputs;
  outputs.reserve((*root)->derivedOutputs.size());
  for (mlir::Value output : (*root)->derivedOutputs)
    outputs.push_back(SpatialValue(*state, rootOrdinal_, output));
  return close(outputs);
}

} // namespace loom::adg
