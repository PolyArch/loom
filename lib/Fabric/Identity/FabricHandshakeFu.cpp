#include "FabricHandshakeInternal.h"

#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/Identity/FabricFuCapabilityTemplate.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <set>
#include <utility>
#include <vector>

namespace loom::fabric {
namespace {

using Arc = std::pair<std::uint32_t, std::uint32_t>;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_handshake_invalid: " + message);
}

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  for (unsigned shift = 0; shift != 32; shift += 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> (24 - shift)));
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 0; shift != 64; shift += 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> (56 - shift)));
}

std::vector<std::uint8_t>
operationKey(const FabricFuOccurrenceNodeRef &operation) {
  return canonicalFabricBytes(operation);
}

std::vector<std::uint8_t>
operationJunctionKey(const FabricFuOccurrenceNodeRef &operation,
                     ::dataflow::OperationSchemaId schema,
                     std::uint32_t caseOrdinal, std::uint8_t family,
                     std::uint64_t position) {
  std::vector<std::uint8_t> key = operationKey(operation);
  auto schemaBytes = ::dataflow::encodeOperationSchemaId(schema);
  if (schemaBytes) {
    const auto bytes = schemaBytes->bytes();
    key.insert(key.end(), bytes.begin(), bytes.end());
  }
  key.push_back(family);
  appendU32(key, caseOrdinal);
  appendU64(key, position);
  return key;
}

detail::HandshakeFragmentSelector
fuCapabilitySelector(FabricFuOccurrenceRef occurrence,
                     FabricFuCapabilityTemplateRef capability) {
  detail::HandshakeFragmentSelector selector;
  selector.kind = detail::HandshakeFragmentSelectorKind::FuCapability;
  selector.fuOccurrence = occurrence;
  selector.fuCapability = capability;
  selector.exclusiveGroup = 0;
  return selector;
}

detail::HandshakeFragmentSelector fuOperationSelector(
    detail::HandshakeFragmentSelectorKind kind,
    FabricFuOccurrenceRef occurrence, FabricFuCapabilityTemplateRef capability,
    FabricFuOccurrenceNodeRef operation, ::dataflow::OperationSchemaId schema,
    std::uint32_t caseOrdinal, std::uint64_t physicalPortOrdinal = 0) {
  detail::HandshakeFragmentSelector selector;
  selector.kind = kind;
  selector.fuOccurrence = occurrence;
  selector.fuCapability = capability;
  selector.fuOperation = detail::HandshakeFuOperationSelector{
      operation, schema, caseOrdinal, physicalPortOrdinal};
  return selector;
}

llvm::Expected<std::uint32_t>
terminalNode(detail::HandshakeOwnerModelBuilder &builder,
             const FabricArtifactView &view, FabricFuOccurrenceRef occurrence,
             const FabricFuCapabilityTemplateEndpointRef &endpoint,
             HandshakeSignalKind signal) {
  if (const auto *boundary =
          std::get_if<FabricFuTemplatePortRef>(&endpoint.payload)) {
    auto physical = view.fuOccurrenceTransportEndpoint(
        {occurrence, boundary->direction, boundary->ordinal});
    if (!physical)
      return invalid("FU capability boundary has no occurrence endpoint");
    return builder.boundarySignal({*physical, signal});
  }
  const auto &port = std::get<FabricFuNodePortRef>(endpoint.payload);
  auto node = deriveFabricFuOccurrenceNode(view, port.node, occurrence);
  if (!node)
    return node.takeError();
  std::vector<std::uint8_t> key = operationKey(*node);
  key.push_back(static_cast<std::uint8_t>(port.direction));
  appendU64(key, port.ordinal);
  key.push_back(static_cast<std::uint8_t>(signal));
  return builder.junction(key);
}

llvm::Expected<std::uint32_t> operationPortNode(
    detail::HandshakeOwnerModelBuilder &builder, const FabricArtifactView &view,
    FabricFuOccurrenceRef occurrence, FabricFuTemplateNodeRef operation,
    FabricPortDirection direction, FabricOrdinal ordinal,
    HandshakeSignalKind signal) {
  return terminalNode(builder, view, occurrence,
                      FabricFuCapabilityTemplateEndpointRef::nodePort(
                          FabricFuNodePortRef{operation, direction, ordinal}),
                      signal);
}

llvm::Expected<bool>
isRegisteredCase(const ::fabric::ResourceContract &contract,
                 std::uint32_t caseOrdinal) {
  auto patternKey = ::fabric::resolveOperationUsePattern(contract, caseOrdinal);
  if (!patternKey)
    return patternKey.takeError();
  const ::fabric::UsePattern pattern = contract.usePattern(*patternKey);
  const auto eventOrder = contract.eventOrder(pattern.timingAndProgress);
  if (pattern.acquire.ordinal() >= eventOrder.size() ||
      pattern.release.ordinal() >= eventOrder.size())
    return invalid("fabric.op use pattern has an invalid event order");
  return eventOrder[pattern.acquire.ordinal()] !=
         eventOrder[pattern.release.ordinal()];
}

llvm::Error compileOperationCase(detail::HandshakeOwnerModelBuilder &builder,
                                 const FabricArtifactView &view,
                                 FabricFuOccurrenceRef occurrence,
                                 FabricFuCapabilityTemplateRef capabilityRef,
                                 FabricFuTemplateNodeRef operation,
                                 ::dataflow::OperationSchemaId schema,
                                 std::uint32_t caseOrdinal,
                                 std::uint32_t inputCount,
                                 std::uint32_t resultCount, bool registered) {
  auto occurrenceOperation =
      deriveFabricFuOccurrenceNode(view, operation, occurrence);
  if (!occurrenceOperation)
    return occurrenceOperation.takeError();

  std::vector<std::uint32_t> inputValid(inputCount);
  std::vector<std::uint32_t> inputReady(inputCount);
  std::vector<std::uint32_t> resultValid(resultCount);
  std::vector<std::uint32_t> resultReady(resultCount);
  for (std::uint32_t ordinal = 0; ordinal < inputCount; ++ordinal) {
    auto valid = operationPortNode(builder, view, occurrence, operation,
                                   FabricPortDirection::Input, ordinal,
                                   HandshakeSignalKind::Valid);
    auto ready = operationPortNode(builder, view, occurrence, operation,
                                   FabricPortDirection::Input, ordinal,
                                   HandshakeSignalKind::Ready);
    if (!valid)
      return valid.takeError();
    if (!ready)
      return ready.takeError();
    inputValid[ordinal] = *valid;
    inputReady[ordinal] = *ready;
  }
  for (std::uint32_t ordinal = 0; ordinal < resultCount; ++ordinal) {
    auto valid = operationPortNode(builder, view, occurrence, operation,
                                   FabricPortDirection::Output, ordinal,
                                   HandshakeSignalKind::Valid);
    auto ready = operationPortNode(builder, view, occurrence, operation,
                                   FabricPortDirection::Output, ordinal,
                                   HandshakeSignalKind::Ready);
    if (!valid)
      return valid.takeError();
    if (!ready)
      return ready.takeError();
    resultValid[ordinal] = *valid;
    resultReady[ordinal] = *ready;
  }

  std::vector<std::uint32_t> inputPrefix(inputCount + 1);
  std::vector<std::uint32_t> inputSuffix(inputCount + 1);
  std::vector<std::uint32_t> resultPrefix(resultCount + 1);
  std::vector<std::uint32_t> resultSuffix(resultCount + 1);
  for (std::uint32_t position = 0; position <= inputCount; ++position) {
    inputPrefix[position] = builder.junction(operationJunctionKey(
        *occurrenceOperation, schema, caseOrdinal, 0, position));
    inputSuffix[position] = builder.junction(operationJunctionKey(
        *occurrenceOperation, schema, caseOrdinal, 1, position));
  }
  for (std::uint32_t position = 0; position <= resultCount; ++position) {
    resultPrefix[position] = builder.junction(operationJunctionKey(
        *occurrenceOperation, schema, caseOrdinal, 2, position));
    resultSuffix[position] = builder.junction(operationJunctionKey(
        *occurrenceOperation, schema, caseOrdinal, 3, position));
  }

  std::vector<Arc> base;
  base.reserve(2 * (inputCount + resultCount));
  for (std::uint32_t position = 0; position < inputCount; ++position) {
    base.emplace_back(inputPrefix[position], inputPrefix[position + 1]);
    base.emplace_back(inputSuffix[position + 1], inputSuffix[position]);
  }
  for (std::uint32_t position = 0; position < resultCount; ++position) {
    base.emplace_back(resultPrefix[position], resultPrefix[position + 1]);
    if (!registered)
      base.emplace_back(resultSuffix[position + 1], resultSuffix[position]);
  }
  builder.addFragment(
      fuOperationSelector(
          detail::HandshakeFragmentSelectorKind::FuOperationCase, occurrence,
          capabilityRef, *occurrenceOperation, schema, caseOrdinal),
      std::move(base));

  for (std::uint32_t ordinal = 0; ordinal < inputCount; ++ordinal) {
    std::vector<Arc> arcs{{inputValid[ordinal], inputPrefix[ordinal + 1]},
                          {inputValid[ordinal], inputSuffix[ordinal]},
                          {inputPrefix[ordinal], inputReady[ordinal]},
                          {inputSuffix[ordinal + 1], inputReady[ordinal]}};
    if (resultCount != 0)
      arcs.emplace_back(resultPrefix.back(), inputReady[ordinal]);
    builder.addFragment(
        fuOperationSelector(
            detail::HandshakeFragmentSelectorKind::FuOperationInputActive,
            occurrence, capabilityRef, *occurrenceOperation, schema,
            caseOrdinal, ordinal),
        std::move(arcs));
  }

  for (std::uint32_t ordinal = 0; ordinal < resultCount; ++ordinal) {
    std::vector<Arc> arcs{
        {resultReady[ordinal], resultPrefix[ordinal + 1]}};
    if (!registered) {
      arcs.emplace_back(resultReady[ordinal], resultSuffix[ordinal]);
      arcs.emplace_back(resultPrefix[ordinal], resultValid[ordinal]);
      arcs.emplace_back(resultSuffix[ordinal + 1], resultValid[ordinal]);
    }
    if (!registered && inputCount != 0)
      arcs.emplace_back(inputPrefix.back(), resultValid[ordinal]);
    builder.addFragment(
        fuOperationSelector(
            detail::HandshakeFragmentSelectorKind::FuOperationResultActive,
            occurrence, capabilityRef, *occurrenceOperation, schema,
            caseOrdinal, ordinal),
        std::move(arcs));
  }
  return llvm::Error::success();
}

llvm::Error compileOperation(detail::HandshakeOwnerModelBuilder &builder,
                             const FabricArtifactView &view,
                             FabricFuOccurrenceRef occurrence,
                             FabricFuCapabilityTemplateRef capabilityRef,
                             FabricFuTemplateNodeRef operation) {
  const ResolvedFabricOpCapabilityView *capability =
      view.resolvedFabricOpCapability(operation);
  if (!capability)
    return invalid("FU capability row contains an unresolved fabric.op");

  std::uint32_t inputCount = 0;
  std::uint32_t resultCount = 0;
  for (const ResolvedFabricOpPhysicalPortView &port :
       capability->physicalPorts) {
    if (port.reference.direction == FabricPortDirection::Input) {
      if (port.reference.ordinal != inputCount++)
        return invalid("fabric.op input-port inventory is not dense");
    } else if (port.reference.ordinal != resultCount++) {
      return invalid("fabric.op result-port inventory is not dense");
    }
  }

  std::vector<::dataflow::OperationSchemaId> schemas =
      capability->enabledOperationSchemas;
  llvm::sort(schemas, [](auto lhs, auto rhs) {
    return ::dataflow::operationSchemaSpelling(lhs) <
           ::dataflow::operationSchemaSpelling(rhs);
  });
  for (::dataflow::OperationSchemaId schema : schemas) {
    auto cases = ::dataflow::semantics::projectActorHandshakeCases(
        schema, inputCount, resultCount);
    if (!cases)
      return cases.takeError();
    for (const ::dataflow::semantics::ActorHandshakeCase &transition : *cases) {
      auto registered = isRegisteredCase(
          capability->resourceStateAndTimingContract, transition.ordinal);
      if (!registered)
        return registered.takeError();
      if (llvm::Error error = compileOperationCase(
              builder, view, occurrence, capabilityRef, operation, schema,
              transition.ordinal, inputCount, resultCount, *registered))
        return error;
    }
  }
  return llvm::Error::success();
}

} // namespace

llvm::Expected<FabricFuHandshakeSelection> makeFuHandshakeSelection(
    const FabricArtifactView &view, FabricFuOccurrenceRef occurrence,
    FabricFuCapabilityTemplateRef capability,
    llvm::ArrayRef<FabricFuOperationHandshakeBinding> operations) {
  if (llvm::Error error = validateFabricRef(view, occurrence))
    return std::move(error);
  const auto definition = view.fuTemplateOf(occurrence);
  if (!definition || capability.fu != *definition)
    return invalid("FU capability row has the wrong occurrence definition");
  const auto inventory = view.fuCapabilityTemplates(*definition);
  if (llvm::Error error =
          validateFabricFuCapabilityTemplateRef(inventory, capability))
    return std::move(error);
  const FabricFuCapabilityTemplateRecord &row = inventory[capability.ordinal];

  std::vector<FabricFuTemplateNodeRef> activeOperations;
  for (const FabricFuTemplateNodeRef &node : row.activeNodes)
    if (node.node == FabricFuNodeKind::Op)
      activeOperations.push_back(node);
  if (operations.size() != activeOperations.size())
    return invalid("FU actor correspondence does not cover every active op");

  std::set<std::vector<std::uint8_t>> boundOperations;
  std::vector<FabricFuOperationHandshakeSelection> selected;
  selected.reserve(operations.size());
  for (const FabricFuOperationHandshakeBinding &binding : operations) {
    if (binding.operation.node != FabricFuNodeKind::Op ||
        binding.operation.fu != *definition ||
        !llvm::is_contained(activeOperations, binding.operation))
      return invalid("FU actor correspondence names an inactive fabric.op");
    if (!boundOperations.insert(canonicalFabricBytes(binding.operation)).second)
      return invalid("FU actor correspondence repeats one fabric.op");
    const ResolvedFabricOpCapabilityView *operationCapability =
        view.resolvedFabricOpCapability(binding.operation);
    if (!operationCapability)
      return invalid("FU actor correspondence names an unresolved fabric.op");
    if (llvm::Error error = operationCapability->admitCorrespondence(
            binding.actor, binding.indexBitWidth, binding.operandPorts,
            binding.resultPorts,
            binding.pointerLayout ? &*binding.pointerLayout : nullptr))
      return std::move(error);
    auto cases = ::dataflow::semantics::projectActorHandshakeCases(
        binding.actor.schema,
        static_cast<std::uint32_t>(binding.operandPorts.size()),
        static_cast<std::uint32_t>(binding.resultPorts.size()));
    if (!cases)
      return cases.takeError();
    auto occurrenceOperation =
        deriveFabricFuOccurrenceNode(view, binding.operation, occurrence);
    if (!occurrenceOperation)
      return occurrenceOperation.takeError();
    selected.push_back(FabricFuOperationHandshakeSelection{
        *occurrenceOperation, binding.actor.schema, binding.operandPorts,
        binding.resultPorts});
  }
  llvm::sort(selected, [](const auto &lhs, const auto &rhs) {
    return canonicalFabricBytes(lhs.operation) <
           canonicalFabricBytes(rhs.operation);
  });
  return FabricFuHandshakeSelection(occurrence, capability,
                                    std::move(selected));
}

namespace detail {

llvm::Expected<HandshakeOwnerModel>
compileFuHandshakeModel(const FabricArtifactView &view,
                        FabricFuOccurrenceRef occurrence) {
  const auto definition = view.fuTemplateOf(occurrence);
  if (!definition)
    return invalid("FU occurrence has no definition");
  HandshakeOwnerModelBuilder builder(FabricHandshakeOwner::fu(occurrence));
  for (auto [ordinal, record] :
       llvm::enumerate(view.fuCapabilityTemplates(*definition))) {
    const FabricFuCapabilityTemplateRef capabilityRef{
        *definition, static_cast<FabricOrdinal>(ordinal)};
    auto terminalEdges = projectFabricFuCapabilityTemplateTerminalEdges(record);
    if (!terminalEdges)
      return terminalEdges.takeError();
    std::vector<Arc> arcs;
    arcs.reserve(terminalEdges->size() * 2);
    for (const FabricFuCapabilityTemplateEdge &edge : *terminalEdges) {
      auto sourceValid = terminalNode(builder, view, occurrence, edge.source,
                                      HandshakeSignalKind::Valid);
      auto destinationValid =
          terminalNode(builder, view, occurrence, edge.destination,
                       HandshakeSignalKind::Valid);
      auto destinationReady =
          terminalNode(builder, view, occurrence, edge.destination,
                       HandshakeSignalKind::Ready);
      auto sourceReady = terminalNode(builder, view, occurrence, edge.source,
                                      HandshakeSignalKind::Ready);
      if (!sourceValid)
        return sourceValid.takeError();
      if (!destinationValid)
        return destinationValid.takeError();
      if (!destinationReady)
        return destinationReady.takeError();
      if (!sourceReady)
        return sourceReady.takeError();
      arcs.emplace_back(*sourceValid, *destinationValid);
      arcs.emplace_back(*destinationReady, *sourceReady);
    }
    builder.addFragment(fuCapabilitySelector(occurrence, capabilityRef),
                        std::move(arcs));
    for (const FabricFuTemplateNodeRef &node : record.activeNodes) {
      if (node.node != FabricFuNodeKind::Op)
        continue;
      if (llvm::Error error =
              compileOperation(builder, view, occurrence, capabilityRef, node))
        return std::move(error);
    }
  }
  return builder.finish();
}

} // namespace detail
} // namespace loom::fabric
