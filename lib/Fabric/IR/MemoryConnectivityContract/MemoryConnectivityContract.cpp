#include "Fabric/IR/MemoryConnectivityContract.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <set>
#include <system_error>
#include <tuple>

namespace fabric {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(std::errc::invalid_argument,
                                 "invalid MemoryConnectivityContractRecord: %s",
                                 message.str().c_str());
}

bool validMatchField(MemoryProviderMatchField field) {
  switch (field) {
  case MemoryProviderMatchField::Range:
  case MemoryProviderMatchField::Prefix:
  case MemoryProviderMatchField::AddressSpace:
  case MemoryProviderMatchField::Context:
    return true;
  }
  return false;
}

bool validAddressTransform(MemoryProviderAddressTransform transform) {
  switch (transform) {
  case MemoryProviderAddressTransform::None:
  case MemoryProviderAddressTransform::ConstantBaseOffset:
    return true;
  }
  return false;
}

bool targetLess(const MemoryDispatchTarget &left,
                const MemoryDispatchTarget &right) {
  if (left.index() != right.index())
    return left.index() < right.index();
  if (const auto *leftManager = std::get_if<ManagerMemoryDispatchTarget>(&left))
    return leftManager->endpointOrdinal <
           std::get<ManagerMemoryDispatchTarget>(right).endpointOrdinal;
  return false;
}

llvm::Error normalizeTargets(std::vector<MemoryDispatchTarget> &targets) {
  if (targets.empty())
    return invalid("dispatch target domain is empty");
  llvm::sort(targets, targetLess);
  if (std::adjacent_find(targets.begin(), targets.end()) != targets.end())
    return invalid("dispatch target domain contains a duplicate target");
  return llvm::Error::success();
}

llvm::Expected<MemoryConnectivityDeclaration>
normalize(MemoryConnectivityDeclaration declaration) {
  for (MemoryOperationPortDispatchDeclaration &port :
       declaration.operationPorts) {
    if (port.capabilityTargetDomains.empty())
      return invalid("operation port has no capability target domain");
    for (std::vector<MemoryDispatchTarget> &targets :
         port.capabilityTargetDomains)
      if (llvm::Error error = normalizeTargets(targets))
        return std::move(error);
  }

  for (MemorySubordinateDispatchDeclaration &subordinate :
       declaration.subordinateEndpoints) {
    if (subordinate.maxExposedBindings == 0)
      return invalid("subordinate provider capacity is zero");
    if (!validAddressTransform(subordinate.addressTransform))
      return invalid("subordinate provider has an unknown address transform");
    for (MemoryProviderMatchField field : subordinate.matchFields)
      if (!validMatchField(field))
        return invalid("subordinate provider has an unknown match field");
    llvm::sort(subordinate.matchFields, [](MemoryProviderMatchField left,
                                           MemoryProviderMatchField right) {
      return static_cast<std::uint32_t>(left) <
             static_cast<std::uint32_t>(right);
    });
    if (std::adjacent_find(subordinate.matchFields.begin(),
                           subordinate.matchFields.end()) !=
        subordinate.matchFields.end())
      return invalid("subordinate provider repeats a match field");
    if (subordinate.maxExposedBindings > 1 && subordinate.matchFields.empty())
      return invalid(
          "multi-binding subordinate provider has no bounded match field");
    if (llvm::Error error = normalizeTargets(subordinate.targetDomain))
      return std::move(error);
  }

  llvm::sort(
      declaration.internalConnections,
      [](const MemoryInternalConnectionDeclaration &left,
         const MemoryInternalConnectionDeclaration &right) {
        return std::tie(left.sourceEndpointOrdinal, left.sinkEndpointOrdinal) <
               std::tie(right.sourceEndpointOrdinal, right.sinkEndpointOrdinal);
      });
  if (std::adjacent_find(declaration.internalConnections.begin(),
                         declaration.internalConnections.end(),
                         [](const MemoryInternalConnectionDeclaration &left,
                            const MemoryInternalConnectionDeclaration &right) {
                           return left.sourceEndpointOrdinal ==
                                      right.sourceEndpointOrdinal &&
                                  left.sinkEndpointOrdinal ==
                                      right.sinkEndpointOrdinal;
                         }) != declaration.internalConnections.end())
    return invalid("internal connectivity contains a duplicate edge");
  return declaration;
}

bool equalDeclaration(const MemoryConnectivityDeclaration &left,
                      const MemoryConnectivityDeclaration &right) {
  if (left.operationPorts.size() != right.operationPorts.size() ||
      left.subordinateEndpoints.size() != right.subordinateEndpoints.size() ||
      left.internalConnections.size() != right.internalConnections.size())
    return false;
  for (std::size_t port = 0; port < left.operationPorts.size(); ++port)
    if (left.operationPorts[port].capabilityTargetDomains !=
        right.operationPorts[port].capabilityTargetDomains)
      return false;
  for (std::size_t index = 0; index < left.subordinateEndpoints.size();
       ++index) {
    const auto &a = left.subordinateEndpoints[index];
    const auto &b = right.subordinateEndpoints[index];
    if (a.maxExposedBindings != b.maxExposedBindings ||
        a.matchFields != b.matchFields ||
        a.addressTransform != b.addressTransform ||
        a.targetDomain != b.targetDomain)
      return false;
  }
  for (std::size_t index = 0; index < left.internalConnections.size();
       ++index) {
    const auto &a = left.internalConnections[index];
    const auto &b = right.internalConnections[index];
    if (a.sourceEndpointOrdinal != b.sourceEndpointOrdinal ||
        a.sinkEndpointOrdinal != b.sinkEndpointOrdinal)
      return false;
  }
  return true;
}

llvm::Error validateTarget(const MemoryDispatchTarget &target,
                           std::uint64_t managerEndpointCount,
                           bool hasLocalMemoryService) {
  if (std::holds_alternative<LocalMemoryDispatchTarget>(target)) {
    if (!hasLocalMemoryService)
      return invalid("dispatch target selects an absent Local Memory Service");
    return llvm::Error::success();
  }
  const auto &manager = std::get<ManagerMemoryDispatchTarget>(target);
  if (manager.endpointOrdinal >= managerEndpointCount)
    return invalid("dispatch target selects an unknown manager endpoint");
  return llvm::Error::success();
}

llvm::Error validateTargets(llvm::ArrayRef<MemoryDispatchTarget> targets,
                            std::uint64_t managerEndpointCount,
                            bool hasLocalMemoryService) {
  for (const MemoryDispatchTarget &target : targets)
    if (llvm::Error error =
            validateTarget(target, managerEndpointCount, hasLocalMemoryService))
      return error;
  return llvm::Error::success();
}

} // namespace

llvm::Expected<MemoryConnectivityContractRecord>
MemoryConnectivityContractRecord::create(
    MemoryConnectivityDeclaration declaration) {
  auto normalized = normalize(std::move(declaration));
  if (!normalized)
    return normalized.takeError();
  return MemoryConnectivityContractRecord(std::move(*normalized));
}

llvm::Expected<MemoryConnectivityContractRecord>
MemoryConnectivityContractRecord::fromCanonical(
    MemoryConnectivityDeclaration declaration) {
  MemoryConnectivityDeclaration original = declaration;
  auto normalized = normalize(std::move(declaration));
  if (!normalized)
    return normalized.takeError();
  if (!equalDeclaration(original, *normalized))
    return invalid("record fields are not in canonical order");
  return MemoryConnectivityContractRecord(std::move(*normalized));
}

llvm::Error validateMemoryConnectivityContract(
    const MemoryConnectivityContractRecord &record,
    llvm::ArrayRef<MemoryOperationPortRecord> operationPorts,
    llvm::ArrayRef<MemoryTransportEndpointDescriptor> transportEndpoints,
    std::uint64_t managerEndpointCount, std::uint64_t subordinateEndpointCount,
    bool hasLocalMemoryService) {
  if (record.operationPorts().size() != operationPorts.size())
    return invalid(
        "operation dispatch inventory does not match operation ports");
  if (record.subordinateEndpoints().size() != subordinateEndpointCount)
    return invalid("subordinate dispatch inventory does not match endpoints");

  std::set<std::uint64_t> operationEndpoints;
  for (std::size_t portOrdinal = 0; portOrdinal < operationPorts.size();
       ++portOrdinal) {
    const MemoryOperationPortRecord &port = operationPorts[portOrdinal];
    const MemoryOperationPortDispatchDeclaration &dispatch =
        record.operationPorts()[portOrdinal];
    if (dispatch.capabilityTargetDomains.size() !=
        port.capabilityAlternatives().size())
      return invalid(
          "operation dispatch alternatives do not match port capability");
    for (llvm::ArrayRef<MemoryDispatchTarget> targets :
         dispatch.capabilityTargetDomains)
      if (llvm::Error error = validateTargets(targets, managerEndpointCount,
                                              hasLocalMemoryService))
        return error;
    operationEndpoints.insert(port.endpointInventory().begin(),
                              port.endpointInventory().end());
  }

  for (const MemorySubordinateDispatchDeclaration &subordinate :
       record.subordinateEndpoints())
    if (llvm::Error error =
            validateTargets(subordinate.targetDomain, managerEndpointCount,
                            hasLocalMemoryService))
      return error;

  if (!operationPorts.empty() &&
      operationEndpoints.size() != transportEndpoints.size())
    return invalid(
        "operation-port inventories do not cover every token endpoint");

  for (const MemoryInternalConnectionDeclaration &connection :
       record.internalConnections()) {
    if (connection.sourceEndpointOrdinal >= transportEndpoints.size() ||
        connection.sinkEndpointOrdinal >= transportEndpoints.size())
      return invalid("internal connection references an unknown endpoint");
    if (!operationEndpoints.count(connection.sourceEndpointOrdinal) ||
        !operationEndpoints.count(connection.sinkEndpointOrdinal))
      return invalid("internal connection references a non-operation endpoint");
    const auto &source = transportEndpoints[connection.sourceEndpointOrdinal];
    const auto &sink = transportEndpoints[connection.sinkEndpointOrdinal];
    if (source.direction != loom::fabric::FabricPortDirection::Output ||
        sink.direction != loom::fabric::FabricPortDirection::Input)
      return invalid("internal connection direction is not output-to-input");
    if (source.tagWidth.has_value() != sink.tagWidth.has_value())
      return invalid("internal connection crosses Fabric token kinds");
    if (source.payloadWidth < sink.payloadWidth)
      return invalid("internal connection narrows its required payload");
  }
  return llvm::Error::success();
}

} // namespace fabric
