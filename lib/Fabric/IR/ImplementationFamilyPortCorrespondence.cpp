//===- ImplementationFamilyPortCorrespondence.cpp ------------------------===//

#include "Fabric/IR/ImplementationFamily.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <functional>

namespace {

llvm::Error reject(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

bool strictlyIncreasing(llvm::ArrayRef<std::uint64_t> ports) {
  return std::adjacent_find(ports.begin(), ports.end(),
                            std::greater_equal<std::uint64_t>()) == ports.end();
}

} // namespace

llvm::Error fabric::verifyImplementationFamilyPortCorrespondence(
    ImplementationFamilyId family,
    const ::dataflow::CanonicalActorSchemaProjection &actor,
    llvm::ArrayRef<std::uint64_t> operandPorts,
    llvm::ArrayRef<std::uint64_t> resultPorts) {
  const std::uint32_t familyIndex = static_cast<std::uint32_t>(family);
  if (familyIndex >= implementationFamilyCount())
    return reject("implementation family is not registered");
  const ImplementationFamilyDescriptor &descriptor =
      implementationFamily(family);
  if (!llvm::is_contained(descriptor.admittedSchemas, actor.schema))
    return reject("actor schema is not admitted by the implementation family");
  if (operandPorts.size() != actor.type.getNumInputs() ||
      resultPorts.size() != actor.type.getNumResults())
    return reject("port correspondence has the wrong actor arity");

  switch (descriptor.typedAdmissionProvider) {
  case TypedAdmissionProviderId::SyncTokenAdmission:
    if (operandPorts != resultPorts || !strictlyIncreasing(operandPorts))
      return reject("token sync must preserve one ordered physical lane image");
    return llvm::Error::success();
  case TypedAdmissionProviderId::MuxTokenAdmission:
    if (operandPorts.empty() || operandPorts.front() != 0 ||
        resultPorts.size() != 1 || resultPorts.front() != 0 ||
        !strictlyIncreasing(operandPorts.drop_front()) ||
        llvm::any_of(operandPorts.drop_front(),
                     [](std::uint64_t port) { return port == 0; }))
      return reject(
          "token mux must preserve selector, choice, and result roles");
    return llvm::Error::success();
  case TypedAdmissionProviderId::DemuxTokenAdmission:
    if (operandPorts.size() != 2 || operandPorts[0] != 0 ||
        operandPorts[1] != 1 || !strictlyIncreasing(resultPorts))
      return reject(
          "token demux must preserve selector, data, and choice roles");
    return llvm::Error::success();
  default:
    break;
  }

  for (auto [ordinal, port] : llvm::enumerate(operandPorts))
    if (port != ordinal)
      return reject("operand correspondence changes a fixed physical role");
  for (auto [ordinal, port] : llvm::enumerate(resultPorts))
    if (port != ordinal)
      return reject("result correspondence changes a fixed physical role");
  return llvm::Error::success();
}
