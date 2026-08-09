//===- ImplementationFamilyPortCorrespondence.cpp ------------------------===//

#include "Fabric/IR/ImplementationFamily.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>
#include <vector>

namespace {

llvm::Error reject(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

std::vector<std::uint64_t>
canonicalDomain(llvm::ArrayRef<std::uint64_t> ports) {
  std::vector<std::uint64_t> result(ports.begin(), ports.end());
  llvm::sort(result);
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

bool contains(llvm::ArrayRef<std::uint64_t> domain, std::uint64_t value) {
  return std::binary_search(domain.begin(), domain.end(), value);
}

llvm::Error enumerateCombinations(
    llvm::ArrayRef<std::uint64_t> domain, std::size_t count,
    llvm::function_ref<llvm::Expected<bool>(llvm::ArrayRef<std::uint64_t>)>
        callback) {
  if (count > domain.size())
    return llvm::Error::success();
  std::vector<std::uint64_t> selected;
  selected.reserve(count);
  bool continueEnumeration = true;
  std::function<llvm::Error(std::size_t)> visit =
      [&](std::size_t begin) -> llvm::Error {
    if (!continueEnumeration)
      return llvm::Error::success();
    if (selected.size() == count) {
      auto result = callback(selected);
      if (!result)
        return result.takeError();
      continueEnumeration = *result;
      return llvm::Error::success();
    }
    const std::size_t remaining = count - selected.size();
    for (std::size_t ordinal = begin;
         continueEnumeration && ordinal + remaining <= domain.size();
         ++ordinal) {
      selected.push_back(domain[ordinal]);
      if (llvm::Error error = visit(ordinal + 1))
        return error;
      selected.pop_back();
    }
    return llvm::Error::success();
  };
  return visit(0);
}

bool fixedRolesAvailable(llvm::ArrayRef<std::uint64_t> domain,
                         std::size_t count) {
  for (std::size_t ordinal = 0; ordinal < count; ++ordinal)
    if (!contains(domain, ordinal))
      return false;
  return true;
}

} // namespace

llvm::Error fabric::forEachImplementationFamilyPortCorrespondence(
    ImplementationFamilyId family,
    const ::dataflow::CanonicalActorSchemaProjection &actor,
    llvm::ArrayRef<std::uint64_t> physicalInputPorts,
    llvm::ArrayRef<std::uint64_t> physicalResultPorts,
    llvm::function_ref<llvm::Expected<bool>(llvm::ArrayRef<std::uint64_t>,
                                            llvm::ArrayRef<std::uint64_t>)>
        callback) {
  const std::uint32_t familyIndex = static_cast<std::uint32_t>(family);
  if (familyIndex >= implementationFamilyCount())
    return reject("implementation family is not registered");
  const ImplementationFamilyDescriptor &descriptor =
      implementationFamily(family);
  if (!llvm::is_contained(descriptor.admittedSchemas, actor.schema))
    return reject("actor schema is not admitted by the implementation family");

  const std::vector<std::uint64_t> inputs = canonicalDomain(physicalInputPorts);
  const std::vector<std::uint64_t> results =
      canonicalDomain(physicalResultPorts);
  const std::size_t inputCount = actor.type.getNumInputs();
  const std::size_t resultCount = actor.type.getNumResults();
  const auto emit = [&](llvm::ArrayRef<std::uint64_t> operandPorts,
                        llvm::ArrayRef<std::uint64_t> resultPorts) {
    return callback(operandPorts, resultPorts);
  };

  switch (descriptor.typedAdmissionProvider) {
  case TypedAdmissionProviderId::SyncTokenAdmission: {
    if (inputCount != resultCount)
      return llvm::Error::success();
    std::vector<std::uint64_t> common;
    std::set_intersection(inputs.begin(), inputs.end(), results.begin(),
                          results.end(), std::back_inserter(common));
    return enumerateCombinations(common, inputCount,
                                 [&](llvm::ArrayRef<std::uint64_t> lanes) {
                                   return emit(lanes, lanes);
                                 });
  }
  case TypedAdmissionProviderId::MuxTokenAdmission: {
    if (inputCount == 0 || resultCount != 1 || !contains(inputs, 0) ||
        !contains(results, 0))
      return llvm::Error::success();
    std::vector<std::uint64_t> choices;
    llvm::copy_if(inputs, std::back_inserter(choices),
                  [](std::uint64_t port) { return port != 0; });
    const std::array<std::uint64_t, 1> resultPort = {0};
    return enumerateCombinations(
        choices, inputCount - 1,
        [&](llvm::ArrayRef<std::uint64_t> selectedChoices)
            -> llvm::Expected<bool> {
          std::vector<std::uint64_t> operands = {0};
          operands.insert(operands.end(), selectedChoices.begin(),
                          selectedChoices.end());
          return emit(operands, resultPort);
        });
  }
  case TypedAdmissionProviderId::DemuxTokenAdmission: {
    if (inputCount != 2 || !contains(inputs, 0) || !contains(inputs, 1))
      return llvm::Error::success();
    const std::array<std::uint64_t, 2> operandPorts = {0, 1};
    return enumerateCombinations(
        results, resultCount,
        [&](llvm::ArrayRef<std::uint64_t> selectedResults) {
          return emit(operandPorts, selectedResults);
        });
  }
  case TypedAdmissionProviderId::FixedVectorSliceAlignMergeAdmission: {
    if (resultCount != 1 || !contains(results, 0))
      return llvm::Error::success();
    std::vector<std::uint64_t> operands;
    if (actor.schema == ::dataflow::OperationSchemaId::VectorExtract) {
      if (inputCount == 0)
        return llvm::Error::success();
      operands.push_back(0);
      for (std::size_t ordinal = 1; ordinal < inputCount; ++ordinal)
        operands.push_back(ordinal + 1);
    } else if (actor.schema == ::dataflow::OperationSchemaId::VectorInsert) {
      for (std::size_t ordinal = 0; ordinal < inputCount; ++ordinal)
        operands.push_back(ordinal);
    } else {
      return llvm::Error::success();
    }
    if (!llvm::all_of(operands, [&](std::uint64_t port) {
          return contains(inputs, port);
        }))
      return llvm::Error::success();
    const std::array<std::uint64_t, 1> resultPort = {0};
    auto emitted = emit(operands, resultPort);
    return emitted ? llvm::Error::success() : emitted.takeError();
  }
  default:
    break;
  }

  if (!fixedRolesAvailable(inputs, inputCount) ||
      !fixedRolesAvailable(results, resultCount))
    return llvm::Error::success();
  std::vector<std::uint64_t> operands(inputCount);
  std::vector<std::uint64_t> resultPorts(resultCount);
  std::iota(operands.begin(), operands.end(), 0);
  std::iota(resultPorts.begin(), resultPorts.end(), 0);
  auto emitted = emit(operands, resultPorts);
  return emitted ? llvm::Error::success() : emitted.takeError();
}

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

  bool found = false;
  if (llvm::Error error = forEachImplementationFamilyPortCorrespondence(
          family, actor, operandPorts, resultPorts,
          [&](llvm::ArrayRef<std::uint64_t> candidateOperands,
              llvm::ArrayRef<std::uint64_t> candidateResults)
              -> llvm::Expected<bool> {
            found = candidateOperands == operandPorts &&
                    candidateResults == resultPorts;
            return !found;
          }))
    return error;
  if (found)
    return llvm::Error::success();

  switch (descriptor.typedAdmissionProvider) {
  case TypedAdmissionProviderId::SyncTokenAdmission:
    return reject("token sync must preserve one ordered physical lane image");
  case TypedAdmissionProviderId::MuxTokenAdmission:
    return reject("token mux must preserve selector, choice, and result roles");
  case TypedAdmissionProviderId::DemuxTokenAdmission:
    return reject("token demux must preserve selector, data, and choice roles");
  case TypedAdmissionProviderId::FixedVectorSliceAlignMergeAdmission:
    return reject("vector slice changes a fixed physical role");
  default:
    return reject("port correspondence changes a fixed physical role");
  }
}
