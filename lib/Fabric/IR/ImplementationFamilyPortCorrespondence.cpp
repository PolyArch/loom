//===- ImplementationFamilyPortCorrespondence.cpp ------------------------===//

#include "ImplementationFamilyBehaviorInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <vector>

namespace {

using namespace fabric;

llvm::Error reject(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

struct RoutedLaneClass final {
  std::uint32_t effectivePayloadWidth = 0;
  std::vector<std::uint64_t> ordinals;
};

llvm::Error enumerateCanonicalClassSequences(
    llvm::ArrayRef<RoutedLaneClass> classes, std::uint32_t laneCount,
    llvm::function_ref<llvm::Expected<bool>(llvm::ArrayRef<std::uint64_t>,
                                            llvm::ArrayRef<std::uint32_t>)>
        callback) {
  std::vector<std::size_t> classUses(classes.size(), 0);
  std::vector<std::uint64_t> image;
  std::vector<std::uint32_t> effectiveWidths;
  image.reserve(laneCount);
  effectiveWidths.reserve(laneCount);
  bool continueEnumeration = true;
  std::function<llvm::Error()> visit = [&]() -> llvm::Error {
    if (!continueEnumeration)
      return llvm::Error::success();
    if (image.size() == laneCount) {
      auto emitted = callback(image, effectiveWidths);
      if (!emitted)
        return emitted.takeError();
      continueEnumeration = *emitted;
      return llvm::Error::success();
    }
    for (std::size_t classOrdinal = 0;
         continueEnumeration && classOrdinal < classes.size(); ++classOrdinal) {
      const RoutedLaneClass &laneClass = classes[classOrdinal];
      std::size_t &used = classUses[classOrdinal];
      if (used == laneClass.ordinals.size())
        continue;
      image.push_back(laneClass.ordinals[used]);
      effectiveWidths.push_back(laneClass.effectivePayloadWidth);
      ++used;
      if (llvm::Error error = visit())
        return error;
      --used;
      effectiveWidths.pop_back();
      image.pop_back();
    }
    return llvm::Error::success();
  };
  return visit();
}

std::vector<RoutedLaneClass> canonicalLaneClasses(
    const std::map<std::uint32_t, std::vector<std::uint64_t>> &byWidth) {
  std::vector<RoutedLaneClass> classes;
  classes.reserve(byWidth.size());
  for (const auto &[width, ordinals] : byWidth)
    classes.push_back(RoutedLaneClass{width, ordinals});
  return classes;
}

bool fixedRolesAvailable(std::size_t physicalCount, std::size_t roleCount) {
  return roleCount <= physicalCount;
}

} // namespace

llvm::Error fabric::detail::forEachCanonicalRoutedTokenLaneImage(
    ImplementationFamilyId family, const FamilyCapabilityParams &params,
    llvm::ArrayRef<std::uint32_t> physicalInputWidths,
    llvm::ArrayRef<std::uint32_t> physicalResultWidths, std::uint32_t laneCount,
    llvm::function_ref<
        llvm::Expected<bool>(const CanonicalRoutedTokenLaneImage &)>
        callback) {
  const auto *routed = std::get_if<RoutedTokenParams>(&params);
  if (!routed)
    return reject("routed-token family has the wrong parameter schema");
  if (laneCount == 0 || laneCount > routed->maxFan)
    return llvm::Error::success();

  std::map<std::uint32_t, std::vector<std::uint64_t>> lanesByWidth;
  if (family == ImplementationFamilyId::TokenSync) {
    const std::size_t count =
        std::min(physicalInputWidths.size(), physicalResultWidths.size());
    for (std::size_t ordinal = 0; ordinal < count; ++ordinal) {
      const std::uint32_t width =
          std::min({routed->maxPayloadBits, physicalInputWidths[ordinal],
                    physicalResultWidths[ordinal]});
      lanesByWidth[width].push_back(ordinal);
    }
  } else if (family == ImplementationFamilyId::TokenMux) {
    if (physicalInputWidths.empty() || physicalResultWidths.empty())
      return llvm::Error::success();
    for (std::size_t ordinal = 1; ordinal < physicalInputWidths.size();
         ++ordinal) {
      const std::uint32_t width =
          std::min({routed->maxPayloadBits, physicalInputWidths[ordinal],
                    physicalResultWidths.front()});
      lanesByWidth[width].push_back(ordinal);
    }
  } else if (family == ImplementationFamilyId::TokenDemux) {
    if (physicalInputWidths.size() < 2)
      return llvm::Error::success();
    for (std::size_t ordinal = 0; ordinal < physicalResultWidths.size();
         ++ordinal) {
      const std::uint32_t width =
          std::min({routed->maxPayloadBits, physicalInputWidths[1],
                    physicalResultWidths[ordinal]});
      lanesByWidth[width].push_back(ordinal);
    }
  } else {
    return reject("implementation family is not a routed-token family");
  }

  const std::vector<RoutedLaneClass> classes =
      canonicalLaneClasses(lanesByWidth);
  return enumerateCanonicalClassSequences(
      classes, laneCount,
      [&](llvm::ArrayRef<std::uint64_t> image,
          llvm::ArrayRef<std::uint32_t> effectiveWidths)
          -> llvm::Expected<bool> {
        CanonicalRoutedTokenLaneImage candidate;
        candidate.effectivePayloadWidths.assign(effectiveWidths.begin(),
                                                effectiveWidths.end());
        if (family == ImplementationFamilyId::TokenSync) {
          candidate.operandPorts.assign(image.begin(), image.end());
          candidate.resultPorts = candidate.operandPorts;
        } else if (family == ImplementationFamilyId::TokenMux) {
          candidate.operandPorts.push_back(0);
          candidate.operandPorts.insert(candidate.operandPorts.end(),
                                        image.begin(), image.end());
          candidate.resultPorts.push_back(0);
        } else {
          candidate.operandPorts = {0, 1};
          candidate.resultPorts.assign(image.begin(), image.end());
        }
        return callback(candidate);
      });
}

llvm::Error fabric::forEachImplementationFamilyPortCorrespondence(
    ImplementationFamilyId family, const FamilyCapabilityParams &params,
    const ::dataflow::CanonicalActorSchemaProjection &actor,
    llvm::ArrayRef<std::uint32_t> physicalInputWidths,
    llvm::ArrayRef<std::uint32_t> physicalResultWidths,
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

  const std::size_t inputCount = actor.type.getNumInputs();
  const std::size_t resultCount = actor.type.getNumResults();
  if (inputCount > std::numeric_limits<std::uint32_t>::max() ||
      resultCount > std::numeric_limits<std::uint32_t>::max())
    return reject("actor port arity exceeds the routed-token domain");
  const auto emit = [&](llvm::ArrayRef<std::uint64_t> operandPorts,
                        llvm::ArrayRef<std::uint64_t> resultPorts) {
    return callback(operandPorts, resultPorts);
  };

  switch (descriptor.typedAdmissionProvider) {
  case TypedAdmissionProviderId::SyncTokenAdmission:
    if (inputCount != resultCount)
      return llvm::Error::success();
    return detail::forEachCanonicalRoutedTokenLaneImage(
        family, params, physicalInputWidths, physicalResultWidths,
        static_cast<std::uint32_t>(inputCount),
        [&](const detail::CanonicalRoutedTokenLaneImage &image) {
          return emit(image.operandPorts, image.resultPorts);
        });
  case TypedAdmissionProviderId::MuxTokenAdmission:
    if (inputCount == 0 || resultCount != 1)
      return llvm::Error::success();
    return detail::forEachCanonicalRoutedTokenLaneImage(
        family, params, physicalInputWidths, physicalResultWidths,
        static_cast<std::uint32_t>(inputCount - 1),
        [&](const detail::CanonicalRoutedTokenLaneImage &image) {
          return emit(image.operandPorts, image.resultPorts);
        });
  case TypedAdmissionProviderId::DemuxTokenAdmission:
    if (inputCount != 2)
      return llvm::Error::success();
    return detail::forEachCanonicalRoutedTokenLaneImage(
        family, params, physicalInputWidths, physicalResultWidths,
        static_cast<std::uint32_t>(resultCount),
        [&](const detail::CanonicalRoutedTokenLaneImage &image) {
          return emit(image.operandPorts, image.resultPorts);
        });
  case TypedAdmissionProviderId::FixedVectorSliceAlignMergeAdmission: {
    if (resultCount != 1 || physicalResultWidths.empty())
      return llvm::Error::success();
    std::vector<std::uint64_t> operands;
    if (actor.schema == ::dataflow::OperationSchemaId::VectorExtract) {
      if (inputCount == 0)
        return llvm::Error::success();
      operands.push_back(0);
      for (std::size_t ordinal = 1; ordinal < inputCount; ++ordinal)
        operands.push_back(ordinal + 1);
    } else if (actor.schema == ::dataflow::OperationSchemaId::VectorInsert) {
      operands.resize(inputCount);
      std::iota(operands.begin(), operands.end(), 0);
    } else {
      return llvm::Error::success();
    }
    if (llvm::any_of(operands, [&](std::uint64_t port) {
          return port >= physicalInputWidths.size();
        }))
      return llvm::Error::success();
    const std::array<std::uint64_t, 1> resultPort = {0};
    auto emitted = emit(operands, resultPort);
    return emitted ? llvm::Error::success() : emitted.takeError();
  }
  default:
    break;
  }

  if (!fixedRolesAvailable(physicalInputWidths.size(), inputCount) ||
      !fixedRolesAvailable(physicalResultWidths.size(), resultCount))
    return llvm::Error::success();
  std::vector<std::uint64_t> operands(inputCount);
  std::vector<std::uint64_t> resultPorts(resultCount);
  std::iota(operands.begin(), operands.end(), 0);
  std::iota(resultPorts.begin(), resultPorts.end(), 0);
  auto emitted = emit(operands, resultPorts);
  return emitted ? llvm::Error::success() : emitted.takeError();
}

llvm::Error fabric::verifyImplementationFamilyPortCorrespondence(
    ImplementationFamilyId family, const FamilyCapabilityParams &params,
    const ::dataflow::CanonicalActorSchemaProjection &actor,
    llvm::ArrayRef<std::uint64_t> operandPorts,
    llvm::ArrayRef<std::uint64_t> resultPorts,
    llvm::ArrayRef<std::uint32_t> physicalInputWidths,
    llvm::ArrayRef<std::uint32_t> physicalResultWidths) {
  if (operandPorts.size() != actor.type.getNumInputs() ||
      resultPorts.size() != actor.type.getNumResults())
    return reject("port correspondence has the wrong actor arity");

  bool found = false;
  if (llvm::Error error = forEachImplementationFamilyPortCorrespondence(
          family, params, actor, physicalInputWidths, physicalResultWidths,
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

  const ImplementationFamilyDescriptor &descriptor =
      implementationFamily(family);
  switch (descriptor.typedAdmissionProvider) {
  case TypedAdmissionProviderId::SyncTokenAdmission:
    return reject("token sync does not use its canonical ordered physical lane "
                  "embedding");
  case TypedAdmissionProviderId::MuxTokenAdmission:
    return reject("token mux does not use its canonical selector, choice, and "
                  "result embedding");
  case TypedAdmissionProviderId::DemuxTokenAdmission:
    return reject("token demux does not use its canonical selector, data, and "
                  "choice embedding");
  case TypedAdmissionProviderId::FixedVectorSliceAlignMergeAdmission:
    return reject("vector slice changes a fixed physical role");
  default:
    return reject("port correspondence changes a fixed physical role");
  }
}
