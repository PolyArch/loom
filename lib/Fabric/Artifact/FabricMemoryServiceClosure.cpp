#include "Fabric/Artifact/FabricMemoryServiceClosure.h"

#include "Fabric/IR/MemoryServiceContract.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DynamicAPInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::fabric {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_memory_service_closure_invalid: " +
                                     message);
}

std::string bytesKey(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

template <typename Ref> std::string refKey(const Ref &reference) {
  const auto bytes = canonicalFabricBytes(reference);
  return bytesKey(bytes);
}

struct FrontierBranch final {
  FabricMemoryEndpointRef endpoint;
  std::vector<SystemServiceTransformRef> path;
  std::optional<FabricMemoryServiceRegionRef> requiredTerminalRegion;
};

struct TerminalBranch final {
  FabricMemoryServiceRef service;
  std::vector<SystemServiceTransformRef> path;
};

std::string branchKey(const FabricMemoryServiceTargetBranch &branch) {
  std::string result;
  for (const SystemServiceTransformRef transform : branch.transformPath) {
    const auto bytes = canonicalFabricBytes(transform);
    result.append(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  }
  const auto region = canonicalFabricBytes(branch.region);
  result.append(reinterpret_cast<const char *>(region.data()), region.size());
  return result;
}

std::string planKey(const FabricMemoryServiceTargetPlan &plan) {
  std::string result;
  for (const FabricMemoryServiceTargetBranch &branch : plan.branches) {
    const std::string branchBytes = branchKey(branch);
    const std::uint64_t size = branchBytes.size();
    for (int shift = 56; shift >= 0; shift -= 8)
      result.push_back(static_cast<char>(size >> shift));
    result.append(branchBytes);
  }
  return result;
}

void canonicalizePlan(FabricMemoryServiceTargetPlan &plan) {
  llvm::sort(plan.branches, [](const auto &left, const auto &right) {
    return branchKey(left) < branchKey(right);
  });
  plan.branches.erase(std::unique(plan.branches.begin(), plan.branches.end()),
                      plan.branches.end());
}

llvm::Expected<std::optional<TerminalBranch>>
terminalBranch(const FabricSystemRootView &system,
               const FrontierBranch &branch) {
  const auto *endpoint =
      std::get_if<SystemServiceEndpointRef>(&branch.endpoint.owner.payload);
  if (!endpoint)
    return invalid(
        "closure endpoint is not owned by a System service endpoint");
  const SystemServiceEndpointOwnerRef *owner =
      system.serviceEndpointOwner(*endpoint);
  if (!owner)
    return invalid("closure endpoint has no canonical owner");
  const auto *service =
      std::get_if<FabricMemoryServiceRef>(&owner->owner().payload);
  if (!service)
    return std::nullopt;
  if (!std::holds_alternative<SystemMemoryServiceRef>(service->payload))
    return invalid("System closure terminates in a non-System memory service");
  return TerminalBranch{*service, branch.path};
}

llvm::Expected<std::optional<SystemServiceTransformRef>>
endpointTransform(const FabricSystemRootView &system,
                  const FabricMemoryEndpointRef &memory) {
  const auto *endpoint =
      std::get_if<SystemServiceEndpointRef>(&memory.owner.payload);
  if (!endpoint)
    return invalid(
        "closure endpoint is not owned by a System service endpoint");
  const SystemServiceEndpointOwnerRef *owner =
      system.serviceEndpointOwner(*endpoint);
  if (!owner)
    return invalid("closure endpoint has no canonical owner");
  const auto *transform =
      std::get_if<SystemServiceTransformRef>(&owner->owner().payload);
  return transform ? std::optional<SystemServiceTransformRef>(*transform)
                   : std::nullopt;
}

llvm::Expected<std::vector<FabricMemoryServiceTargetPlan>>
terminalPlans(const FabricSystemRootView &system,
              llvm::ArrayRef<FrontierBranch> frontier) {
  std::vector<TerminalBranch> terminals;
  terminals.reserve(frontier.size());
  for (const FrontierBranch &branch : frontier) {
    auto terminal = terminalBranch(system, branch);
    if (!terminal)
      return terminal.takeError();
    if (!*terminal)
      return invalid("unfinished transform endpoint reached terminal closure");
    terminals.push_back(std::move(**terminal));
  }

  std::vector<FabricMemoryServiceTargetPlan> plans(1);
  for (auto [branchIndex, terminal] : llvm::enumerate(terminals)) {
    const auto &serviceRef =
        std::get<SystemMemoryServiceRef>(terminal.service.payload);
    const ::fabric::MemoryServiceContractRecord *service =
        system.memoryService(serviceRef);
    if (!service)
      return invalid("terminal memory service has no canonical contract");
    std::vector<FabricMemoryServiceTargetPlan> expanded;
    expanded.reserve(plans.size() * service->regions().size());
    for (const FabricMemoryServiceTargetPlan &plan : plans) {
      for (FabricOrdinal ordinal = 0; ordinal < service->regions().size();
           ++ordinal) {
        const FabricMemoryServiceRegionRef region{terminal.service, ordinal};
        if (frontier[branchIndex].requiredTerminalRegion &&
            *frontier[branchIndex].requiredTerminalRegion != region)
          continue;
        FabricMemoryServiceTargetPlan next = plan;
        next.branches.push_back({terminal.path, region});
        expanded.push_back(std::move(next));
      }
    }
    plans = std::move(expanded);
  }
  for (FabricMemoryServiceTargetPlan &plan : plans)
    canonicalizePlan(plan);
  llvm::sort(plans, [](const auto &left, const auto &right) {
    return planKey(left) < planKey(right);
  });
  plans.erase(std::unique(plans.begin(), plans.end()), plans.end());
  return plans;
}

} // namespace

llvm::Expected<std::vector<FabricMemoryServiceTargetPlan>>
projectFabricMemoryServiceTargetPlans(const FabricSystemRootView &system,
                                      SystemServiceEndpointRef endpoint) {
  FabricMemoryEndpointRef root{FabricMemoryEndpointOwnerRef::of(endpoint), 0};
  if (!system.artifact().memoryEndpointRole(root))
    return invalid("root endpoint has no Memory-plane endpoint zero");

  std::map<std::string, FabricMemoryEndpointRef> connections;
  for (const FabricMemoryServiceConnectionPayload &connection :
       system.artifact().memoryServiceConnections())
    connections.emplace(refKey(connection.source), connection.destination);

  const std::size_t transformLimit =
      system.artifact().systemServiceTransforms().size();
  std::vector<std::vector<FrontierBranch>> worklist{{{root, {}, std::nullopt}}};
  std::vector<FabricMemoryServiceTargetPlan> plans;
  while (!worklist.empty()) {
    std::vector<FrontierBranch> frontier = std::move(worklist.back());
    worklist.pop_back();
    bool advanced = false;
    bool infeasible = false;
    for (std::size_t position = 0; position < frontier.size(); ++position) {
      FrontierBranch &branch = frontier[position];
      const auto role = system.artifact().memoryEndpointRole(branch.endpoint);
      if (!role)
        return invalid("closure contains an unknown memory endpoint");
      if (*role == FabricMemoryEndpointRole::Manager) {
        const auto found = connections.find(refKey(branch.endpoint));
        if (found == connections.end()) {
          infeasible = true;
          break;
        }
        branch.endpoint = found->second;
        worklist.push_back(std::move(frontier));
        advanced = true;
        break;
      }
      if (*role != FabricMemoryEndpointRole::Subordinate)
        return invalid("closure continuation is not subordinate");
      auto terminal = terminalBranch(system, branch);
      if (!terminal)
        return terminal.takeError();
      if (*terminal)
        continue;
      if (branch.requiredTerminalRegion) {
        infeasible = true;
        break;
      }
      auto transform = endpointTransform(system, branch.endpoint);
      if (!transform)
        return transform.takeError();
      if (!*transform)
        return invalid(
            "subordinate endpoint has no service or transform owner");
      if (branch.path.size() >= transformLimit ||
          llvm::is_contained(branch.path, **transform)) {
        infeasible = true;
        break;
      }
      const SystemServiceTransformRecord *record =
          system.serviceTransform(**transform);
      if (!record)
        return invalid("closure transform has no canonical contract");
      if (!llvm::is_contained(record->inputs(), branch.endpoint))
        return invalid("closure enters a transform through a non-input");
      std::vector<SystemServiceTransformRef> path = branch.path;
      path.push_back(**transform);
      if (const auto *coherent =
              std::get_if<CoherentMemoryTransform>(&record->contract())) {
        for (const auto &correspondence : coherent->regions) {
          for (const FabricMemoryEndpointRef output : record->outputs()) {
            std::vector<FrontierBranch> next = frontier;
            next[position] = {output, path, correspondence.output};
            worklist.push_back(std::move(next));
          }
        }
      } else {
        std::vector<FrontierBranch> outputs;
        outputs.reserve(record->outputs().size());
        for (const FabricMemoryEndpointRef output : record->outputs())
          outputs.push_back({output, path, std::nullopt});
        frontier.erase(frontier.begin() + position);
        frontier.insert(frontier.begin() + position,
                        std::make_move_iterator(outputs.begin()),
                        std::make_move_iterator(outputs.end()));
        worklist.push_back(std::move(frontier));
      }
      advanced = true;
      break;
    }
    if (infeasible || advanced)
      continue;
    auto terminal = terminalPlans(system, frontier);
    if (!terminal)
      return terminal.takeError();
    plans.insert(plans.end(), std::make_move_iterator(terminal->begin()),
                 std::make_move_iterator(terminal->end()));
  }
  llvm::sort(plans, [](const auto &left, const auto &right) {
    return planKey(left) < planKey(right);
  });
  plans.erase(std::unique(plans.begin(), plans.end()), plans.end());
  return plans;
}

namespace {

using mlir::presburger::IntegerPolyhedron;
using mlir::presburger::IntegerRelation;
using mlir::presburger::PresburgerRelation;
using mlir::presburger::PresburgerSet;
using mlir::presburger::PresburgerSpace;

llvm::DynamicAPInt dynamicUnsigned(std::uint64_t value) {
  return llvm::DynamicAPInt(llvm::APInt(65, value, /*isSigned=*/false));
}

llvm::DynamicAPInt dynamicPowerOfTwo(unsigned exponent) {
  llvm::APInt value(exponent + 2, 1, /*isSigned=*/false);
  value <<= exponent;
  return llvm::DynamicAPInt(value);
}

llvm::SmallVector<llvm::DynamicAPInt>
constraintRow(const IntegerRelation &relation) {
  return llvm::SmallVector<llvm::DynamicAPInt>(relation.getNumVars() + 1,
                                               llvm::DynamicAPInt(0));
}

void addLowerBound(IntegerRelation &relation, unsigned position,
                   const llvm::DynamicAPInt &lower) {
  auto row = constraintRow(relation);
  row[position] = llvm::DynamicAPInt(1);
  row.back() = -lower;
  relation.addInequality(row);
}

void addUpperBound(IntegerRelation &relation, unsigned position,
                   const llvm::DynamicAPInt &upper) {
  auto row = constraintRow(relation);
  row[position] = llvm::DynamicAPInt(-1);
  row.back() = upper;
  relation.addInequality(row);
}

llvm::Error addInterval(IntegerRelation &relation, unsigned position,
                        std::uint64_t base, std::uint64_t size) {
  if (size == 0)
    return invalid("address interval has zero size");
  const auto end = llvm::checkedAddUnsigned(base, size);
  if (!end)
    return invalid("address interval overflows u64");
  addLowerBound(relation, position, dynamicUnsigned(base));
  addUpperBound(relation, position, dynamicUnsigned(*end - 1));
  return llvm::Error::success();
}

void addAddressWidth(IntegerRelation &relation, unsigned position,
                     std::uint32_t width) {
  addLowerBound(relation, position, llvm::DynamicAPInt(0));
  addUpperBound(relation, position,
                dynamicPowerOfTwo(width) - llvm::DynamicAPInt(1));
}

llvm::Expected<PresburgerSet>
intervalSet(FabricMemoryServiceSourceInterval interval) {
  IntegerPolyhedron set(PresburgerSpace::getSetSpace(1));
  if (llvm::Error error =
          addInterval(set, 0, interval.addressBaseBytes, interval.sizeBytes))
    return std::move(error);
  return PresburgerSet(set);
}

llvm::Expected<PresburgerRelation>
sourceIdentity(FabricMemoryServiceSourceInterval interval) {
  IntegerRelation relation(PresburgerSpace::getRelationSpace(1, 1));
  auto equality = constraintRow(relation);
  equality[0] = llvm::DynamicAPInt(-1);
  equality[1] = llvm::DynamicAPInt(1);
  relation.addEquality(equality);
  if (llvm::Error error = addInterval(relation, 0, interval.addressBaseBytes,
                                      interval.sizeBytes))
    return std::move(error);
  return PresburgerRelation(relation);
}

PresburgerRelation addressOffsetRelation(const AddressOffsetTransform &offset) {
  IntegerRelation relation(PresburgerSpace::getRelationSpace(1, 1));
  auto equality = constraintRow(relation);
  equality[0] = llvm::DynamicAPInt(-1);
  equality[1] = llvm::DynamicAPInt(1);
  equality.back() = -llvm::DynamicAPInt(offset.signedOffset);
  relation.addEquality(equality);
  addAddressWidth(relation, 0, offset.addressWidth);
  addAddressWidth(relation, 1, offset.addressWidth);
  return PresburgerRelation(relation);
}

PresburgerRelation
addressMaskXorRelation(const AddressMaskXorTransform &transform) {
  const unsigned width = transform.addressWidth;
  IntegerRelation relation(
      PresburgerSpace::getRelationSpace(1, 1, /*numSymbols=*/0, width));
  auto input = constraintRow(relation);
  auto output = constraintRow(relation);
  input[0] = llvm::DynamicAPInt(1);
  output[1] = llvm::DynamicAPInt(1);
  const unsigned localOffset = 2;
  for (unsigned bit = 0; bit < width; ++bit) {
    const llvm::DynamicAPInt weight = dynamicPowerOfTwo(bit);
    input[localOffset + bit] = -weight;
    const bool retained = (transform.andMask >> bit) & 1;
    const bool inverted = (transform.xorMask >> bit) & 1;
    if (retained) {
      if (inverted) {
        output[localOffset + bit] = weight;
        output.back() -= weight;
      } else {
        output[localOffset + bit] = -weight;
      }
    } else if (inverted) {
      output.back() -= weight;
    }
    addLowerBound(relation, localOffset + bit, llvm::DynamicAPInt(0));
    addUpperBound(relation, localOffset + bit, llvm::DynamicAPInt(1));
  }
  relation.addEquality(input);
  relation.addEquality(output);
  return PresburgerRelation(relation);
}

PresburgerRelation
staticInterleaveRelation(const StaticInterleaveTransform &transform,
                         std::uint64_t outputOrdinal) {
  IntegerRelation relation(PresburgerSpace::getRelationSpace(1, 1,
                                                             /*numSymbols=*/0,
                                                             /*numLocals=*/3));
  const unsigned quotient = 2;
  const unsigned remainder = 3;
  const unsigned outer = 4;
  const llvm::DynamicAPInt granule = dynamicUnsigned(transform.granuleBytes);
  const llvm::DynamicAPInt outputCount = dynamicUnsigned(transform.outputCount);

  auto divide = constraintRow(relation);
  divide[0] = llvm::DynamicAPInt(1);
  divide[quotient] = -granule;
  divide[remainder] = llvm::DynamicAPInt(-1);
  relation.addEquality(divide);

  auto select = constraintRow(relation);
  select[quotient] = llvm::DynamicAPInt(1);
  select[outer] = -outputCount;
  select.back() = -dynamicUnsigned(outputOrdinal);
  relation.addEquality(select);

  auto compress = constraintRow(relation);
  compress[1] = llvm::DynamicAPInt(1);
  compress[outer] = -granule;
  compress[remainder] = llvm::DynamicAPInt(-1);
  relation.addEquality(compress);

  addLowerBound(relation, quotient, llvm::DynamicAPInt(0));
  addLowerBound(relation, remainder, llvm::DynamicAPInt(0));
  addUpperBound(relation, remainder, granule - llvm::DynamicAPInt(1));
  addLowerBound(relation, outer, llvm::DynamicAPInt(0));
  return PresburgerRelation(relation);
}

llvm::Expected<const ::fabric::MemoryServiceRegionDeclaration *>
regionDeclaration(const FabricSystemRootView &system,
                  FabricMemoryServiceRegionRef region) {
  const auto *service =
      std::get_if<SystemMemoryServiceRef>(&region.service.payload);
  if (!service)
    return invalid("System target region belongs to a non-System service");
  const ::fabric::MemoryServiceContractRecord *contract =
      system.memoryService(*service);
  if (!contract || region.ordinal >= contract->regions().size())
    return invalid("System target region has no canonical declaration");
  return &contract->regions()[region.ordinal];
}

llvm::Expected<PresburgerRelation> coherentRegionRelation(
    const FabricSystemRootView &system,
    const CoherentMemoryRegionCorrespondence &correspondence) {
  auto input = regionDeclaration(system, correspondence.input);
  if (!input)
    return input.takeError();
  auto output = regionDeclaration(system, correspondence.output);
  if (!output)
    return output.takeError();
  if ((*input)->sizeBytes != (*output)->sizeBytes)
    return invalid("CoherentMemory correspondence has unequal region sizes");

  IntegerRelation relation(PresburgerSpace::getRelationSpace(1, 1));
  auto equality = constraintRow(relation);
  equality[0] = llvm::DynamicAPInt(-1);
  equality[1] = llvm::DynamicAPInt(1);
  equality.back() = dynamicUnsigned((*input)->addressBaseBytes) -
                    dynamicUnsigned((*output)->addressBaseBytes);
  relation.addEquality(equality);
  if (llvm::Error error = addInterval(relation, 0, (*input)->addressBaseBytes,
                                      (*input)->sizeBytes))
    return std::move(error);
  if (llvm::Error error = addInterval(relation, 1, (*output)->addressBaseBytes,
                                      (*output)->sizeBytes))
    return std::move(error);
  return PresburgerRelation(relation);
}

PresburgerRelation compose(PresburgerRelation relation,
                           const PresburgerRelation &next) {
  relation.compose(next);
  return relation;
}

struct AddressEnvelope final {
  std::uint64_t lower;
  std::uint64_t upper;
};

bool contains(const ::fabric::MemoryServiceRegionDeclaration &region,
              AddressEnvelope envelope) {
  const auto end =
      llvm::checkedAddUnsigned(region.addressBaseBytes, region.sizeBytes);
  return end && envelope.lower >= region.addressBaseBytes &&
         envelope.upper < *end;
}

std::optional<AddressEnvelope>
offsetEnvelope(AddressEnvelope input, const AddressOffsetTransform &offset) {
  const __int128 lower = static_cast<__int128>(input.lower) +
                         static_cast<__int128>(offset.signedOffset);
  const __int128 upper = static_cast<__int128>(input.upper) +
                         static_cast<__int128>(offset.signedOffset);
  const unsigned __int128 limit =
      (static_cast<unsigned __int128>(1) << offset.addressWidth) - 1;
  if (lower < 0 || upper < 0 || static_cast<unsigned __int128>(upper) > limit)
    return std::nullopt;
  return AddressEnvelope{static_cast<std::uint64_t>(lower),
                         static_cast<std::uint64_t>(upper)};
}

std::optional<AddressEnvelope>
maskEnvelope(AddressEnvelope input, const AddressMaskXorTransform &mask) {
  const std::uint64_t widthMask =
      mask.addressWidth == 64 ? std::numeric_limits<std::uint64_t>::max()
                              : (std::uint64_t{1} << mask.addressWidth) - 1;
  if (input.upper > widthMask)
    return std::nullopt;
  const std::uint64_t fixed = mask.xorMask & ~mask.andMask & widthMask;
  return AddressEnvelope{fixed, fixed | mask.andMask};
}

struct ExactFrontierBranch final {
  FabricMemoryEndpointRef endpoint;
  std::vector<SystemServiceTransformRef> path;
  PresburgerRelation addresses;
  std::optional<FabricMemoryServiceRegionRef> requiredTerminalRegion;
  bool coversWholeSource;
  std::optional<AddressEnvelope> envelope;
};

struct ExactTerminalGroup final {
  FabricMemoryServiceTargetBranch branch;
  PresburgerSet sourceDomain;
};

llvm::Expected<std::vector<FabricMemoryServiceTargetPlan>>
exactTerminalPlans(const FabricSystemRootView &system,
                   llvm::ArrayRef<ExactFrontierBranch> frontier,
                   FabricMemoryServiceSourceInterval sourceInterval) {
  auto sourceDomain = intervalSet(sourceInterval);
  if (!sourceDomain)
    return sourceDomain.takeError();

  std::vector<ExactTerminalGroup> groups;
  std::map<std::string, std::size_t> groupByKey;
  for (const ExactFrontierBranch &frontierBranch : frontier) {
    FrontierBranch structural{frontierBranch.endpoint, frontierBranch.path,
                              std::nullopt};
    auto terminal = terminalBranch(system, structural);
    if (!terminal)
      return terminal.takeError();
    if (!*terminal)
      return invalid("exact closure contains an unfinished transform branch");
    const auto &serviceRef =
        std::get<SystemMemoryServiceRef>((*terminal)->service.payload);
    const ::fabric::MemoryServiceContractRecord *service =
        system.memoryService(serviceRef);
    if (!service)
      return invalid("exact closure terminal has no service contract");

    for (FabricOrdinal ordinal = 0; ordinal < service->regions().size();
         ++ordinal) {
      FabricMemoryServiceRegionRef region{(*terminal)->service, ordinal};
      if (frontierBranch.requiredTerminalRegion &&
          *frontierBranch.requiredTerminalRegion != region)
        continue;
      const auto &declaration = service->regions()[ordinal];
      auto terminalDomain =
          intervalSet({declaration.addressBaseBytes, declaration.sizeBytes});
      if (!terminalDomain)
        return terminalDomain.takeError();
      const bool wholeSourceContained =
          frontierBranch.coversWholeSource && frontierBranch.envelope &&
          contains(declaration, *frontierBranch.envelope);
      PresburgerSet domain =
          wholeSourceContained
              ? *sourceDomain
              : frontierBranch.addresses.intersectRange(*terminalDomain)
                    .getDomainSet()
                    .intersect(*sourceDomain);
      if (domain.isIntegerEmpty())
        continue;
      if (!wholeSourceContained)
        domain =
            PresburgerSet(domain.computeReprWithOnlyDivLocals()).coalesce();
      FabricMemoryServiceTargetBranch branch{(*terminal)->path, region};
      const std::string key = branchKey(branch);
      const auto [found, inserted] = groupByKey.try_emplace(key, groups.size());
      if (inserted) {
        groups.push_back({std::move(branch), std::move(domain)});
      } else {
        groups[found->second].sourceDomain.unionInPlace(domain);
        groups[found->second].sourceDomain =
            groups[found->second].sourceDomain.coalesce();
      }
    }
  }

  llvm::sort(groups, [](const auto &left, const auto &right) {
    return branchKey(left.branch) < branchKey(right.branch);
  });
  if (groups.empty())
    return std::vector<FabricMemoryServiceTargetPlan>{};

  const PresburgerSet empty =
      PresburgerSet::getEmpty(PresburgerSpace::getSetSpace(1));
  std::vector<PresburgerSet> suffix(groups.size() + 1, empty);
  for (std::size_t index = groups.size(); index-- != 0;)
    suffix[index] =
        suffix[index + 1].unionSet(groups[index].sourceDomain).coalesce();

  std::vector<FabricMemoryServiceTargetPlan> plans;
  std::vector<std::size_t> selected;
  std::function<void(std::size_t, PresburgerSet)> enumerate =
      [&](std::size_t index, PresburgerSet covered) {
        if (covered.isEqual(*sourceDomain)) {
          FabricMemoryServiceTargetPlan plan;
          plan.branches.reserve(selected.size());
          for (std::size_t selectedIndex : selected)
            plan.branches.push_back(groups[selectedIndex].branch);
          canonicalizePlan(plan);
          plans.push_back(std::move(plan));
          return;
        }
        if (index == groups.size())
          return;
        if (!sourceDomain->isSubsetOf(
                covered.unionSet(suffix[index]).coalesce()))
          return;

        enumerate(index + 1, covered);
        if (!covered.intersect(groups[index].sourceDomain).isIntegerEmpty())
          return;
        selected.push_back(index);
        enumerate(index + 1,
                  covered.unionSet(groups[index].sourceDomain).coalesce());
        selected.pop_back();
      };
  enumerate(0, empty);

  llvm::sort(plans, [](const auto &left, const auto &right) {
    return planKey(left) < planKey(right);
  });
  plans.erase(std::unique(plans.begin(), plans.end()), plans.end());
  return plans;
}

llvm::Expected<std::vector<FabricMemoryServiceTargetPlan>>
projectExactTargetPlans(const FabricSystemRootView &system,
                        SystemServiceEndpointRef endpoint,
                        FabricMemoryServiceSourceInterval sourceInterval) {
  auto identity = sourceIdentity(sourceInterval);
  if (!identity)
    return identity.takeError();
  FabricMemoryEndpointRef root{FabricMemoryEndpointOwnerRef::of(endpoint), 0};
  if (!system.artifact().memoryEndpointRole(root))
    return invalid("root endpoint has no Memory-plane endpoint zero");

  std::map<std::string, FabricMemoryEndpointRef> connections;
  for (const FabricMemoryServiceConnectionPayload &connection :
       system.artifact().memoryServiceConnections())
    connections.emplace(refKey(connection.source), connection.destination);

  std::vector<ExactFrontierBranch> frontier;
  const auto sourceEnd = llvm::checkedAddUnsigned(
      sourceInterval.addressBaseBytes, sourceInterval.sizeBytes);
  if (!sourceEnd)
    return invalid("source interval overflows u64");
  frontier.push_back(
      {root,
       {},
       std::move(*identity),
       std::nullopt,
       true,
       AddressEnvelope{sourceInterval.addressBaseBytes, *sourceEnd - 1}});
  const std::size_t transformLimit =
      system.artifact().systemServiceTransforms().size();
  while (!frontier.empty()) {
    for (ExactFrontierBranch &branch : frontier) {
      const auto role = system.artifact().memoryEndpointRole(branch.endpoint);
      if (!role)
        return invalid("exact closure contains an unknown memory endpoint");
      if (*role != FabricMemoryEndpointRole::Manager)
        continue;
      const auto found = connections.find(refKey(branch.endpoint));
      if (found == connections.end()) {
        branch.addresses = PresburgerRelation::getEmpty(
            PresburgerSpace::getRelationSpace(1, 1));
        branch.coversWholeSource = false;
        branch.envelope = std::nullopt;
      } else {
        branch.endpoint = found->second;
      }
    }
    frontier.erase(llvm::remove_if(frontier,
                                   [](const auto &branch) {
                                     return !branch.coversWholeSource &&
                                            branch.addresses.isIntegerEmpty();
                                   }),
                   frontier.end());
    if (frontier.empty())
      return std::vector<FabricMemoryServiceTargetPlan>{};

    bool allTerminal = true;
    std::vector<ExactFrontierBranch> next;
    for (ExactFrontierBranch &branch : frontier) {
      if (system.artifact().memoryEndpointRole(branch.endpoint) !=
          FabricMemoryEndpointRole::Subordinate)
        return invalid("exact closure continuation is not subordinate");
      FrontierBranch structural{branch.endpoint, branch.path, std::nullopt};
      auto terminal = terminalBranch(system, structural);
      if (!terminal)
        return terminal.takeError();
      if (*terminal) {
        next.push_back(std::move(branch));
        continue;
      }
      allTerminal = false;
      if (branch.requiredTerminalRegion)
        continue;
      auto transform = endpointTransform(system, branch.endpoint);
      if (!transform)
        return transform.takeError();
      if (!*transform)
        return invalid("exact closure endpoint has no transform owner");
      if (branch.path.size() >= transformLimit ||
          llvm::is_contained(branch.path, **transform))
        continue;
      const SystemServiceTransformRecord *record =
          system.serviceTransform(**transform);
      if (!record)
        return invalid("exact closure transform has no canonical contract");
      if (!llvm::is_contained(record->inputs(), branch.endpoint))
        return invalid("exact closure enters a transform through a non-input");
      std::vector<SystemServiceTransformRef> path = branch.path;
      path.push_back(**transform);

      const auto append =
          [&](FabricMemoryEndpointRef output, PresburgerRelation addresses,
              bool coversWholeSource, std::optional<AddressEnvelope> envelope,
              std::optional<FabricMemoryServiceRegionRef> requiredRegion) {
            if (coversWholeSource || !addresses.isIntegerEmpty())
              next.push_back({output, path, std::move(addresses),
                              requiredRegion, coversWholeSource, envelope});
          };
      if (const auto *offset =
              std::get_if<AddressOffsetTransform>(&record->contract())) {
        const auto envelope = branch.envelope
                                  ? offsetEnvelope(*branch.envelope, *offset)
                                  : std::nullopt;
        append(record->outputs().front(),
               compose(branch.addresses, addressOffsetRelation(*offset)),
               branch.coversWholeSource && envelope.has_value(), envelope,
               std::nullopt);
      } else if (const auto *mask = std::get_if<AddressMaskXorTransform>(
                     &record->contract())) {
        const auto envelope = branch.envelope
                                  ? maskEnvelope(*branch.envelope, *mask)
                                  : std::nullopt;
        append(record->outputs().front(),
               compose(branch.addresses, addressMaskXorRelation(*mask)),
               branch.coversWholeSource && envelope.has_value(), envelope,
               std::nullopt);
      } else if (const auto *interleave =
                     std::get_if<StaticInterleaveTransform>(
                         &record->contract())) {
        for (auto [ordinal, output] : llvm::enumerate(record->outputs()))
          append(output,
                 compose(branch.addresses,
                         staticInterleaveRelation(*interleave, ordinal)),
                 false, std::nullopt, std::nullopt);
      } else {
        const auto &coherent =
            std::get<CoherentMemoryTransform>(record->contract());
        for (const auto &correspondence : coherent.regions) {
          auto relation = coherentRegionRelation(system, correspondence);
          if (!relation)
            return relation.takeError();
          PresburgerRelation addresses = compose(branch.addresses, *relation);
          std::optional<AddressEnvelope> envelope;
          if (branch.coversWholeSource && branch.envelope) {
            auto input = regionDeclaration(system, correspondence.input);
            if (!input)
              return input.takeError();
            auto output = regionDeclaration(system, correspondence.output);
            if (!output)
              return output.takeError();
            if (contains(**input, *branch.envelope)) {
              const auto lower = llvm::checkedAddUnsigned(
                  (*output)->addressBaseBytes,
                  branch.envelope->lower - (*input)->addressBaseBytes);
              const auto upper = llvm::checkedAddUnsigned(
                  (*output)->addressBaseBytes,
                  branch.envelope->upper - (*input)->addressBaseBytes);
              if (lower && upper)
                envelope = AddressEnvelope{*lower, *upper};
            }
          }
          for (FabricMemoryEndpointRef output : record->outputs())
            append(output, addresses,
                   branch.coversWholeSource && envelope.has_value(), envelope,
                   correspondence.output);
        }
      }
    }
    if (allTerminal)
      return exactTerminalPlans(system, next, sourceInterval);
    frontier = std::move(next);
  }
  return std::vector<FabricMemoryServiceTargetPlan>{};
}

} // namespace

llvm::Expected<std::vector<FabricMemoryServiceTargetPlan>>
projectFabricMemoryServiceTargetPlans(
    const FabricSystemRootView &system, SystemServiceEndpointRef endpoint,
    FabricMemoryServiceSourceInterval sourceInterval) {
  return projectExactTargetPlans(system, endpoint, sourceInterval);
}

} // namespace loom::fabric
