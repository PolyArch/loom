#include "MappingConstraintCanonicalization.h"

#include "Mapping/IR/MappingAttrs.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;

namespace loom::mapping::detail {
namespace {

enum class ClauseKind : std::uint32_t {
  DomainRestriction = 0,
  Equal = 1,
  Disjoint = 2,
};

void appendU32Be(std::string &output, std::uint32_t value) {
  for (unsigned byte = 0; byte < 4; ++byte)
    output.push_back(static_cast<char>(value >> (8 * (3 - byte))));
}

void appendU64Be(std::string &output, std::uint64_t value) {
  for (unsigned byte = 0; byte < 8; ++byte)
    output.push_back(static_cast<char>(value >> (8 * (7 - byte))));
}

void appendFramed(std::string &output, llvm::StringRef value) {
  appendU32Be(output, value.size());
  output.append(value.data(), value.size());
}

std::vector<std::uint8_t> unsignedBytes(DenseI8ArrayAttr record) {
  std::vector<std::uint8_t> result;
  result.reserve(record.size());
  for (std::int8_t byte : record.asArrayRef())
    result.push_back(static_cast<std::uint8_t>(byte));
  return result;
}

std::string integerKey(const llvm::APInt &value) {
  const unsigned byteCount = std::max(1u, (value.getActiveBits() + 7) / 8);
  const llvm::APInt extended = value.zextOrTrunc(byteCount * 8);
  std::string result;
  result.reserve(byteCount);
  for (unsigned byte = 0; byte < byteCount; ++byte)
    result.push_back(static_cast<char>(
        extended.extractBitsAsZExtValue(8, 8 * (byteCount - 1 - byte))));
  return result;
}

DenseI8ArrayAttr canonicalRecord(Attribute attribute) {
  return llvm::TypeSwitch<Attribute, DenseI8ArrayAttr>(attribute)
      .Case<::mapping::ActorRefAttr, ::mapping::LogicalMemoryRootRefAttr,
            ::mapping::GraphProducerEndpointRefAttr,
            ::mapping::GraphConsumerEndpointRefAttr,
            ::mapping::RootThreadLaunchRefAttr,
            ::mapping::RootedGraphLaunchRefAttr,
            ::mapping::SystemServiceObligationKeyAttr,
            ::mapping::CanonicalServiceLegKeyAttr,
            ::mapping::SystemTransferTerminalKeyAttr,
            ::mapping::FabricFuOccurrenceRefAttr,
            ::mapping::FabricAccCoreOccurrenceRefAttr,
            ::mapping::FabricSpatialCoreOccurrenceRefAttr,
            ::mapping::FabricPeOccurrenceRefAttr,
            ::mapping::InstructionContextRefAttr,
            ::mapping::FabricMemoryOccurrenceRefAttr,
            ::mapping::FabricPhysicalTraversalRefAttr,
            ::mapping::FabricResourceStateRefAttr,
            ::mapping::FabricTransportEndpointRefAttr,
            ::mapping::FabricMemoryOperationPortRefAttr,
            ::mapping::FabricMemoryServiceRefAttr,
            ::mapping::FabricMemoryServiceRegionRefAttr>(
          [](auto value) { return value.getRecord(); })
      .Default(DenseI8ArrayAttr());
}

std::uint32_t projectionOrdinal(Attribute projection) {
  if (auto spatial =
          dyn_cast<::mapping::SpatialConstraintProjectionKeyAttr>(projection))
    return spatial.getValue();
  return cast<::mapping::SystemConstraintProjectionKeyAttr>(projection)
      .getValue();
}

struct UnsignedRange final {
  llvm::APInt lower;
  llvm::APInt upper;
};

int compareUnsigned(const llvm::APInt &lhs, const llvm::APInt &rhs) {
  const unsigned width = std::max(lhs.getBitWidth(), rhs.getBitWidth());
  const llvm::APInt left = lhs.zext(width);
  const llvm::APInt right = rhs.zext(width);
  if (left.ult(right))
    return -1;
  if (right.ult(left))
    return 1;
  return 0;
}

llvm::APInt maximumUnsigned(const llvm::APInt &lhs, const llvm::APInt &rhs) {
  const unsigned width = std::max(lhs.getBitWidth(), rhs.getBitWidth());
  llvm::APInt left = lhs.zext(width);
  llvm::APInt right = rhs.zext(width);
  return left.ult(right) ? right : left;
}

UnsignedRange intervalRange(::mapping::ConstraintUnsignedIntervalAttr value) {
  return {value.getLower().getValue(), value.getUpper().getValue()};
}

std::vector<UnsignedRange> normalizeRanges(ArrayRef<Attribute> values) {
  std::vector<UnsignedRange> ranges;
  ranges.reserve(values.size());
  for (Attribute value : values)
    ranges.push_back(
        intervalRange(cast<::mapping::ConstraintUnsignedIntervalAttr>(value)));
  llvm::sort(ranges, [](const UnsignedRange &lhs, const UnsignedRange &rhs) {
    const int lower = compareUnsigned(lhs.lower, rhs.lower);
    return lower != 0 ? lower < 0 : compareUnsigned(lhs.upper, rhs.upper) < 0;
  });

  std::vector<UnsignedRange> result;
  for (UnsignedRange &range : ranges) {
    if (result.empty() ||
        compareUnsigned(result.back().upper, range.lower) < 0) {
      result.push_back(std::move(range));
      continue;
    }
    result.back().upper = maximumUnsigned(result.back().upper, range.upper);
  }
  return result;
}

std::vector<UnsignedRange> intersectRanges(ArrayRef<UnsignedRange> lhs,
                                           ArrayRef<UnsignedRange> rhs) {
  std::vector<UnsignedRange> result;
  std::size_t left = 0;
  std::size_t right = 0;
  while (left < lhs.size() && right < rhs.size()) {
    const llvm::APInt lower =
        maximumUnsigned(lhs[left].lower, rhs[right].lower);
    const llvm::APInt upper =
        compareUnsigned(lhs[left].upper, rhs[right].upper) < 0
            ? lhs[left].upper
            : rhs[right].upper;
    if (compareUnsigned(lower, upper) < 0)
      result.push_back({lower, upper});
    const int endOrder = compareUnsigned(lhs[left].upper, rhs[right].upper);
    if (endOrder <= 0)
      ++left;
    if (endOrder >= 0)
      ++right;
  }
  return result;
}

IntegerAttr canonicalUnsignedInteger(MLIRContext *context,
                                     const llvm::APInt &value) {
  const unsigned width = std::max(1u, value.getActiveBits());
  return IntegerAttr::get(
      IntegerType::get(context, width, IntegerType::Unsigned),
      value.zextOrTrunc(width));
}

std::vector<Attribute> rangeAttributes(MLIRContext *context,
                                       ArrayRef<UnsignedRange> ranges) {
  std::vector<Attribute> result;
  result.reserve(ranges.size());
  for (const UnsignedRange &range : ranges)
    result.push_back(::mapping::ConstraintUnsignedIntervalAttr::get(
        context, canonicalUnsignedInteger(context, range.lower),
        canonicalUnsignedInteger(context, range.upper)));
  return result;
}

struct UnionFind final {
  void insert(llvm::StringRef key) { parent.try_emplace(key.str(), key.str()); }

  std::string find(llvm::StringRef key) {
    auto found = parent.find(key.str());
    if (found == parent.end()) {
      insert(key);
      found = parent.find(key.str());
    }
    if (found->second == found->first)
      return found->first;
    found->second = find(found->second);
    return found->second;
  }

  void unite(llvm::StringRef lhs, llvm::StringRef rhs) {
    const std::string left = find(lhs);
    const std::string right = find(rhs);
    if (left == right)
      return;
    const std::string &representative = std::min(left, right);
    const std::string &other = representative == left ? right : left;
    parent[other] = representative;
  }

  std::map<std::string, std::string> parent;
};

struct ProjectionClauses final {
  Attribute projection;
  UnionFind equality;
  std::map<std::string, Attribute> subjects;
  std::vector<std::pair<std::string, ArrayAttr>> restrictions;
  std::vector<std::vector<std::string>> disjoint;
};

struct CanonicalClause final {
  ClauseKind kind;
  Attribute projection;
  Attribute subject;
  ArrayAttr values;

  std::string key() const {
    std::string result;
    appendU32Be(result, static_cast<std::uint32_t>(kind));
    appendU32Be(result, projectionOrdinal(projection));
    if (subject)
      result += constraintAttributeKey(subject);
    result.push_back('\0');
    for (Attribute value : values) {
      result += constraintAttributeKey(value);
      result.push_back('\0');
    }
    return result;
  }
};

Attribute constraintProjection(Operation *operation) {
  return operation->getAttr("projection");
}

void rememberSubject(ProjectionClauses &clauses, Attribute subject) {
  const std::string key = constraintAttributeKey(subject);
  clauses.equality.insert(key);
  clauses.subjects.try_emplace(key, subject);
}

ProjectionClauses &
projectionState(std::map<std::uint32_t, ProjectionClauses> &states,
                Attribute projection) {
  const std::uint32_t ordinal = projectionOrdinal(projection);
  auto [position, inserted] = states.try_emplace(
      ordinal, ProjectionClauses{projection, {}, {}, {}, {}});
  (void)inserted;
  return position->second;
}

std::vector<CanonicalClause>
canonicalizeClauses(Block &body, ConstraintDomainTransform normalizeDomain,
                    ConstraintDomainIntersection intersectDomains) {
  std::map<std::uint32_t, ProjectionClauses> states;
  for (Operation &operation : body) {
    if (auto equal = dyn_cast<::mapping::ConstraintEqualOp>(operation)) {
      ProjectionClauses &state =
          projectionState(states, constraintProjection(equal));
      std::vector<std::string> keys;
      for (Attribute subject : equal.getSubjects()) {
        rememberSubject(state, subject);
        keys.push_back(constraintAttributeKey(subject));
      }
      llvm::sort(keys);
      keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
      for (std::size_t index = 1; index < keys.size(); ++index)
        state.equality.unite(keys.front(), keys[index]);
      continue;
    }
    if (auto restriction =
            dyn_cast<::mapping::ConstraintDomainRestrictionOp>(operation)) {
      ProjectionClauses &state =
          projectionState(states, constraintProjection(restriction));
      rememberSubject(state, restriction.getSubject());
      state.restrictions.emplace_back(
          constraintAttributeKey(restriction.getSubject()),
          restriction.getAdmissibleDomain());
      continue;
    }
    auto disjoint = cast<::mapping::ConstraintDisjointOp>(operation);
    ProjectionClauses &state =
        projectionState(states, constraintProjection(disjoint));
    std::vector<std::string> keys;
    for (Attribute subject : disjoint.getSubjects()) {
      rememberSubject(state, subject);
      keys.push_back(constraintAttributeKey(subject));
    }
    llvm::sort(keys);
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    if (keys.size() >= 2)
      state.disjoint.push_back(std::move(keys));
  }

  std::vector<CanonicalClause> result;
  MLIRContext *context = body.getParentOp()->getContext();
  for (auto &[ordinal, state] : states) {
    (void)ordinal;
    std::map<std::string, std::vector<std::string>> classes;
    for (const auto &[key, subject] : state.subjects) {
      (void)subject;
      classes[state.equality.find(key)].push_back(key);
    }
    for (auto &[representative, members] : classes) {
      (void)representative;
      llvm::sort(members);
      if (members.size() < 2)
        continue;
      SmallVector<Attribute> subjects;
      for (const std::string &member : members)
        subjects.push_back(state.subjects.at(member));
      result.push_back({ClauseKind::Equal, state.projection, Attribute(),
                        ArrayAttr::get(context, subjects)});
    }

    std::map<std::string, std::vector<Attribute>> domains;
    for (const auto &[subject, authoredDomain] : state.restrictions) {
      const std::string representative = state.equality.find(subject);
      std::vector<Attribute> normalized =
          normalizeDomain(context, state.projection, authoredDomain.getValue());
      auto [position, inserted] =
          domains.try_emplace(representative, normalized);
      if (!inserted)
        position->second = intersectDomains(context, state.projection,
                                            position->second, normalized);
    }

    std::set<std::vector<std::string>> disjointGroups;
    for (const std::vector<std::string> &authored : state.disjoint) {
      std::vector<std::string> representatives;
      representatives.reserve(authored.size());
      for (const std::string &subject : authored)
        representatives.push_back(state.equality.find(subject));
      std::vector<std::string> sorted = representatives;
      llvm::sort(sorted);
      for (std::size_t index = 1; index < sorted.size(); ++index)
        if (sorted[index - 1] == sorted[index])
          domains[sorted[index]].clear();

      llvm::sort(representatives);
      representatives.erase(
          std::unique(representatives.begin(), representatives.end()),
          representatives.end());
      representatives.erase(
          std::remove_if(representatives.begin(), representatives.end(),
                         [&](const std::string &representative) {
                           auto domain = domains.find(representative);
                           return domain != domains.end() &&
                                  domain->second.empty();
                         }),
          representatives.end());
      if (representatives.size() >= 2)
        disjointGroups.insert(std::move(representatives));
    }

    for (const auto &[representative, domain] : domains) {
      SmallVector<Attribute> values(domain.begin(), domain.end());
      result.push_back({ClauseKind::DomainRestriction, state.projection,
                        state.subjects.at(representative),
                        ArrayAttr::get(context, values)});
    }
    for (const std::vector<std::string> &group : disjointGroups) {
      SmallVector<Attribute> subjects;
      for (const std::string &representative : group)
        subjects.push_back(state.subjects.at(representative));
      result.push_back({ClauseKind::Disjoint, state.projection, Attribute(),
                        ArrayAttr::get(context, subjects)});
    }
  }

  llvm::sort(result,
             [](const CanonicalClause &lhs, const CanonicalClause &rhs) {
               return lhs.key() < rhs.key();
             });
  return result;
}

Operation *createClause(OpBuilder &builder, Location location,
                        const CanonicalClause &clause) {
  StringRef operationName;
  switch (clause.kind) {
  case ClauseKind::DomainRestriction:
    operationName =
        ::mapping::ConstraintDomainRestrictionOp::getOperationName();
    break;
  case ClauseKind::Equal:
    operationName = ::mapping::ConstraintEqualOp::getOperationName();
    break;
  case ClauseKind::Disjoint:
    operationName = ::mapping::ConstraintDisjointOp::getOperationName();
    break;
  }
  OperationState state(location, operationName);
  state.addAttribute("projection", clause.projection);
  if (clause.kind == ClauseKind::DomainRestriction) {
    state.addAttribute("subject", clause.subject);
    state.addAttribute("admissible_domain", clause.values);
  } else {
    state.addAttribute("subjects", clause.values);
  }
  return builder.create(state);
}

} // namespace

std::string constraintAttributeKey(Attribute attribute) {
  if (DenseI8ArrayAttr record = canonicalRecord(attribute)) {
    std::vector<std::uint8_t> bytes = unsignedBytes(record);
    return std::string(reinterpret_cast<const char *>(bytes.data()),
                       bytes.size());
  }
  if (auto value = dyn_cast<::mapping::ComputeRealizationRefAttr>(attribute)) {
    std::string result;
    appendU64Be(result, value.getEntity());
    return result;
  }
  if (auto value = dyn_cast<::mapping::MemoryRealizationRefAttr>(attribute)) {
    std::string result;
    appendU64Be(result, value.getEntity());
    return result;
  }
  if (auto value = dyn_cast<::mapping::ConstraintSpatialMappingReferenceAttr>(
          attribute)) {
    std::string result;
    appendU64Be(result, value.getOrdinal());
    return result;
  }
  if (auto terminal =
          dyn_cast<::mapping::SpatialTransferTerminalAttr>(attribute)) {
    std::string result;
    appendU32Be(result, terminal.getConsumer() ? 1 : 0);
    appendFramed(result, constraintAttributeKey(terminal.getProducer()));
    if (terminal.getConsumer())
      appendFramed(result, constraintAttributeKey(terminal.getConsumer()));
    return result;
  }
  if (auto tuple = dyn_cast<::mapping::ConstraintFuContextAttr>(attribute)) {
    std::string result;
    appendFramed(result, constraintAttributeKey(tuple.getFu()));
    appendFramed(result, constraintAttributeKey(tuple.getInstructionContext()));
    return result;
  }
  if (auto interval =
          dyn_cast<::mapping::ConstraintUnsignedIntervalAttr>(attribute)) {
    std::string result;
    appendFramed(result, integerKey(interval.getLower().getValue()));
    appendFramed(result, integerKey(interval.getUpper().getValue()));
    return result;
  }
  if (auto region =
          dyn_cast<::mapping::ConstraintAddressRegionAttr>(attribute)) {
    std::string result;
    appendFramed(result, constraintAttributeKey(region.getService()));
    appendU32Be(result, region.getIntervals().size());
    for (Attribute interval : region.getIntervals())
      appendFramed(result, constraintAttributeKey(interval));
    return result;
  }
  llvm_unreachable("unexpected MappingConstraintSet attribute");
}

std::vector<Attribute>
normalizeExactConstraintDomain(ArrayRef<Attribute> values) {
  std::vector<Attribute> result(values.begin(), values.end());
  llvm::sort(result, [](Attribute lhs, Attribute rhs) {
    return constraintAttributeKey(lhs) < constraintAttributeKey(rhs);
  });
  result.erase(std::unique(result.begin(), result.end(),
                           [](Attribute lhs, Attribute rhs) {
                             return constraintAttributeKey(lhs) ==
                                    constraintAttributeKey(rhs);
                           }),
               result.end());
  return result;
}

std::vector<Attribute>
intersectExactConstraintDomains(ArrayRef<Attribute> lhs,
                                ArrayRef<Attribute> rhs) {
  std::vector<Attribute> result;
  std::set_intersection(
      lhs.begin(), lhs.end(), rhs.begin(), rhs.end(),
      std::back_inserter(result), [](Attribute left, Attribute right) {
        return constraintAttributeKey(left) < constraintAttributeKey(right);
      });
  return result;
}

std::vector<Attribute>
normalizeUnsignedIntervalConstraintDomain(MLIRContext *context,
                                          ArrayRef<Attribute> values) {
  return rangeAttributes(context, normalizeRanges(values));
}

std::vector<Attribute> intersectUnsignedIntervalConstraintDomains(
    MLIRContext *context, ArrayRef<Attribute> lhs, ArrayRef<Attribute> rhs) {
  const std::vector<UnsignedRange> left = normalizeRanges(lhs);
  const std::vector<UnsignedRange> right = normalizeRanges(rhs);
  return rangeAttributes(context, intersectRanges(left, right));
}

void canonicalizeConstraintClauses(
    Block &body, Location location, ConstraintDomainTransform normalizeDomain,
    ConstraintDomainIntersection intersectDomains) {
  std::vector<CanonicalClause> clauses =
      canonicalizeClauses(body, normalizeDomain, intersectDomains);
  while (!body.empty())
    body.front().erase();
  OpBuilder builder(body.getParentOp()->getContext());
  builder.setInsertionPointToEnd(&body);
  for (const CanonicalClause &clause : clauses)
    createClause(builder, location, clause);
}

} // namespace loom::mapping::detail
