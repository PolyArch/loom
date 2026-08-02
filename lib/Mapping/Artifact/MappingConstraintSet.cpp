#include "Mapping/Artifact/MappingConstraintSet.h"

#include "Common/ArtifactFinalizer.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/IR/MappingAttrs.h"
#include "Mapping/IR/MappingDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;

namespace loom::mapping {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "mapping_constraint_set_invalid: " + message);
}

std::vector<std::uint8_t> unsignedBytes(DenseI8ArrayAttr record) {
  std::vector<std::uint8_t> result;
  result.reserve(record.size());
  for (std::int8_t byte : record.asArrayRef())
    result.push_back(static_cast<std::uint8_t>(byte));
  return result;
}

llvm::Expected<ArtifactIdentity>
decodeIdentity(::mapping::ArtifactIdentityAttr attribute) {
  return ArtifactIdentity::fromBytes(unsignedBytes(attribute.getRecord()));
}

template <typename Ref, typename Attr>
llvm::Expected<Ref> decodeDataflow(Attr attribute,
                                   const ArtifactIdentity &owner) {
  return ::dataflow::decodeDataflowReference<Ref>(
      unsignedBytes(attribute.getRecord()), owner);
}

template <typename Ref, typename Attr>
llvm::Expected<Ref> decodeFabric(Attr attribute) {
  return ::loom::fabric::decodeFabricRef<Ref>(
      unsignedBytes(attribute.getRecord()));
}

template <typename T>
llvm::Expected<T> contextual(llvm::Expected<T> value,
                             const llvm::Twine &context) {
  if (!value)
    return llvm::joinErrors(invalid(context), value.takeError());
  return std::move(*value);
}

llvm::Error contextual(llvm::Error error, const llvm::Twine &context) {
  if (!error)
    return llvm::Error::success();
  return llvm::joinErrors(invalid(context), std::move(error));
}

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
            ::mapping::FabricFuOccurrenceRefAttr,
            ::mapping::FabricPeOccurrenceRefAttr,
            ::mapping::InstructionContextRefAttr,
            ::mapping::FabricMemoryOccurrenceRefAttr,
            ::mapping::FabricPhysicalTraversalRefAttr,
            ::mapping::FabricResourceStateRefAttr,
            ::mapping::FabricTransportEndpointRefAttr,
            ::mapping::FabricMemoryOperationPortRefAttr,
            ::mapping::FabricMemoryServiceRefAttr>(
          [](auto value) { return value.getRecord(); })
      .Default(DenseI8ArrayAttr());
}

std::string attributeKey(Attribute attribute) {
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
  if (auto terminal =
          dyn_cast<::mapping::SpatialTransferTerminalAttr>(attribute)) {
    std::string result;
    appendU32Be(result, terminal.getConsumer() ? 1 : 0);
    appendFramed(result, attributeKey(terminal.getProducer()));
    if (terminal.getConsumer())
      appendFramed(result, attributeKey(terminal.getConsumer()));
    return result;
  }
  if (auto tuple = dyn_cast<::mapping::ConstraintFuContextAttr>(attribute)) {
    std::string result;
    appendFramed(result, attributeKey(tuple.getFu()));
    appendFramed(result, attributeKey(tuple.getInstructionContext()));
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
    appendFramed(result, attributeKey(region.getService()));
    appendU32Be(result, region.getIntervals().size());
    for (Attribute interval : region.getIntervals())
      appendFramed(result, attributeKey(interval));
    return result;
  }
  llvm_unreachable("unexpected Spatial MappingConstraintSet attribute");
}

std::vector<Attribute> normalizeExactSet(ArrayRef<Attribute> values) {
  std::vector<Attribute> result(values.begin(), values.end());
  llvm::sort(result, [](Attribute lhs, Attribute rhs) {
    return attributeKey(lhs) < attributeKey(rhs);
  });
  result.erase(std::unique(result.begin(), result.end(),
                           [](Attribute lhs, Attribute rhs) {
                             return attributeKey(lhs) == attributeKey(rhs);
                           }),
               result.end());
  return result;
}

std::vector<Attribute> intersectExactSets(ArrayRef<Attribute> lhs,
                                          ArrayRef<Attribute> rhs) {
  std::vector<Attribute> result;
  std::set_intersection(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(),
                        std::back_inserter(result),
                        [](Attribute left, Attribute right) {
                          return attributeKey(left) < attributeKey(right);
                        });
  return result;
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

bool isIntervalProjection(::mapping::SpatialConstraintProjection projection) {
  return projection ==
         ::mapping::SpatialConstraintProjection::NetAssignedTagValues;
}

std::vector<Attribute> normalizeAddressRegions(MLIRContext *context,
                                               ArrayRef<Attribute> values) {
  struct ServiceRanges final {
    ::mapping::FabricMemoryServiceRefAttr service;
    std::vector<Attribute> intervals;
  };
  std::map<std::string, ServiceRanges> byService;
  for (Attribute value : values) {
    auto region = cast<::mapping::ConstraintAddressRegionAttr>(value);
    const std::string key = attributeKey(region.getService());
    auto [position, inserted] =
        byService.try_emplace(key, ServiceRanges{region.getService(), {}});
    (void)inserted;
    position->second.intervals.insert(position->second.intervals.end(),
                                      region.getIntervals().begin(),
                                      region.getIntervals().end());
  }

  std::vector<Attribute> result;
  for (auto &[key, service] : byService) {
    (void)key;
    std::vector<UnsignedRange> ranges = normalizeRanges(service.intervals);
    if (ranges.empty())
      continue;
    std::vector<Attribute> intervals = rangeAttributes(context, ranges);
    result.push_back(::mapping::ConstraintAddressRegionAttr::get(
        context, service.service, ArrayAttr::get(context, intervals)));
  }
  return result;
}

std::vector<Attribute>
normalizeDomain(MLIRContext *context,
                ::mapping::SpatialConstraintProjection projection,
                ArrayRef<Attribute> values) {
  if (isIntervalProjection(projection))
    return rangeAttributes(context, normalizeRanges(values));
  if (projection == ::mapping::SpatialConstraintProjection::MemoryAddressRegion)
    return normalizeAddressRegions(context, values);
  return normalizeExactSet(values);
}

std::vector<Attribute>
intersectDomains(MLIRContext *context,
                 ::mapping::SpatialConstraintProjection projection,
                 ArrayRef<Attribute> lhs, ArrayRef<Attribute> rhs) {
  if (isIntervalProjection(projection)) {
    const std::vector<UnsignedRange> left = normalizeRanges(lhs);
    const std::vector<UnsignedRange> right = normalizeRanges(rhs);
    return rangeAttributes(context, intersectRanges(left, right));
  }
  if (projection ==
      ::mapping::SpatialConstraintProjection::MemoryAddressRegion) {
    std::map<std::string, ::mapping::ConstraintAddressRegionAttr>
        rightByService;
    for (Attribute value : normalizeAddressRegions(context, rhs)) {
      auto region = cast<::mapping::ConstraintAddressRegionAttr>(value);
      rightByService.emplace(attributeKey(region.getService()), region);
    }
    std::vector<Attribute> result;
    for (Attribute value : normalizeAddressRegions(context, lhs)) {
      auto leftRegion = cast<::mapping::ConstraintAddressRegionAttr>(value);
      auto rightRegion =
          rightByService.find(attributeKey(leftRegion.getService()));
      if (rightRegion == rightByService.end())
        continue;
      const std::vector<UnsignedRange> leftRanges =
          normalizeRanges(leftRegion.getIntervals());
      const std::vector<UnsignedRange> rightRanges =
          normalizeRanges(rightRegion->second.getIntervals());
      std::vector<Attribute> intervals =
          rangeAttributes(context, intersectRanges(leftRanges, rightRanges));
      if (!intervals.empty())
        result.push_back(::mapping::ConstraintAddressRegionAttr::get(
            context, leftRegion.getService(),
            ArrayAttr::get(context, intervals)));
    }
    return result;
  }
  return intersectExactSets(lhs, rhs);
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
  ::mapping::SpatialConstraintProjection projection;
  ::mapping::SpatialConstraintProjectionAttr projectionAttr;
  UnionFind equality;
  std::map<std::string, Attribute> subjects;
  std::vector<std::pair<std::string, ArrayAttr>> restrictions;
  std::vector<std::vector<std::string>> disjoint;
};

struct CanonicalClause final {
  ClauseKind kind;
  ::mapping::SpatialConstraintProjectionAttr projection;
  Attribute subject;
  ArrayAttr values;

  std::string key() const {
    std::string result;
    appendU32Be(result, static_cast<std::uint32_t>(kind));
    appendU32Be(result, static_cast<std::uint32_t>(projection.getValue()));
    if (subject)
      result += attributeKey(subject);
    result.push_back('\0');
    for (Attribute value : values) {
      result += attributeKey(value);
      result.push_back('\0');
    }
    return result;
  }
};

void rememberSubject(ProjectionClauses &clauses, Attribute subject) {
  const std::string key = attributeKey(subject);
  clauses.equality.insert(key);
  clauses.subjects.try_emplace(key, subject);
}

ProjectionClauses &
projectionState(std::map<std::uint32_t, ProjectionClauses> &states,
                ::mapping::SpatialConstraintProjectionAttr projection) {
  const auto ordinal = static_cast<std::uint32_t>(projection.getValue());
  auto [position, inserted] = states.try_emplace(
      ordinal,
      ProjectionClauses{projection.getValue(), projection, {}, {}, {}, {}});
  (void)inserted;
  return position->second;
}

llvm::Expected<std::vector<CanonicalClause>>
canonicalizeClauses(::mapping::ConstraintsSpatialOp root) {
  std::map<std::uint32_t, ProjectionClauses> states;
  for (Operation &operation : root.getBody().front()) {
    if (auto equal = dyn_cast<::mapping::ConstraintEqualOp>(operation)) {
      ProjectionClauses &state =
          projectionState(states, equal.getProjectionAttr());
      std::vector<std::string> keys;
      for (Attribute subject : equal.getSubjects()) {
        rememberSubject(state, subject);
        keys.push_back(attributeKey(subject));
      }
      llvm::sort(keys);
      keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
      if (keys.size() >= 2)
        for (std::size_t index = 1; index < keys.size(); ++index)
          state.equality.unite(keys.front(), keys[index]);
      continue;
    }
    if (auto restriction =
            dyn_cast<::mapping::ConstraintDomainRestrictionOp>(operation)) {
      ProjectionClauses &state =
          projectionState(states, restriction.getProjectionAttr());
      rememberSubject(state, restriction.getSubject());
      state.restrictions.emplace_back(attributeKey(restriction.getSubject()),
                                      restriction.getAdmissibleDomain());
      continue;
    }
    auto disjoint = cast<::mapping::ConstraintDisjointOp>(operation);
    ProjectionClauses &state =
        projectionState(states, disjoint.getProjectionAttr());
    std::vector<std::string> keys;
    for (Attribute subject : disjoint.getSubjects()) {
      rememberSubject(state, subject);
      keys.push_back(attributeKey(subject));
    }
    llvm::sort(keys);
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    if (keys.size() >= 2)
      state.disjoint.push_back(std::move(keys));
  }

  std::vector<CanonicalClause> result;
  MLIRContext *context = root.getContext();
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
      result.push_back({ClauseKind::Equal, state.projectionAttr, Attribute(),
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
      result.push_back({ClauseKind::DomainRestriction, state.projectionAttr,
                        state.subjects.at(representative),
                        ArrayAttr::get(context, values)});
    }
    for (const std::vector<std::string> &group : disjointGroups) {
      SmallVector<Attribute> subjects;
      for (const std::string &representative : group)
        subjects.push_back(state.subjects.at(representative));
      result.push_back({ClauseKind::Disjoint, state.projectionAttr, Attribute(),
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

struct ParsedSpatialConstraintRoot final {
  std::unique_ptr<MLIRContext> context;
  OwningOpRef<ModuleOp> module;
  ::mapping::ConstraintsSpatialOp root;
};

llvm::Expected<ParsedSpatialConstraintRoot>
parseSpatialConstraintRoot(const CanonicalSemanticBytes &canonicalBytes) {
  std::string wrapped = "module {\n";
  wrapped.append(reinterpret_cast<const char *>(canonicalBytes.bytes().data()),
                 canonicalBytes.bytes().size());
  wrapped += "}\n";

  DialectRegistry registry;
  registry.insert<::mapping::MappingDialect>();
  auto context =
      std::make_unique<MLIRContext>(registry, MLIRContext::Threading::DISABLED);
  auto module = parseSourceString<ModuleOp>(wrapped, context.get());
  if (!module)
    return invalid("canonical Spatial constraint payload cannot be parsed");

  ::mapping::ConstraintsSpatialOp root;
  unsigned rootCount = 0;
  for (Operation &operation : module->getBody()->without_terminator()) {
    auto candidate = dyn_cast<::mapping::ConstraintsSpatialOp>(operation);
    if (!candidate)
      return invalid("constraint artifact contains a non-Spatial root");
    root = candidate;
    ++rootCount;
  }
  if (rootCount != 1)
    return invalid("constraint artifact must contain exactly one Spatial root");
  if (failed(verify(root)))
    return invalid("Spatial constraint root is structurally invalid");
  return ParsedSpatialConstraintRoot{std::move(context), std::move(module),
                                     root};
}

bool hasComputeRealization(const TechMappingView &techMapping,
                           std::uint64_t entity) {
  return llvm::any_of(techMapping.computeRealizations(),
                      [&](const TechComputeRealizationView &realization) {
                        return realization.entityId == entity;
                      });
}

bool hasMemoryRealization(const TechMappingView &techMapping,
                          std::uint64_t entity) {
  return llvm::any_of(techMapping.memoryRealizations(),
                      [&](const TechMemoryRealizationView &realization) {
                        return realization.entityId == entity;
                      });
}

llvm::Error validateResidualProducer(
    const ::dataflow::CanonicalGraphProducerEndpointRef &producer,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping) {
  if (llvm::Error error = dataflow.validate(producer))
    return contextual(std::move(error),
                      "constraint producer endpoint does not resolve");
  if (!techMapping.residualLogicalNet(producer))
    return invalid("constraint producer has no residual logical net");
  return llvm::Error::success();
}

llvm::Error validateResidualSink(
    const ::dataflow::CanonicalGraphProducerEndpointRef &producer,
    const ::dataflow::CanonicalGraphConsumerEndpointRef &consumer,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping) {
  if (llvm::Error error = dataflow.validate(consumer))
    return contextual(std::move(error),
                      "constraint consumer endpoint does not resolve");
  auto consumers = dataflow.graphConsumers(producer);
  if (!consumers)
    return contextual(consumers.takeError(),
                      "constraint producer relation does not resolve");
  if (llvm::find(*consumers, consumer) == consumers->end())
    return invalid("constraint transfer sink is not fed by its producer");
  const TechResidualLogicalNetView *net =
      techMapping.residualLogicalNet(producer);
  if (!net || llvm::find(net->sinks, consumer) == net->sinks.end())
    return invalid("constraint transfer sink is realization-internal");
  return llvm::Error::success();
}

bool isMappedAddressedMemoryActor(const TechMappingView &techMapping,
                                  ::dataflow::ActorRef actor) {
  for (const TechMemoryRealizationView &realization :
       techMapping.memoryRealizations())
    for (const TechMemoryActorView &mapped : realization.actors)
      if (mapped.actor == actor)
        return true;
  return false;
}

llvm::Expected<SpatialConstraintSubject>
decodeSubject(::mapping::SpatialConstraintProjection projection,
              Attribute attribute,
              const ::dataflow::CanonicalDataflowProgramView &dataflow,
              const TechMappingView &techMapping) {
  using Projection = ::mapping::SpatialConstraintProjection;
  switch (projection) {
  case Projection::ComputePlacement:
  case Projection::ComputeParentPe:
  case Projection::ComputeInstructionContext:
  case Projection::ComputeFuContext: {
    const std::uint64_t entity =
        cast<::mapping::ComputeRealizationRefAttr>(attribute).getEntity();
    if (!hasComputeRealization(techMapping, entity))
      return invalid(
          "constraint names a stale or wrong-kind ComputeRealization");
    return SpatialConstraintSubject(TechComputeRealizationRef{entity});
  }
  case Projection::MemoryPlacement: {
    const std::uint64_t entity =
        cast<::mapping::MemoryRealizationRefAttr>(attribute).getEntity();
    if (!hasMemoryRealization(techMapping, entity))
      return invalid(
          "constraint names a stale or wrong-kind MemoryRealization");
    return SpatialConstraintSubject(TechMemoryRealizationRef{entity});
  }
  case Projection::NetAssignedTagValues:
  case Projection::NetSelectedPhysicalTraversals:
  case Projection::NetTraversalResourceStates: {
    auto producer =
        decodeDataflow<::dataflow::CanonicalGraphProducerEndpointRef>(
            cast<::mapping::GraphProducerEndpointRefAttr>(attribute),
            dataflow.identity());
    if (!producer)
      return contextual(producer.takeError(),
                        "constraint producer reference is malformed");
    if (llvm::Error error =
            validateResidualProducer(*producer, dataflow, techMapping))
      return std::move(error);
    return SpatialConstraintSubject(std::move(*producer));
  }
  case Projection::SpatialTransferAttachment: {
    auto terminal = cast<::mapping::SpatialTransferTerminalAttr>(attribute);
    auto producer =
        decodeDataflow<::dataflow::CanonicalGraphProducerEndpointRef>(
            terminal.getProducer(), dataflow.identity());
    if (!producer)
      return contextual(producer.takeError(),
                        "constraint transfer producer is malformed");
    if (llvm::Error error =
            validateResidualProducer(*producer, dataflow, techMapping))
      return std::move(error);
    std::optional<::dataflow::CanonicalGraphConsumerEndpointRef> consumer;
    if (terminal.getConsumer()) {
      auto decoded =
          decodeDataflow<::dataflow::CanonicalGraphConsumerEndpointRef>(
              terminal.getConsumer(), dataflow.identity());
      if (!decoded)
        return contextual(decoded.takeError(),
                          "constraint transfer consumer is malformed");
      if (llvm::Error error =
              validateResidualSink(*producer, *decoded, dataflow, techMapping))
        return std::move(error);
      consumer = std::move(*decoded);
    }
    return SpatialConstraintSubject(SpatialConstraintTransferTerminal{
        std::move(*producer), std::move(consumer)});
  }
  case Projection::MemoryOperationPort: {
    auto actor = decodeDataflow<::dataflow::ActorRef>(
        cast<::mapping::ActorRefAttr>(attribute), dataflow.identity());
    if (!actor)
      return contextual(actor.takeError(),
                        "constraint memory actor reference is malformed");
    auto resolved = dataflow.resolve(*actor);
    if (!resolved)
      return contextual(resolved.takeError(),
                        "constraint memory actor does not resolve");
    if (!isa<::dataflow::LoadOp, ::dataflow::StoreOp>(resolved->op) ||
        !isMappedAddressedMemoryActor(techMapping, *actor))
      return invalid(
          "memory_operation_port subject is not a realized load or store");
    return SpatialConstraintSubject(std::move(*actor));
  }
  case Projection::MemoryBoundServices:
  case Projection::MemoryAddressRegion: {
    auto root = decodeDataflow<::dataflow::LogicalMemoryRootRef>(
        cast<::mapping::LogicalMemoryRootRefAttr>(attribute),
        dataflow.identity());
    if (!root)
      return contextual(root.takeError(),
                        "constraint logical memory root is malformed");
    auto resolved = dataflow.resolve(*root);
    if (!resolved)
      return contextual(resolved.takeError(),
                        "constraint logical memory root does not resolve");
    return SpatialConstraintSubject(std::move(*root));
  }
  }
  llvm_unreachable("unknown Spatial constraint projection");
}

template <typename Ref, typename Attr>
llvm::Expected<Ref>
decodeValidatedFabric(Attr attribute,
                      const ::loom::fabric::FabricArtifactView &fabric,
                      const llvm::Twine &description) {
  auto reference = decodeFabric<Ref>(attribute);
  if (!reference)
    return contextual(reference.takeError(), description + " is malformed");
  if (llvm::Error error = ::loom::fabric::validateFabricRef(fabric, *reference))
    return contextual(std::move(error), description + " does not resolve");
  return std::move(*reference);
}

SpatialConstraintUnsignedInterval
decodeInterval(::mapping::ConstraintUnsignedIntervalAttr interval) {
  return SpatialConstraintUnsignedInterval{interval.getLower().getValue(),
                                           interval.getUpper().getValue()};
}

llvm::Expected<SpatialConstraintDomainValue>
decodeDomainValue(::mapping::SpatialConstraintProjection projection,
                  Attribute attribute,
                  const ::loom::fabric::FabricArtifactView &fabric) {
  using Projection = ::mapping::SpatialConstraintProjection;
  switch (projection) {
  case Projection::ComputePlacement: {
    auto value = decodeValidatedFabric<::loom::fabric::FabricFuOccurrenceRef>(
        cast<::mapping::FabricFuOccurrenceRefAttr>(attribute), fabric,
        "constraint FU occurrence");
    if (!value)
      return value.takeError();
    return SpatialConstraintDomainValue(std::move(*value));
  }
  case Projection::ComputeParentPe: {
    auto value = decodeValidatedFabric<::loom::fabric::FabricPeOccurrenceRef>(
        cast<::mapping::FabricPeOccurrenceRefAttr>(attribute), fabric,
        "constraint PE occurrence");
    if (!value)
      return value.takeError();
    return SpatialConstraintDomainValue(std::move(*value));
  }
  case Projection::ComputeInstructionContext: {
    auto value = decodeValidatedFabric<::loom::fabric::InstructionContextRef>(
        cast<::mapping::InstructionContextRefAttr>(attribute), fabric,
        "constraint instruction context");
    if (!value)
      return value.takeError();
    return SpatialConstraintDomainValue(std::move(*value));
  }
  case Projection::ComputeFuContext: {
    auto tuple = cast<::mapping::ConstraintFuContextAttr>(attribute);
    auto fu = decodeValidatedFabric<::loom::fabric::FabricFuOccurrenceRef>(
        tuple.getFu(), fabric, "constraint FU/context FU");
    if (!fu)
      return fu.takeError();
    auto context = decodeValidatedFabric<::loom::fabric::InstructionContextRef>(
        tuple.getInstructionContext(), fabric,
        "constraint FU/context instruction context");
    if (!context)
      return context.takeError();
    return SpatialConstraintDomainValue(
        SpatialConstraintFuContext{std::move(*fu), std::move(*context)});
  }
  case Projection::MemoryPlacement: {
    auto value =
        decodeValidatedFabric<::loom::fabric::FabricMemoryOccurrenceRef>(
            cast<::mapping::FabricMemoryOccurrenceRefAttr>(attribute), fabric,
            "constraint memory occurrence");
    if (!value)
      return value.takeError();
    return SpatialConstraintDomainValue(std::move(*value));
  }
  case Projection::NetAssignedTagValues:
    return SpatialConstraintDomainValue(decodeInterval(
        cast<::mapping::ConstraintUnsignedIntervalAttr>(attribute)));
  case Projection::NetSelectedPhysicalTraversals: {
    auto value =
        decodeValidatedFabric<::loom::fabric::FabricPhysicalTraversalRef>(
            cast<::mapping::FabricPhysicalTraversalRefAttr>(attribute), fabric,
            "constraint physical traversal");
    if (!value)
      return value.takeError();
    return SpatialConstraintDomainValue(std::move(*value));
  }
  case Projection::NetTraversalResourceStates: {
    auto value = decodeValidatedFabric<::loom::fabric::FabricResourceStateRef>(
        cast<::mapping::FabricResourceStateRefAttr>(attribute), fabric,
        "constraint resource state");
    if (!value)
      return value.takeError();
    return SpatialConstraintDomainValue(std::move(*value));
  }
  case Projection::SpatialTransferAttachment: {
    auto value =
        decodeValidatedFabric<::loom::fabric::FabricTransportEndpointRef>(
            cast<::mapping::FabricTransportEndpointRefAttr>(attribute), fabric,
            "constraint transport endpoint");
    if (!value)
      return value.takeError();
    return SpatialConstraintDomainValue(std::move(*value));
  }
  case Projection::MemoryOperationPort: {
    auto value =
        decodeValidatedFabric<::loom::fabric::FabricMemoryOperationPortRef>(
            cast<::mapping::FabricMemoryOperationPortRefAttr>(attribute),
            fabric, "constraint memory operation port");
    if (!value)
      return value.takeError();
    return SpatialConstraintDomainValue(std::move(*value));
  }
  case Projection::MemoryBoundServices: {
    auto value = decodeValidatedFabric<::loom::fabric::FabricMemoryServiceRef>(
        cast<::mapping::FabricMemoryServiceRefAttr>(attribute), fabric,
        "constraint memory service");
    if (!value)
      return value.takeError();
    return SpatialConstraintDomainValue(std::move(*value));
  }
  case Projection::MemoryAddressRegion: {
    auto region = cast<::mapping::ConstraintAddressRegionAttr>(attribute);
    auto service =
        decodeValidatedFabric<::loom::fabric::FabricMemoryServiceRef>(
            region.getService(), fabric,
            "constraint address-region memory service");
    if (!service)
      return service.takeError();
    std::vector<SpatialConstraintUnsignedInterval> intervals;
    intervals.reserve(region.getIntervals().size());
    for (Attribute interval : region.getIntervals())
      intervals.push_back(decodeInterval(
          cast<::mapping::ConstraintUnsignedIntervalAttr>(interval)));
    return SpatialConstraintDomainValue(SpatialConstraintAddressRegion{
        std::move(*service), std::move(intervals)});
  }
  }
  llvm_unreachable("unknown Spatial constraint projection");
}

llvm::Expected<std::vector<SpatialConstraintSubject>>
decodeSubjects(::mapping::SpatialConstraintProjection projection,
               ArrayAttr attributes,
               const ::dataflow::CanonicalDataflowProgramView &dataflow,
               const TechMappingView &techMapping) {
  std::vector<SpatialConstraintSubject> result;
  result.reserve(attributes.size());
  for (Attribute attribute : attributes) {
    auto subject = decodeSubject(projection, attribute, dataflow, techMapping);
    if (!subject)
      return subject.takeError();
    result.push_back(std::move(*subject));
  }
  return result;
}

llvm::Expected<std::vector<SpatialConstraintDomainValue>>
decodeDomain(::mapping::SpatialConstraintProjection projection,
             ArrayAttr attributes,
             const ::loom::fabric::FabricArtifactView &fabric) {
  std::vector<SpatialConstraintDomainValue> result;
  result.reserve(attributes.size());
  for (Attribute attribute : attributes) {
    auto value = decodeDomainValue(projection, attribute, fabric);
    if (!value)
      return value.takeError();
    result.push_back(std::move(*value));
  }
  return result;
}

struct PreparedSpatialConstraintSet final {
  ArtifactRootReference reference;
  CanonicalSemanticBytes canonicalBytes;
  std::unique_ptr<MLIRContext> context;
  OwningOpRef<ModuleOp> module;
  ::mapping::ConstraintsSpatialOp root;
};

llvm::Expected<PreparedSpatialConstraintSet>
prepareSpatialConstraintSet(::mapping::ConstraintsSpatialOp source) {
  auto canonicalBytes = writeCanonicalSpatialConstraintAssembly(source);
  if (!canonicalBytes)
    return canonicalBytes.takeError();
  auto parsed = parseSpatialConstraintRoot(*canonicalBytes);
  if (!parsed)
    return parsed.takeError();
  ArtifactRootReference reference{
      mappingConstraintSetSchema.identity.str(),
      mappingConstraintSetSchema.version,
      finalizeArtifactIdentity(mappingConstraintSetSchema, *canonicalBytes)};
  return PreparedSpatialConstraintSet{
      std::move(reference), std::move(*canonicalBytes),
      std::move(parsed->context), std::move(parsed->module), parsed->root};
}

llvm::Error requirePublishedUpstreams(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ArtifactStore &store) {
  const ArtifactRootReference dataflowReference{
      ::dataflow::canonicalDataflowSchema.identity.str(),
      ::dataflow::canonicalDataflowSchema.version, dataflow.identity()};
  const ArtifactRootReference techMappingReference{
      mappingArtifactSchema.identity.str(), mappingArtifactSchema.version,
      techMapping.identity()};
  const ArtifactRootReference fabricReference{
      ::loom::fabric::fabricArtifactSchema.identity.str(),
      ::loom::fabric::fabricArtifactSchema.version, fabric.identity()};
  auto dataflowBytes = store.get(dataflowReference);
  if (!dataflowBytes)
    return dataflowBytes.takeError();
  auto techMappingBytes = store.get(techMappingReference);
  if (!techMappingBytes)
    return techMappingBytes.takeError();
  auto fabricBytes = store.get(fabricReference);
  if (!fabricBytes)
    return fabricBytes.takeError();
  return llvm::Error::success();
}

llvm::Expected<SpatialMappingConstraintSetView>
strictImport(const ArtifactIdentity &identity,
             const CanonicalSemanticBytes &canonicalBytes,
             const ArtifactStore &store) {
  if (finalizeArtifactIdentity(mappingConstraintSetSchema, canonicalBytes) !=
      identity)
    return invalid("constraint identity does not match canonical bytes");
  auto parsed = parseSpatialConstraintRoot(canonicalBytes);
  if (!parsed)
    return parsed.takeError();

  auto dataflowIdentity = decodeIdentity(parsed->root.getDataflow());
  if (!dataflowIdentity)
    return dataflowIdentity.takeError();
  auto techMappingIdentity = decodeIdentity(parsed->root.getTechMapping());
  if (!techMappingIdentity)
    return techMappingIdentity.takeError();
  auto fabricIdentity = decodeIdentity(parsed->root.getFabric());
  if (!fabricIdentity)
    return fabricIdentity.takeError();

  ArtifactRootReference dataflowReference{
      ::dataflow::canonicalDataflowSchema.identity.str(),
      ::dataflow::canonicalDataflowSchema.version, *dataflowIdentity};
  auto dataflow = ::dataflow::importCanonicalDataflow(dataflowReference, store);
  if (!dataflow)
    return dataflow.takeError();
  auto dataflowView = dataflow->view();
  if (!dataflowView)
    return dataflowView.takeError();

  ArtifactRootReference techMappingReference{
      mappingArtifactSchema.identity.str(), mappingArtifactSchema.version,
      *techMappingIdentity};
  auto techMapping = importTechMapping(techMappingReference, store);
  if (!techMapping)
    return techMapping.takeError();

  ArtifactRootReference fabricReference{
      ::loom::fabric::fabricArtifactSchema.identity.str(),
      ::loom::fabric::fabricArtifactSchema.version, *fabricIdentity};
  auto fabric = ::loom::fabric::importEntireFabricRoot(fabricReference, store);
  if (!fabric)
    return fabric.takeError();

  auto view = SpatialMappingConstraintSetView::import(
      identity, parsed->root, *dataflowView, techMapping->view(),
      fabric->view());
  if (!view)
    return view.takeError();
  auto rewritten = writeCanonicalSpatialConstraintAssembly(parsed->root);
  if (!rewritten)
    return rewritten.takeError();
  if (!rewritten->bytes().equals(canonicalBytes.bytes()))
    return invalid("stored Spatial constraint payload is not canonical");
  return view;
}

llvm::Error publishPreparedSpatialConstraintSet(
    const PreparedSpatialConstraintSet &prepared, const ArtifactStore &store) {
  auto stored = store.put(mappingConstraintSetSchema, prepared.canonicalBytes);
  if (!stored)
    return stored.takeError();
  if (*stored != prepared.reference.artifact)
    return invalid("ArtifactStore returned a different constraint identity");
  return llvm::Error::success();
}

} // namespace

llvm::Expected<CanonicalSemanticBytes>
writeCanonicalSpatialConstraintAssembly(::mapping::ConstraintsSpatialOp root) {
  OwningOpRef<Operation *> clone(root->clone());
  auto canonical = cast<::mapping::ConstraintsSpatialOp>(clone.get());
  if (failed(verify(canonical)))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Spatial MappingConstraintSet is structurally invalid");

  auto clauses = canonicalizeClauses(canonical);
  if (!clauses)
    return clauses.takeError();
  Block &body = canonical.getBody().front();
  while (!body.empty())
    body.front().erase();
  OpBuilder builder(canonical.getContext());
  builder.setInsertionPointToEnd(&body);
  for (const CanonicalClause &clause : *clauses)
    createClause(builder, canonical.getLoc(), clause);
  if (failed(verify(canonical)))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "canonical Spatial MappingConstraintSet is structurally invalid");

  std::string text;
  llvm::raw_string_ostream stream(text);
  canonical.print(stream, OpPrintingFlags().enableDebugInfo(false));
  stream << '\n';
  stream.flush();
  return CanonicalSemanticBytes(
      std::vector<std::uint8_t>(text.begin(), text.end()));
}

llvm::Expected<SpatialMappingConstraintSetView>
SpatialMappingConstraintSetView::import(
    const ArtifactIdentity &identity, ::mapping::ConstraintsSpatialOp root,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric) {
  auto dataflowIdentity = decodeIdentity(root.getDataflow());
  if (!dataflowIdentity)
    return dataflowIdentity.takeError();
  auto techMappingIdentity = decodeIdentity(root.getTechMapping());
  if (!techMappingIdentity)
    return techMappingIdentity.takeError();
  auto fabricIdentity = decodeIdentity(root.getFabric());
  if (!fabricIdentity)
    return fabricIdentity.takeError();

  if (*dataflowIdentity != dataflow.identity())
    return invalid(
        "Spatial constraint dataflow binding does not match importer");
  if (*techMappingIdentity != techMapping.identity())
    return invalid(
        "Spatial constraint TechMapping binding does not match importer");
  if (*fabricIdentity != fabric.identity())
    return invalid("Spatial constraint Fabric binding does not match importer");
  if (techMapping.dataflowIdentity() != dataflow.identity() ||
      techMapping.fabricIdentity() != fabric.identity())
    return invalid(
        "Spatial constraint inputs do not form one exact D/T/F tuple");

  std::vector<SpatialConstraintClauseView> clauses;
  clauses.reserve(std::distance(root.getBody().front().begin(),
                                root.getBody().front().end()));
  for (Operation &operation : root.getBody().front()) {
    if (auto restriction =
            dyn_cast<::mapping::ConstraintDomainRestrictionOp>(operation)) {
      auto subject =
          decodeSubject(restriction.getProjection(), restriction.getSubject(),
                        dataflow, techMapping);
      if (!subject)
        return subject.takeError();
      auto domain = decodeDomain(restriction.getProjection(),
                                 restriction.getAdmissibleDomain(), fabric);
      if (!domain)
        return domain.takeError();
      clauses.emplace_back(SpatialDomainRestrictionView{
          restriction.getProjection(), std::move(*subject),
          std::move(*domain)});
      continue;
    }
    if (auto equal = dyn_cast<::mapping::ConstraintEqualOp>(operation)) {
      auto subjects = decodeSubjects(equal.getProjection(), equal.getSubjects(),
                                     dataflow, techMapping);
      if (!subjects)
        return subjects.takeError();
      clauses.emplace_back(
          SpatialEqualView{equal.getProjection(), std::move(*subjects)});
      continue;
    }
    auto disjoint = cast<::mapping::ConstraintDisjointOp>(operation);
    auto subjects =
        decodeSubjects(disjoint.getProjection(), disjoint.getSubjects(),
                       dataflow, techMapping);
    if (!subjects)
      return subjects.takeError();
    clauses.emplace_back(
        SpatialDisjointView{disjoint.getProjection(), std::move(*subjects)});
  }

  return SpatialMappingConstraintSetView(
      identity, std::move(*dataflowIdentity), std::move(*techMappingIdentity),
      std::move(*fabricIdentity), std::move(clauses));
}

llvm::Expected<FinalizedSpatialMappingConstraintSet>
finalizeSpatialMappingConstraintSet(::mapping::ConstraintsSpatialOp source,
                                    const ArtifactStore &store) {
  auto prepared = prepareSpatialConstraintSet(source);
  if (!prepared)
    return prepared.takeError();
  auto view = strictImport(prepared->reference.artifact,
                           prepared->canonicalBytes, store);
  if (!view)
    return view.takeError();
  if (llvm::Error error = publishPreparedSpatialConstraintSet(*prepared, store))
    return std::move(error);
  return FinalizedSpatialMappingConstraintSet(
      std::move(prepared->reference), std::move(prepared->canonicalBytes),
      std::move(*view));
}

llvm::Expected<FinalizedSpatialMappingConstraintSet>
finalizeSpatialMappingConstraintSet(
    ::mapping::ConstraintsSpatialOp source,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ArtifactStore &store) {
  if (llvm::Error error =
          requirePublishedUpstreams(dataflow, techMapping, fabric, store))
    return std::move(error);
  auto prepared = prepareSpatialConstraintSet(source);
  if (!prepared)
    return prepared.takeError();
  auto view = SpatialMappingConstraintSetView::import(
      prepared->reference.artifact, prepared->root, dataflow, techMapping,
      fabric);
  if (!view)
    return view.takeError();
  if (llvm::Error error = publishPreparedSpatialConstraintSet(*prepared, store))
    return std::move(error);
  return FinalizedSpatialMappingConstraintSet(
      std::move(prepared->reference), std::move(prepared->canonicalBytes),
      std::move(*view));
}

llvm::Expected<FinalizedSpatialMappingConstraintSet>
importSpatialMappingConstraintSet(const ArtifactRootReference &reference,
                                  const ArtifactStore &store) {
  if (reference.schemaIdentity != mappingConstraintSetSchema.identity ||
      reference.schemaVersion != mappingConstraintSetSchema.version)
    return invalid("root reference has the wrong constraint schema");
  auto canonicalBytes = store.get(reference);
  if (!canonicalBytes)
    return canonicalBytes.takeError();
  auto view = strictImport(reference.artifact, *canonicalBytes, store);
  if (!view)
    return view.takeError();
  return FinalizedSpatialMappingConstraintSet(
      reference, std::move(*canonicalBytes), std::move(*view));
}

} // namespace loom::mapping
