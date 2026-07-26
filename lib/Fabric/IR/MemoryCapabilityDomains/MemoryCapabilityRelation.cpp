#include "Fabric/IR/MemoryCapabilityRelation.h"

#include "Dataflow/IR/OperationSchemaCodec.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <map>
#include <system_error>
#include <tuple>

using dataflow::OperationSchemaId;

namespace fabric::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(std::errc::invalid_argument, "%s",
                                 message.str().c_str());
}

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendFrame(std::vector<std::uint8_t> &bytes,
                 llvm::ArrayRef<std::uint8_t> field) {
  appendU64(bytes, field.size());
  bytes.insert(bytes.end(), field.begin(), field.end());
}

llvm::Expected<ReducedFiniteDomain>
singletonFinite(std::vector<std::uint8_t> bytes) {
  if (bytes.empty())
    return invalid("memory capability physical facts must not be empty");
  return ReducedFiniteDomain{{ReducedFiniteAtom{std::move(bytes)}}};
}

llvm::Expected<ReducedFiniteDomain>
usePatternDomain(llvm::ArrayRef<UsePatternKey> patterns) {
  if (patterns.empty())
    return invalid("memory capability has no admissible use pattern");
  std::vector<ReducedFiniteAtom> atoms;
  atoms.reserve(patterns.size());
  for (UsePatternKey pattern : patterns) {
    std::vector<std::uint8_t> bytes;
    appendU32(bytes, pattern.ordinal());
    atoms.push_back({std::move(bytes)});
  }
  llvm::sort(atoms, [](const auto &left, const auto &right) {
    return left.bytes < right.bytes;
  });
  for (std::size_t index = 1; index < atoms.size(); ++index)
    if (atoms[index - 1].bytes == atoms[index].bytes)
      return invalid("memory capability repeats an admissible use pattern");
  return ReducedFiniteDomain{std::move(atoms)};
}

llvm::Expected<std::uint32_t>
decodeUsePattern(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() != 4)
    return invalid("memory capability use pattern is not one u32be value");
  std::uint32_t value = 0;
  for (std::uint8_t byte : bytes)
    value = (value << 8) | byte;
  return value;
}

llvm::Expected<std::vector<std::uint8_t>>
actorKey(const MemoryCapabilityRelationEntry &entry) {
  return encodeMemoryActorContractDomain(entry.actorContractDomain);
}

llvm::Expected<std::vector<std::uint8_t>>
accessKey(const MemoryCapabilityRelationEntry &entry) {
  if (!entry.accessDomain)
    return std::vector<std::uint8_t>{};
  return encodeParameterizedMemoryAccessDomain(*entry.accessDomain);
}

std::vector<std::uint8_t> patternKey(llvm::ArrayRef<UsePatternKey> patterns) {
  std::vector<std::uint8_t> bytes;
  for (UsePatternKey pattern : patterns)
    appendU32(bytes, pattern.ordinal());
  return bytes;
}

llvm::Expected<std::vector<std::uint8_t>>
entryKey(const MemoryCapabilityRelationEntry &entry) {
  auto actor = actorKey(entry);
  auto access = accessKey(entry);
  if (!actor)
    return actor.takeError();
  if (!access)
    return access.takeError();
  std::vector<std::uint8_t> bytes;
  appendFrame(bytes, *actor);
  appendU32(bytes, entry.accessDomain ? 1 : 0);
  if (entry.accessDomain)
    appendFrame(bytes, *access);
  appendFrame(bytes, entry.physicalFacts);
  appendU64(bytes, entry.admissibleUsePatterns.size());
  for (UsePatternKey pattern : entry.admissibleUsePatterns)
    appendU32(bytes, pattern.ordinal());
  return bytes;
}

struct PartitionKey {
  OperationSchemaId schema;
  std::uint32_t clauseTag;
  bool hasAccess;

  bool operator<(const PartitionKey &other) const {
    return std::tie(schema, clauseTag, hasAccess) <
           std::tie(other.schema, other.clauseTag, other.hasAccess);
  }
};

llvm::Expected<std::vector<MemoryCapabilityRelationEntry>>
reduceEntries(mlir::MLIRContext *context,
              llvm::ArrayRef<MemoryCapabilityRelationEntry> input) {
  std::map<PartitionKey, std::vector<ReducedProductRow>> partitions;
  for (const MemoryCapabilityRelationEntry &entry : input) {
    auto physical = singletonFinite(entry.physicalFacts);
    auto patterns = usePatternDomain(entry.admissibleUsePatterns);
    if (!physical)
      return physical.takeError();
    if (!patterns)
      return patterns.takeError();
    for (const MemoryActorContractClause &clause :
         entry.actorContractDomain.clauses()) {
      auto actor = projectMemoryActorContractClause(clause);
      if (!actor)
        return actor.takeError();
      if (!entry.accessDomain) {
        ReducedProductRow row = actor->fields;
        row.push_back(*physical);
        row.push_back(*patterns);
        partitions[{entry.actorContractDomain.actorSchema(), actor->tag, false}]
            .push_back(std::move(row));
        continue;
      }
      for (const MemoryAccessClass &access :
           entry.accessDomain->accessClasses()) {
        auto accessRow = projectMemoryAccessClass(access);
        if (!accessRow)
          return accessRow.takeError();
        ReducedProductRow row = actor->fields;
        row.insert(row.end(), accessRow->begin(), accessRow->end());
        row.push_back(*physical);
        row.push_back(*patterns);
        partitions[{entry.actorContractDomain.actorSchema(), actor->tag, true}]
            .push_back(std::move(row));
      }
    }
  }

  std::vector<MemoryCapabilityRelationEntry> pieces;
  for (auto &[partition, rows] : partitions) {
    const std::size_t actorFieldCount =
        rows.front().size() - (partition.hasAccess ? 9 : 2);
    llvm::SmallVector<bool, 16> grouping(actorFieldCount, true);
    if (partition.hasAccess) {
      grouping.push_back(false);
      grouping.append(6, true);
    }
    grouping.push_back(false);
    grouping.push_back(true);
    auto reduced = reduceProductRelation(rows, grouping);
    if (!reduced)
      return reduced.takeError();

    for (const ReducedProductRow &row : *reduced) {
      MemoryActorClauseRelation actorRelation{
          partition.clauseTag,
          ReducedProductRow(row.begin(), row.begin() + actorFieldCount)};
      auto clause = importMemoryActorContractClause(actorRelation, context);
      if (!clause)
        return clause.takeError();
      auto actors =
          MemoryActorContractDomain::fromCanonical(partition.schema, {*clause});
      if (!actors)
        return actors.takeError();

      std::size_t cursor = actorFieldCount;
      std::optional<ParameterizedMemoryAccessDomain> accesses;
      if (partition.hasAccess) {
        ReducedProductRow accessRow(row.begin() + cursor,
                                    row.begin() + cursor + 7);
        auto access = importMemoryAccessClass(accessRow);
        if (!access)
          return access.takeError();
        auto domain = ParameterizedMemoryAccessDomain::fromCanonical({*access});
        if (!domain)
          return domain.takeError();
        accesses = std::move(*domain);
        cursor += 7;
      }

      const auto *physical = std::get_if<ReducedFiniteDomain>(&row[cursor++]);
      const auto *patterns = std::get_if<ReducedFiniteDomain>(&row[cursor++]);
      if (!physical || physical->atoms.size() != 1 || !patterns ||
          cursor != row.size())
        return invalid("normalized memory capability relation is malformed");
      std::vector<UsePatternKey> usePatterns;
      usePatterns.reserve(patterns->atoms.size());
      for (const ReducedFiniteAtom &atom : patterns->atoms) {
        auto ordinal = decodeUsePattern(atom.bytes);
        if (!ordinal)
          return ordinal.takeError();
        usePatterns.emplace_back(*ordinal);
      }
      pieces.push_back({std::move(*actors), std::move(accesses),
                        physical->atoms.front().bytes, std::move(usePatterns)});
    }
  }
  return pieces;
}

llvm::Expected<std::vector<MemoryCapabilityRelationEntry>>
mergeDomains(std::vector<MemoryCapabilityRelationEntry> pieces) {
  std::vector<std::vector<std::uint8_t>> previousEncoding;
  while (true) {
    using ActorMergeKey =
        std::tuple<OperationSchemaId, std::vector<std::uint8_t>,
                   std::vector<std::uint8_t>, std::vector<std::uint8_t>>;
    std::map<ActorMergeKey, std::vector<MemoryCapabilityRelationEntry>>
        bySuffix;
    for (MemoryCapabilityRelationEntry &piece : pieces) {
      auto access = accessKey(piece);
      if (!access)
        return access.takeError();
      bySuffix[{piece.actorContractDomain.actorSchema(), std::move(*access),
                piece.physicalFacts, patternKey(piece.admissibleUsePatterns)}]
          .push_back(std::move(piece));
    }
    pieces.clear();
    for (auto &[key, group] : bySuffix) {
      std::vector<MemoryActorContractClause> clauses;
      for (const auto &piece : group)
        clauses.insert(clauses.end(),
                       piece.actorContractDomain.clauses().begin(),
                       piece.actorContractDomain.clauses().end());
      auto actors =
          MemoryActorContractDomain::create(std::get<0>(key), clauses);
      if (!actors)
        return actors.takeError();
      MemoryCapabilityRelationEntry merged = std::move(group.front());
      merged.actorContractDomain = std::move(*actors);
      pieces.push_back(std::move(merged));
    }

    using AccessMergeKey =
        std::tuple<std::vector<std::uint8_t>, std::vector<std::uint8_t>,
                   std::vector<std::uint8_t>>;
    std::map<AccessMergeKey, std::vector<MemoryCapabilityRelationEntry>>
        byPrefix;
    for (MemoryCapabilityRelationEntry &piece : pieces) {
      auto actor = actorKey(piece);
      if (!actor)
        return actor.takeError();
      byPrefix[{std::move(*actor), piece.physicalFacts,
                patternKey(piece.admissibleUsePatterns)}]
          .push_back(std::move(piece));
    }
    pieces.clear();
    for (auto &[key, group] : byPrefix) {
      const bool hasAccess = group.front().accessDomain.has_value();
      for (const auto &piece : group)
        if (piece.accessDomain.has_value() != hasAccess)
          return invalid("fence and addressed memory capabilities overlap");
      std::optional<ParameterizedMemoryAccessDomain> mergedAccess;
      if (hasAccess) {
        std::vector<MemoryAccessClass> classes;
        for (const auto &piece : group)
          classes.insert(classes.end(),
                         piece.accessDomain->accessClasses().begin(),
                         piece.accessDomain->accessClasses().end());
        auto accesses = ParameterizedMemoryAccessDomain::create(classes);
        if (!accesses)
          return accesses.takeError();
        mergedAccess = std::move(*accesses);
      }
      MemoryCapabilityRelationEntry merged = std::move(group.front());
      merged.accessDomain = std::move(mergedAccess);
      pieces.push_back(std::move(merged));
    }

    std::vector<std::vector<std::uint8_t>> encoding;
    encoding.reserve(pieces.size());
    for (const auto &piece : pieces) {
      auto bytes = entryKey(piece);
      if (!bytes)
        return bytes.takeError();
      encoding.push_back(std::move(*bytes));
    }
    llvm::sort(encoding);
    if (encoding == previousEncoding)
      break;
    previousEncoding = std::move(encoding);
  }
  return pieces;
}

} // namespace

llvm::Expected<std::vector<MemoryCapabilityRelationEntry>>
normalizeMemoryCapabilityRelation(
    mlir::MLIRContext *context,
    llvm::ArrayRef<MemoryCapabilityRelationEntry> entries) {
  if (!context)
    return invalid("memory capability relation requires an MLIR context");
  if (entries.empty())
    return invalid("memory capability relation must not be empty");

  std::vector<MemoryCapabilityRelationEntry> prepared(entries.begin(),
                                                      entries.end());
  for (MemoryCapabilityRelationEntry &entry : prepared) {
    if (entry.physicalFacts.empty())
      return invalid("memory capability physical facts must not be empty");
    llvm::sort(entry.admissibleUsePatterns,
               [](UsePatternKey left, UsePatternKey right) {
                 return left.ordinal() < right.ordinal();
               });
    if (std::adjacent_find(entry.admissibleUsePatterns.begin(),
                           entry.admissibleUsePatterns.end()) !=
        entry.admissibleUsePatterns.end())
      return invalid("memory capability repeats an admissible use pattern");
  }

  auto reduced = reduceEntries(context, prepared);
  if (!reduced)
    return reduced.takeError();
  auto merged = mergeDomains(std::move(*reduced));
  if (!merged)
    return merged.takeError();

  struct EncodedEntry {
    std::vector<std::uint8_t> bytes;
    MemoryCapabilityRelationEntry entry;
  };
  std::vector<EncodedEntry> ordered;
  ordered.reserve(merged->size());
  for (MemoryCapabilityRelationEntry &entry : *merged) {
    auto bytes = entryKey(entry);
    if (!bytes)
      return bytes.takeError();
    ordered.push_back({std::move(*bytes), std::move(entry)});
  }
  llvm::sort(ordered, [](const auto &left, const auto &right) {
    return left.bytes < right.bytes;
  });
  std::vector<MemoryCapabilityRelationEntry> result;
  result.reserve(ordered.size());
  for (EncodedEntry &entry : ordered)
    result.push_back(std::move(entry.entry));
  return result;
}

llvm::Expected<bool>
memoryAccessDomainCovers(const ParameterizedMemoryAccessDomain &superset,
                         const ParameterizedMemoryAccessDomain &subset) {
  std::vector<ReducedProductRow> supersetRows;
  std::vector<ReducedProductRow> subsetRows;
  for (const MemoryAccessClass &access : superset.accessClasses()) {
    auto row = projectMemoryAccessClass(access);
    if (!row)
      return row.takeError();
    supersetRows.push_back(std::move(*row));
  }
  for (const MemoryAccessClass &access : subset.accessClasses()) {
    auto row = projectMemoryAccessClass(access);
    if (!row)
      return row.takeError();
    subsetRows.push_back(std::move(*row));
  }
  return reducedProductRelationCovers(supersetRows, subsetRows);
}

llvm::Expected<bool> memoryCapabilityDomainsOverlap(
    const MemoryActorContractDomain &leftActors,
    const std::optional<ParameterizedMemoryAccessDomain> &leftAccesses,
    const MemoryActorContractDomain &rightActors,
    const std::optional<ParameterizedMemoryAccessDomain> &rightAccesses) {
  if (leftActors.actorSchema() != rightActors.actorSchema() ||
      leftAccesses.has_value() != rightAccesses.has_value())
    return false;

  for (const MemoryActorContractClause &leftClause : leftActors.clauses()) {
    auto leftActor = projectMemoryActorContractClause(leftClause);
    if (!leftActor)
      return leftActor.takeError();
    for (const MemoryActorContractClause &rightClause : rightActors.clauses()) {
      auto rightActor = projectMemoryActorContractClause(rightClause);
      if (!rightActor)
        return rightActor.takeError();
      if (leftActor->tag != rightActor->tag)
        continue;
      if (!leftAccesses) {
        auto overlap = reducedProductRelationsOverlap({leftActor->fields},
                                                      {rightActor->fields});
        if (!overlap || *overlap)
          return overlap;
        continue;
      }
      for (const MemoryAccessClass &leftAccess :
           leftAccesses->accessClasses()) {
        auto leftRow = projectMemoryAccessClass(leftAccess);
        if (!leftRow)
          return leftRow.takeError();
        for (const MemoryAccessClass &rightAccess :
             rightAccesses->accessClasses()) {
          auto rightRow = projectMemoryAccessClass(rightAccess);
          if (!rightRow)
            return rightRow.takeError();
          ReducedProductRow leftCombined = leftActor->fields;
          leftCombined.insert(leftCombined.end(), leftRow->begin(),
                              leftRow->end());
          ReducedProductRow rightCombined = rightActor->fields;
          rightCombined.insert(rightCombined.end(), rightRow->begin(),
                               rightRow->end());
          auto overlap =
              reducedProductRelationsOverlap({leftCombined}, {rightCombined});
          if (!overlap || *overlap)
            return overlap;
        }
      }
    }
  }
  return false;
}

} // namespace fabric::detail
