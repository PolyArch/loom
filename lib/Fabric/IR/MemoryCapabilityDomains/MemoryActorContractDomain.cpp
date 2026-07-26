#include "Fabric/IR/MemoryActorContractDomain.h"

#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Fabric/IR/ReducedProductRelation.h"

#include "llvm/ADT/STLExtras.h"

#include <limits>
#include <map>
#include <system_error>
#include <type_traits>

namespace fabric {
namespace {

using FiniteAtom = detail::ReducedFiniteAtom;
using FiniteDomain = detail::ReducedFiniteDomain;
using RelationDomain = detail::ReducedProductDomain;
using RelationRow = detail::ReducedProductRow;

enum class ClauseTag : std::uint32_t {
  LoadStorePlain = 0,
  LoadStoreAtomic = 1,
  AtomicRmw = 2,
  CompareExchange = 3,
  Fence = 4,
};

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

void appendFramed(std::vector<std::uint8_t> &bytes,
                  llvm::ArrayRef<std::uint8_t> field) {
  appendU64(bytes, field.size());
  bytes.insert(bytes.end(), field.begin(), field.end());
}

class Reader {
public:
  explicit Reader(llvm::ArrayRef<std::uint8_t> bytes) : bytes_(bytes) {}

  llvm::Expected<std::uint32_t> readU32(const llvm::Twine &field) {
    if (remaining() < 4)
      return invalid(field + " is truncated");
    std::uint32_t value = 0;
    for (unsigned index = 0; index < 4; ++index)
      value = (value << 8) | bytes_[offset_++];
    return value;
  }

  llvm::Expected<std::uint64_t> readU64(const llvm::Twine &field) {
    if (remaining() < 8)
      return invalid(field + " is truncated");
    std::uint64_t value = 0;
    for (unsigned index = 0; index < 8; ++index)
      value = (value << 8) | bytes_[offset_++];
    return value;
  }

  llvm::Expected<llvm::ArrayRef<std::uint8_t>>
  readFrame(const llvm::Twine &field) {
    auto size = readU64(field + " length");
    if (!size)
      return size.takeError();
    if (*size > remaining())
      return invalid(field + " is truncated");
    llvm::ArrayRef<std::uint8_t> result = bytes_.slice(offset_, *size);
    offset_ += *size;
    return result;
  }

  llvm::Error finish(const llvm::Twine &record) const {
    if (offset_ != bytes_.size())
      return invalid(record + " has trailing bytes");
    return llvm::Error::success();
  }

  std::size_t remaining() const { return bytes_.size() - offset_; }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
  std::size_t offset_ = 0;
};

llvm::Expected<std::vector<std::uint8_t>>
atomBytes(dataflow::AtomicOrdering value) {
  auto encoded = dataflow::encodeAtomicOrdering(value);
  if (!encoded)
    return encoded.takeError();
  return std::vector<std::uint8_t>(encoded->bytes().begin(),
                                   encoded->bytes().end());
}

llvm::Expected<std::vector<std::uint8_t>>
atomBytes(dataflow::AtomicRmwKind value) {
  auto encoded = dataflow::encodeAtomicRmwKind(value);
  if (!encoded)
    return encoded.takeError();
  return std::vector<std::uint8_t>(encoded->bytes().begin(),
                                   encoded->bytes().end());
}

llvm::Expected<std::vector<std::uint8_t>>
atomBytes(std::optional<dataflow::VectorAtomicGranularity> value) {
  auto encoded = dataflow::encodeOptionalVectorAtomicGranularity(value);
  if (!encoded)
    return encoded.takeError();
  return std::vector<std::uint8_t>(encoded->bytes().begin(),
                                   encoded->bytes().end());
}

llvm::Expected<std::vector<std::uint8_t>>
atomBytes(const dataflow::SyncScopeProjection &value) {
  auto encoded = dataflow::encodeSyncScopeRef(value);
  if (!encoded)
    return encoded.takeError();
  return std::vector<std::uint8_t>(encoded->bytes().begin(),
                                   encoded->bytes().end());
}

llvm::Expected<std::vector<std::uint8_t>> atomBytes(bool value) {
  auto encoded = dataflow::encodeCanonicalBoolean(value);
  if (!encoded)
    return encoded.takeError();
  return std::vector<std::uint8_t>(encoded->bytes().begin(),
                                   encoded->bytes().end());
}

llvm::Expected<std::vector<std::uint8_t>>
atomBytes(CompareExchangeOrderingPair value) {
  auto success = atomBytes(value.success);
  if (!success)
    return success.takeError();
  auto failure = atomBytes(value.failure);
  if (!failure)
    return failure.takeError();
  std::vector<std::uint8_t> bytes;
  appendFramed(bytes, *success);
  appendFramed(bytes, *failure);
  return bytes;
}

template <typename Value>
llvm::Expected<FiniteDomain> finiteDomain(llvm::ArrayRef<Value> values) {
  if (values.empty())
    return invalid("memory actor contract field domain must not be empty");
  std::vector<FiniteAtom> atoms;
  atoms.reserve(values.size());
  for (const Value &value : values) {
    auto bytes = atomBytes(value);
    if (!bytes)
      return bytes.takeError();
    atoms.push_back(FiniteAtom{std::move(*bytes)});
  }
  llvm::sort(atoms, [](const FiniteAtom &lhs, const FiniteAtom &rhs) {
    return lhs.bytes < rhs.bytes;
  });
  for (std::size_t index = 1; index < atoms.size(); ++index)
    if (atoms[index - 1].bytes == atoms[index].bytes)
      return invalid("memory actor contract field contains a duplicate");
  return FiniteDomain{std::move(atoms)};
}

llvm::Expected<FiniteDomain> finiteDomain(const std::vector<bool> &values) {
  if (values.empty())
    return invalid("memory actor contract field domain must not be empty");
  std::vector<FiniteAtom> atoms;
  atoms.reserve(values.size());
  for (bool value : values) {
    auto bytes = atomBytes(value);
    if (!bytes)
      return bytes.takeError();
    atoms.push_back(FiniteAtom{std::move(*bytes)});
  }
  llvm::sort(atoms, [](const FiniteAtom &lhs, const FiniteAtom &rhs) {
    return lhs.bytes < rhs.bytes;
  });
  for (std::size_t index = 1; index < atoms.size(); ++index)
    if (atoms[index - 1].bytes == atoms[index].bytes)
      return invalid("memory actor contract field contains a duplicate");
  return FiniteDomain{std::move(atoms)};
}

llvm::Expected<RelationRow>
clauseRow(const LoadStorePlainContractClause &clause) {
  auto volatility = finiteDomain(clause.volatileValues);
  if (!volatility)
    return volatility.takeError();
  return RelationRow{std::move(*volatility)};
}

llvm::Expected<RelationRow>
clauseRow(const LoadStoreAtomicContractClause &clause) {
  auto orderings = finiteDomain<dataflow::AtomicOrdering>(clause.orderings);
  auto scopes = finiteDomain<dataflow::SyncScopeProjection>(clause.syncScopes);
  auto granularity =
      finiteDomain<std::optional<dataflow::VectorAtomicGranularity>>(
          clause.vectorGranularityValues);
  auto volatility = finiteDomain(clause.volatileValues);
  if (!orderings)
    return orderings.takeError();
  if (!scopes)
    return scopes.takeError();
  if (!granularity)
    return granularity.takeError();
  if (!volatility)
    return volatility.takeError();
  return RelationRow{std::move(*orderings), std::move(*scopes),
                     std::move(*granularity), std::move(*volatility)};
}

llvm::Expected<RelationRow> clauseRow(const AtomicRmwContractClause &clause) {
  auto kinds = finiteDomain<dataflow::AtomicRmwKind>(clause.rmwKinds);
  auto orderings = finiteDomain<dataflow::AtomicOrdering>(clause.orderings);
  auto scopes = finiteDomain<dataflow::SyncScopeProjection>(clause.syncScopes);
  auto granularity =
      finiteDomain<std::optional<dataflow::VectorAtomicGranularity>>(
          clause.vectorGranularityValues);
  auto volatility = finiteDomain(clause.volatileValues);
  if (!kinds)
    return kinds.takeError();
  if (!orderings)
    return orderings.takeError();
  if (!scopes)
    return scopes.takeError();
  if (!granularity)
    return granularity.takeError();
  if (!volatility)
    return volatility.takeError();
  return RelationRow{std::move(*kinds), std::move(*orderings),
                     std::move(*scopes), std::move(*granularity),
                     std::move(*volatility)};
}

llvm::Expected<RelationRow>
clauseRow(const CompareExchangeContractClause &clause) {
  auto pairs = finiteDomain<CompareExchangeOrderingPair>(clause.orderingPairs);
  auto scopes = finiteDomain<dataflow::SyncScopeProjection>(clause.syncScopes);
  auto granularity =
      finiteDomain<std::optional<dataflow::VectorAtomicGranularity>>(
          clause.vectorGranularityValues);
  auto weakness = finiteDomain(clause.weakValues);
  auto volatility = finiteDomain(clause.volatileValues);
  if (!pairs)
    return pairs.takeError();
  if (!scopes)
    return scopes.takeError();
  if (!granularity)
    return granularity.takeError();
  if (!weakness)
    return weakness.takeError();
  if (!volatility)
    return volatility.takeError();
  return RelationRow{std::move(*pairs), std::move(*scopes),
                     std::move(*granularity), std::move(*weakness),
                     std::move(*volatility)};
}

llvm::Expected<RelationRow> clauseRow(const FenceContractClause &clause) {
  auto orderings = finiteDomain<dataflow::AtomicOrdering>(clause.orderings);
  auto scopes = finiteDomain<dataflow::SyncScopeProjection>(clause.syncScopes);
  if (!orderings)
    return orderings.takeError();
  if (!scopes)
    return scopes.takeError();
  return RelationRow{std::move(*orderings), std::move(*scopes)};
}

llvm::Expected<dataflow::AtomicOrdering>
decodeAtom(llvm::ArrayRef<std::uint8_t> bytes, dataflow::AtomicOrdering *) {
  return dataflow::decodeAtomicOrdering(bytes);
}

llvm::Expected<dataflow::AtomicRmwKind>
decodeAtom(llvm::ArrayRef<std::uint8_t> bytes, dataflow::AtomicRmwKind *) {
  return dataflow::decodeAtomicRmwKind(bytes);
}

llvm::Expected<std::optional<dataflow::VectorAtomicGranularity>>
decodeAtom(llvm::ArrayRef<std::uint8_t> bytes,
           std::optional<dataflow::VectorAtomicGranularity> *) {
  return dataflow::decodeOptionalVectorAtomicGranularity(bytes);
}

llvm::Expected<bool> decodeAtom(llvm::ArrayRef<std::uint8_t> bytes, bool *) {
  return dataflow::decodeCanonicalBoolean(bytes);
}

llvm::Expected<CompareExchangeOrderingPair>
decodeAtom(llvm::ArrayRef<std::uint8_t> bytes, CompareExchangeOrderingPair *) {
  Reader reader(bytes);
  auto successBytes = reader.readFrame("success ordering");
  if (!successBytes)
    return successBytes.takeError();
  auto failureBytes = reader.readFrame("failure ordering");
  if (!failureBytes)
    return failureBytes.takeError();
  if (llvm::Error error = reader.finish("compare-exchange ordering pair"))
    return std::move(error);
  auto success = dataflow::decodeAtomicOrdering(*successBytes);
  if (!success)
    return success.takeError();
  auto failure = dataflow::decodeAtomicOrdering(*failureBytes);
  if (!failure)
    return failure.takeError();
  return CompareExchangeOrderingPair{*success, *failure};
}

llvm::Expected<dataflow::SyncScopeProjection>
decodeScopeAtom(llvm::ArrayRef<std::uint8_t> bytes,
                mlir::MLIRContext *context) {
  if (!context)
    return invalid("memory actor contract import requires an MLIR context");
  return dataflow::decodeSyncScopeRef(bytes, context);
}

template <typename Value>
llvm::Expected<std::vector<Value>> valuesOf(const RelationDomain &domain) {
  const auto *finite = std::get_if<FiniteDomain>(&domain);
  if (!finite)
    return invalid("memory actor contract relation field is not finite");
  std::vector<Value> values;
  values.reserve(finite->atoms.size());
  for (const FiniteAtom &atom : finite->atoms) {
    auto value = decodeAtom(atom.bytes, static_cast<Value *>(nullptr));
    if (!value)
      return value.takeError();
    values.push_back(*value);
  }
  return values;
}

using ScopeCatalog =
    std::map<std::vector<std::uint8_t>, dataflow::SyncScopeProjection>;

llvm::Expected<std::vector<dataflow::SyncScopeProjection>>
scopeValuesOf(const RelationDomain &domain, const ScopeCatalog &catalog) {
  const auto *finite = std::get_if<FiniteDomain>(&domain);
  if (!finite)
    return invalid("memory actor contract scope field is not finite");
  std::vector<dataflow::SyncScopeProjection> values;
  values.reserve(finite->atoms.size());
  for (const FiniteAtom &atom : finite->atoms) {
    auto found = catalog.find(atom.bytes);
    if (found == catalog.end())
      return invalid("reduced synchronization scope is not owner-backed");
    values.push_back(found->second);
  }
  return values;
}

llvm::Expected<LoadStorePlainContractClause>
clauseFromRow(const RelationRow &row, LoadStorePlainContractClause *,
              const ScopeCatalog &) {
  if (row.size() != 1)
    return invalid("plain memory contract clause has the wrong field count");
  auto volatility = valuesOf<bool>(row[0]);
  if (!volatility)
    return volatility.takeError();
  return LoadStorePlainContractClause{std::move(*volatility)};
}

llvm::Expected<LoadStoreAtomicContractClause>
clauseFromRow(const RelationRow &row, LoadStoreAtomicContractClause *,
              const ScopeCatalog &catalog) {
  if (row.size() != 4)
    return invalid("atomic access clause has the wrong field count");
  auto orderings = valuesOf<dataflow::AtomicOrdering>(row[0]);
  auto scopes = scopeValuesOf(row[1], catalog);
  auto granularity =
      valuesOf<std::optional<dataflow::VectorAtomicGranularity>>(row[2]);
  auto volatility = valuesOf<bool>(row[3]);
  if (!orderings)
    return orderings.takeError();
  if (!scopes)
    return scopes.takeError();
  if (!granularity)
    return granularity.takeError();
  if (!volatility)
    return volatility.takeError();
  return LoadStoreAtomicContractClause{
      std::move(*orderings), std::move(*scopes), std::move(*granularity),
      std::move(*volatility)};
}

llvm::Expected<AtomicRmwContractClause>
clauseFromRow(const RelationRow &row, AtomicRmwContractClause *,
              const ScopeCatalog &catalog) {
  if (row.size() != 5)
    return invalid("atomic RMW clause has the wrong field count");
  auto kinds = valuesOf<dataflow::AtomicRmwKind>(row[0]);
  auto orderings = valuesOf<dataflow::AtomicOrdering>(row[1]);
  auto scopes = scopeValuesOf(row[2], catalog);
  auto granularity =
      valuesOf<std::optional<dataflow::VectorAtomicGranularity>>(row[3]);
  auto volatility = valuesOf<bool>(row[4]);
  if (!kinds)
    return kinds.takeError();
  if (!orderings)
    return orderings.takeError();
  if (!scopes)
    return scopes.takeError();
  if (!granularity)
    return granularity.takeError();
  if (!volatility)
    return volatility.takeError();
  return AtomicRmwContractClause{std::move(*kinds), std::move(*orderings),
                                 std::move(*scopes), std::move(*granularity),
                                 std::move(*volatility)};
}

llvm::Expected<CompareExchangeContractClause>
clauseFromRow(const RelationRow &row, CompareExchangeContractClause *,
              const ScopeCatalog &catalog) {
  if (row.size() != 5)
    return invalid("compare-exchange clause has the wrong field count");
  auto pairs = valuesOf<CompareExchangeOrderingPair>(row[0]);
  auto scopes = scopeValuesOf(row[1], catalog);
  auto granularity =
      valuesOf<std::optional<dataflow::VectorAtomicGranularity>>(row[2]);
  auto weakness = valuesOf<bool>(row[3]);
  auto volatility = valuesOf<bool>(row[4]);
  if (!pairs)
    return pairs.takeError();
  if (!scopes)
    return scopes.takeError();
  if (!granularity)
    return granularity.takeError();
  if (!weakness)
    return weakness.takeError();
  if (!volatility)
    return volatility.takeError();
  return CompareExchangeContractClause{
      std::move(*pairs), std::move(*scopes), std::move(*granularity),
      std::move(*weakness), std::move(*volatility)};
}

llvm::Expected<FenceContractClause> clauseFromRow(const RelationRow &row,
                                                  FenceContractClause *,
                                                  const ScopeCatalog &catalog) {
  if (row.size() != 2)
    return invalid("fence clause has the wrong field count");
  auto orderings = valuesOf<dataflow::AtomicOrdering>(row[0]);
  auto scopes = scopeValuesOf(row[1], catalog);
  if (!orderings)
    return orderings.takeError();
  if (!scopes)
    return scopes.takeError();
  return FenceContractClause{std::move(*orderings), std::move(*scopes)};
}

template <typename Clause>
llvm::Expected<std::vector<Clause>>
reduceClauses(llvm::ArrayRef<Clause> clauses, const ScopeCatalog &catalog) {
  if (clauses.empty())
    return std::vector<Clause>{};
  std::vector<RelationRow> rows;
  rows.reserve(clauses.size());
  for (const Clause &clause : clauses) {
    auto row = clauseRow(clause);
    if (!row)
      return row.takeError();
    rows.push_back(std::move(*row));
  }
  llvm::SmallVector<bool, 8> grouping(rows.front().size(), true);
  auto reduced = detail::reduceProductRelation(rows, grouping);
  if (!reduced)
    return reduced.takeError();
  std::vector<Clause> result;
  result.reserve(reduced->size());
  for (const RelationRow &row : *reduced) {
    auto clause = clauseFromRow(row, static_cast<Clause *>(nullptr), catalog);
    if (!clause)
      return clause.takeError();
    result.push_back(std::move(*clause));
  }
  return result;
}

ClauseTag clauseTag(const MemoryActorContractClause &clause) {
  return std::visit(
      [](const auto &typed) -> ClauseTag {
        using Clause = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<Clause, LoadStorePlainContractClause>)
          return ClauseTag::LoadStorePlain;
        if constexpr (std::is_same_v<Clause, LoadStoreAtomicContractClause>)
          return ClauseTag::LoadStoreAtomic;
        if constexpr (std::is_same_v<Clause, AtomicRmwContractClause>)
          return ClauseTag::AtomicRmw;
        if constexpr (std::is_same_v<Clause, CompareExchangeContractClause>)
          return ClauseTag::CompareExchange;
        return ClauseTag::Fence;
      },
      clause);
}

bool clauseAllowed(dataflow::OperationSchemaId schema, ClauseTag tag) {
  switch (schema) {
  case dataflow::OperationSchemaId::DataflowLoad:
  case dataflow::OperationSchemaId::DataflowStore:
    return tag == ClauseTag::LoadStorePlain ||
           tag == ClauseTag::LoadStoreAtomic;
  case dataflow::OperationSchemaId::DataflowAtomicRmw:
    return tag == ClauseTag::AtomicRmw;
  case dataflow::OperationSchemaId::DataflowCmpXchg:
    return tag == ClauseTag::CompareExchange;
  case dataflow::OperationSchemaId::DataflowFence:
    return tag == ClauseTag::Fence;
  default:
    return false;
  }
}

bool isReleaseOrAcqRel(dataflow::AtomicOrdering ordering) {
  return ordering == dataflow::AtomicOrdering::Release ||
         ordering == dataflow::AtomicOrdering::AcqRel;
}

bool isAcquireOrAcqRel(dataflow::AtomicOrdering ordering) {
  return ordering == dataflow::AtomicOrdering::Acquire ||
         ordering == dataflow::AtomicOrdering::AcqRel;
}

llvm::Error validateClauseSemantics(dataflow::OperationSchemaId schema,
                                    const MemoryActorContractClause &clause) {
  const ClauseTag tag = clauseTag(clause);
  if (!clauseAllowed(schema, tag))
    return invalid("memory actor contract clause does not match its schema");
  if (const auto *atomic =
          std::get_if<LoadStoreAtomicContractClause>(&clause)) {
    for (dataflow::AtomicOrdering ordering : atomic->orderings) {
      if (schema == dataflow::OperationSchemaId::DataflowLoad &&
          isReleaseOrAcqRel(ordering))
        return invalid("atomic load capability admits an illegal ordering");
      if (schema == dataflow::OperationSchemaId::DataflowStore &&
          isAcquireOrAcqRel(ordering))
        return invalid("atomic store capability admits an illegal ordering");
    }
  }
  if (const auto *rmw = std::get_if<AtomicRmwContractClause>(&clause))
    if (llvm::is_contained(rmw->orderings, dataflow::AtomicOrdering::Unordered))
      return invalid("atomic RMW capability admits unordered execution");
  if (const auto *exchange =
          std::get_if<CompareExchangeContractClause>(&clause))
    for (CompareExchangeOrderingPair pair : exchange->orderingPairs)
      if (pair.success == dataflow::AtomicOrdering::Unordered ||
          pair.failure == dataflow::AtomicOrdering::Unordered ||
          isReleaseOrAcqRel(pair.failure))
        return invalid(
            "compare-exchange capability admits an illegal ordering pair");
  if (const auto *fence = std::get_if<FenceContractClause>(&clause))
    for (dataflow::AtomicOrdering ordering : fence->orderings)
      if (ordering == dataflow::AtomicOrdering::Unordered ||
          ordering == dataflow::AtomicOrdering::Monotonic)
        return invalid("fence capability admits an illegal ordering");
  return llvm::Error::success();
}

template <typename Value>
llvm::Error appendDomain(std::vector<std::uint8_t> &bytes,
                         const std::vector<Value> &values) {
  if (values.empty())
    return invalid("memory actor contract field domain must not be empty");
  std::vector<std::uint8_t> field;
  appendU64(field, values.size());
  for (const Value &value : values) {
    auto encoded = atomBytes(value);
    if (!encoded)
      return encoded.takeError();
    appendFramed(field, *encoded);
  }
  appendFramed(bytes, field);
  return llvm::Error::success();
}

llvm::Error appendDomain(std::vector<std::uint8_t> &bytes,
                         const std::vector<bool> &values) {
  if (values.empty())
    return invalid("memory actor contract field domain must not be empty");
  std::vector<std::uint8_t> field;
  appendU64(field, values.size());
  for (bool value : values) {
    auto encoded = atomBytes(value);
    if (!encoded)
      return encoded.takeError();
    appendFramed(field, *encoded);
  }
  appendFramed(bytes, field);
  return llvm::Error::success();
}

llvm::Expected<std::vector<std::uint8_t>>
encodeClause(const MemoryActorContractClause &clause) {
  std::vector<std::uint8_t> bytes;
  appendU32(bytes, static_cast<std::uint32_t>(clauseTag(clause)));
  llvm::Error error = std::visit(
      [&](const auto &typed) -> llvm::Error {
        using Clause = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<Clause, LoadStorePlainContractClause>) {
          appendU64(bytes, 1);
          return appendDomain(bytes, typed.volatileValues);
        } else if constexpr (std::is_same_v<Clause,
                                            LoadStoreAtomicContractClause>) {
          appendU64(bytes, 4);
          if (llvm::Error field = appendDomain(bytes, typed.orderings))
            return field;
          if (llvm::Error field = appendDomain(bytes, typed.syncScopes))
            return field;
          if (llvm::Error field =
                  appendDomain(bytes, typed.vectorGranularityValues))
            return field;
          return appendDomain(bytes, typed.volatileValues);
        } else if constexpr (std::is_same_v<Clause, AtomicRmwContractClause>) {
          appendU64(bytes, 5);
          if (llvm::Error field = appendDomain(bytes, typed.rmwKinds))
            return field;
          if (llvm::Error field = appendDomain(bytes, typed.orderings))
            return field;
          if (llvm::Error field = appendDomain(bytes, typed.syncScopes))
            return field;
          if (llvm::Error field =
                  appendDomain(bytes, typed.vectorGranularityValues))
            return field;
          return appendDomain(bytes, typed.volatileValues);
        } else if constexpr (std::is_same_v<Clause,
                                            CompareExchangeContractClause>) {
          appendU64(bytes, 5);
          if (llvm::Error field = appendDomain(bytes, typed.orderingPairs))
            return field;
          if (llvm::Error field = appendDomain(bytes, typed.syncScopes))
            return field;
          if (llvm::Error field =
                  appendDomain(bytes, typed.vectorGranularityValues))
            return field;
          if (llvm::Error field = appendDomain(bytes, typed.weakValues))
            return field;
          return appendDomain(bytes, typed.volatileValues);
        } else {
          appendU64(bytes, 2);
          if (llvm::Error field = appendDomain(bytes, typed.orderings))
            return field;
          return appendDomain(bytes, typed.syncScopes);
        }
      },
      clause);
  if (error)
    return std::move(error);
  return bytes;
}

llvm::Expected<std::vector<MemoryActorContractClause>>
normalizeClauses(llvm::ArrayRef<MemoryActorContractClause> clauses) {
  if (clauses.empty())
    return invalid("memory actor contract domain must not be empty");

  ScopeCatalog scopeCatalog;
  std::vector<LoadStorePlainContractClause> plain;
  std::vector<LoadStoreAtomicContractClause> atomic;
  std::vector<AtomicRmwContractClause> rmw;
  std::vector<CompareExchangeContractClause> exchange;
  std::vector<FenceContractClause> fence;
  for (const MemoryActorContractClause &clause : clauses) {
    std::visit(
        [&](const auto &typed) {
          using Clause = std::decay_t<decltype(typed)>;
          if constexpr (std::is_same_v<Clause, LoadStorePlainContractClause>)
            plain.push_back(typed);
          else if constexpr (std::is_same_v<Clause,
                                            LoadStoreAtomicContractClause>)
            atomic.push_back(typed);
          else if constexpr (std::is_same_v<Clause, AtomicRmwContractClause>)
            rmw.push_back(typed);
          else if constexpr (std::is_same_v<Clause,
                                            CompareExchangeContractClause>)
            exchange.push_back(typed);
          else
            fence.push_back(typed);
        },
        clause);
  }

  for (const MemoryActorContractClause &clause : clauses) {
    llvm::Error scopeError = llvm::Error::success();
    std::visit(
        [&](const auto &typed) {
          using Clause = std::decay_t<decltype(typed)>;
          if constexpr (!std::is_same_v<Clause, LoadStorePlainContractClause>) {
            for (const dataflow::SyncScopeProjection &scope :
                 typed.syncScopes) {
              auto bytes = atomBytes(scope);
              if (!bytes) {
                scopeError = bytes.takeError();
                return;
              }
              scopeCatalog.try_emplace(std::move(*bytes), scope);
            }
          }
        },
        clause);
    if (scopeError)
      return std::move(scopeError);
  }

  auto reducedPlain =
      reduceClauses<LoadStorePlainContractClause>(plain, scopeCatalog);
  auto reducedAtomic =
      reduceClauses<LoadStoreAtomicContractClause>(atomic, scopeCatalog);
  auto reducedRmw = reduceClauses<AtomicRmwContractClause>(rmw, scopeCatalog);
  auto reducedExchange =
      reduceClauses<CompareExchangeContractClause>(exchange, scopeCatalog);
  auto reducedFence = reduceClauses<FenceContractClause>(fence, scopeCatalog);
  if (!reducedPlain)
    return reducedPlain.takeError();
  if (!reducedAtomic)
    return reducedAtomic.takeError();
  if (!reducedRmw)
    return reducedRmw.takeError();
  if (!reducedExchange)
    return reducedExchange.takeError();
  if (!reducedFence)
    return reducedFence.takeError();

  std::vector<MemoryActorContractClause> result;
  result.reserve(reducedPlain->size() + reducedAtomic->size() +
                 reducedRmw->size() + reducedExchange->size() +
                 reducedFence->size());
  for (auto &clause : *reducedPlain)
    result.emplace_back(std::move(clause));
  for (auto &clause : *reducedAtomic)
    result.emplace_back(std::move(clause));
  for (auto &clause : *reducedRmw)
    result.emplace_back(std::move(clause));
  for (auto &clause : *reducedExchange)
    result.emplace_back(std::move(clause));
  for (auto &clause : *reducedFence)
    result.emplace_back(std::move(clause));

  struct EncodedClause {
    std::vector<std::uint8_t> bytes;
    MemoryActorContractClause clause;
  };
  std::vector<EncodedClause> ordered;
  ordered.reserve(result.size());
  for (MemoryActorContractClause &clause : result) {
    auto bytes = encodeClause(clause);
    if (!bytes)
      return bytes.takeError();
    ordered.push_back({std::move(*bytes), std::move(clause)});
  }
  llvm::sort(ordered, [](const EncodedClause &lhs, const EncodedClause &rhs) {
    return lhs.bytes < rhs.bytes;
  });
  result.clear();
  for (EncodedClause &entry : ordered)
    result.push_back(std::move(entry.clause));
  return result;
}

llvm::Expected<std::vector<std::uint8_t>>
encodeDomainRecord(dataflow::OperationSchemaId schema,
                   llvm::ArrayRef<MemoryActorContractClause> clauses) {
  auto schemaBytes = dataflow::encodeOperationSchemaId(schema);
  if (!schemaBytes)
    return schemaBytes.takeError();
  std::vector<std::uint8_t> bytes;
  appendFramed(bytes, schemaBytes->bytes());
  appendU64(bytes, clauses.size());
  for (const MemoryActorContractClause &clause : clauses) {
    auto encoded = encodeClause(clause);
    if (!encoded)
      return encoded.takeError();
    appendFramed(bytes, *encoded);
  }
  return bytes;
}

template <typename Value>
llvm::Expected<std::vector<Value>>
readDomain(Reader &reader, llvm::StringRef field,
           mlir::MLIRContext *context = nullptr) {
  auto bytes = reader.readFrame(field);
  if (!bytes)
    return bytes.takeError();
  Reader domain(*bytes);
  auto count = domain.readU64(field + " count");
  if (!count)
    return count.takeError();
  if (*count == 0 || *count > domain.remaining() / 8)
    return invalid(field + " has an invalid count");
  std::vector<Value> values;
  values.reserve(*count);
  for (std::uint64_t index = 0; index < *count; ++index) {
    auto atom = domain.readFrame(field + " atom");
    if (!atom)
      return atom.takeError();
    if constexpr (std::is_same_v<Value, dataflow::SyncScopeProjection>) {
      auto value = decodeScopeAtom(*atom, context);
      if (!value)
        return value.takeError();
      values.push_back(*value);
    } else {
      auto value = decodeAtom(*atom, static_cast<Value *>(nullptr));
      if (!value)
        return value.takeError();
      values.push_back(*value);
    }
  }
  if (llvm::Error error = domain.finish(field))
    return std::move(error);
  return values;
}

llvm::Expected<MemoryActorContractClause>
decodeClause(llvm::ArrayRef<std::uint8_t> bytes, mlir::MLIRContext *context) {
  Reader reader(bytes);
  auto rawTag = reader.readU32("memory contract clause variant");
  if (!rawTag)
    return rawTag.takeError();
  auto fieldCount = reader.readU64("memory contract clause field count");
  if (!fieldCount)
    return fieldCount.takeError();

  switch (static_cast<ClauseTag>(*rawTag)) {
  case ClauseTag::LoadStorePlain: {
    if (*fieldCount != 1)
      return invalid("plain clause has the wrong field count");
    auto volatility = readDomain<bool>(reader, "volatile values");
    if (!volatility)
      return volatility.takeError();
    if (llvm::Error error = reader.finish("plain clause"))
      return std::move(error);
    return MemoryActorContractClause(
        LoadStorePlainContractClause{std::move(*volatility)});
  }
  case ClauseTag::LoadStoreAtomic: {
    if (*fieldCount != 4)
      return invalid("atomic access clause has the wrong field count");
    auto orderings =
        readDomain<dataflow::AtomicOrdering>(reader, "atomic orderings");
    auto scopes = readDomain<dataflow::SyncScopeProjection>(
        reader, "synchronization scopes", context);
    auto granularity =
        readDomain<std::optional<dataflow::VectorAtomicGranularity>>(
            reader, "vector granularity values");
    auto volatility = readDomain<bool>(reader, "volatile values");
    if (!orderings)
      return orderings.takeError();
    if (!scopes)
      return scopes.takeError();
    if (!granularity)
      return granularity.takeError();
    if (!volatility)
      return volatility.takeError();
    if (llvm::Error error = reader.finish("atomic access clause"))
      return std::move(error);
    return MemoryActorContractClause(LoadStoreAtomicContractClause{
        std::move(*orderings), std::move(*scopes), std::move(*granularity),
        std::move(*volatility)});
  }
  case ClauseTag::AtomicRmw: {
    if (*fieldCount != 5)
      return invalid("atomic RMW clause has the wrong field count");
    auto kinds = readDomain<dataflow::AtomicRmwKind>(reader, "RMW kinds");
    auto orderings =
        readDomain<dataflow::AtomicOrdering>(reader, "atomic orderings");
    auto scopes = readDomain<dataflow::SyncScopeProjection>(
        reader, "synchronization scopes", context);
    auto granularity =
        readDomain<std::optional<dataflow::VectorAtomicGranularity>>(
            reader, "vector granularity values");
    auto volatility = readDomain<bool>(reader, "volatile values");
    if (!kinds)
      return kinds.takeError();
    if (!orderings)
      return orderings.takeError();
    if (!scopes)
      return scopes.takeError();
    if (!granularity)
      return granularity.takeError();
    if (!volatility)
      return volatility.takeError();
    if (llvm::Error error = reader.finish("atomic RMW clause"))
      return std::move(error);
    return MemoryActorContractClause(AtomicRmwContractClause{
        std::move(*kinds), std::move(*orderings), std::move(*scopes),
        std::move(*granularity), std::move(*volatility)});
  }
  case ClauseTag::CompareExchange: {
    if (*fieldCount != 5)
      return invalid("compare-exchange clause has the wrong field count");
    auto pairs = readDomain<CompareExchangeOrderingPair>(
        reader, "compare-exchange ordering pairs");
    auto scopes = readDomain<dataflow::SyncScopeProjection>(
        reader, "synchronization scopes", context);
    auto granularity =
        readDomain<std::optional<dataflow::VectorAtomicGranularity>>(
            reader, "vector granularity values");
    auto weakness = readDomain<bool>(reader, "weak values");
    auto volatility = readDomain<bool>(reader, "volatile values");
    if (!pairs)
      return pairs.takeError();
    if (!scopes)
      return scopes.takeError();
    if (!granularity)
      return granularity.takeError();
    if (!weakness)
      return weakness.takeError();
    if (!volatility)
      return volatility.takeError();
    if (llvm::Error error = reader.finish("compare-exchange clause"))
      return std::move(error);
    return MemoryActorContractClause(CompareExchangeContractClause{
        std::move(*pairs), std::move(*scopes), std::move(*granularity),
        std::move(*weakness), std::move(*volatility)});
  }
  case ClauseTag::Fence: {
    if (*fieldCount != 2)
      return invalid("fence clause has the wrong field count");
    auto orderings =
        readDomain<dataflow::AtomicOrdering>(reader, "fence orderings");
    auto scopes = readDomain<dataflow::SyncScopeProjection>(
        reader, "synchronization scopes", context);
    if (!orderings)
      return orderings.takeError();
    if (!scopes)
      return scopes.takeError();
    if (llvm::Error error = reader.finish("fence clause"))
      return std::move(error);
    return MemoryActorContractClause(
        FenceContractClause{std::move(*orderings), std::move(*scopes)});
  }
  }
  return invalid("unknown memory contract clause variant");
}

template <typename Value>
bool containsValue(llvm::ArrayRef<Value> values, const Value &value) {
  return llvm::is_contained(values, value);
}

bool containsValue(const std::vector<bool> &values, bool value) {
  return llvm::is_contained(values, value);
}

bool containsClause(const LoadStorePlainContractClause &clause,
                    const dataflow::PlainAccessProjection &projection) {
  return containsValue(clause.volatileValues, projection.isVolatile);
}

bool containsClause(const LoadStoreAtomicContractClause &clause,
                    const dataflow::AtomicAccessProjection &projection) {
  return containsValue(llvm::ArrayRef(clause.orderings), projection.ordering) &&
         containsValue(llvm::ArrayRef(clause.syncScopes), projection.scope) &&
         containsValue(llvm::ArrayRef(clause.vectorGranularityValues),
                       projection.vectorGranularity) &&
         containsValue(clause.volatileValues, projection.isVolatile);
}

bool containsClause(const AtomicRmwContractClause &clause,
                    const dataflow::AtomicRmwProjection &projection) {
  return containsValue(llvm::ArrayRef(clause.rmwKinds), projection.kind) &&
         containsValue(llvm::ArrayRef(clause.orderings),
                       projection.access.ordering) &&
         containsValue(llvm::ArrayRef(clause.syncScopes),
                       projection.access.scope) &&
         containsValue(llvm::ArrayRef(clause.vectorGranularityValues),
                       projection.access.vectorGranularity) &&
         containsValue(clause.volatileValues, projection.access.isVolatile);
}

bool containsClause(const CompareExchangeContractClause &clause,
                    const dataflow::CompareExchangeProjection &projection) {
  return containsValue(
             llvm::ArrayRef(clause.orderingPairs),
             CompareExchangeOrderingPair{projection.successOrdering,
                                         projection.failureOrdering}) &&
         containsValue(llvm::ArrayRef(clause.syncScopes), projection.scope) &&
         containsValue(llvm::ArrayRef(clause.vectorGranularityValues),
                       projection.vectorGranularity) &&
         containsValue(clause.weakValues, projection.weak) &&
         containsValue(clause.volatileValues, projection.isVolatile);
}

bool containsClause(const FenceContractClause &clause,
                    const dataflow::FenceProjection &projection) {
  return containsValue(llvm::ArrayRef(clause.orderings), projection.ordering) &&
         containsValue(llvm::ArrayRef(clause.syncScopes), projection.scope);
}

} // namespace

llvm::Expected<MemoryActorContractDomain> MemoryActorContractDomain::create(
    dataflow::OperationSchemaId actorSchema,
    llvm::ArrayRef<MemoryActorContractClause> clauses) {
  if (static_cast<std::uint32_t>(actorSchema) >=
      dataflow::operationSchemaCount())
    return invalid("memory actor contract domain has an unknown schema");
  for (const MemoryActorContractClause &clause : clauses)
    if (llvm::Error error = validateClauseSemantics(actorSchema, clause))
      return std::move(error);
  auto normalized = normalizeClauses(clauses);
  if (!normalized)
    return normalized.takeError();
  return MemoryActorContractDomain(actorSchema, std::move(*normalized));
}

llvm::Expected<MemoryActorContractDomain>
MemoryActorContractDomain::fromCanonical(
    dataflow::OperationSchemaId actorSchema,
    llvm::ArrayRef<MemoryActorContractClause> clauses) {
  auto encoded = encodeDomainRecord(actorSchema, clauses);
  if (!encoded)
    return encoded.takeError();
  auto normalized = create(actorSchema, clauses);
  if (!normalized)
    return normalized.takeError();
  auto canonical = encodeDomainRecord(actorSchema, normalized->clauses());
  if (!canonical)
    return canonical.takeError();
  if (*encoded != *canonical)
    return invalid("memory actor contract domain is not canonical");
  return normalized;
}

bool MemoryActorContractDomain::contains(
    const dataflow::CanonicalActorSchemaProjection &actor) const {
  if (actor.schema != actorSchema_)
    return false;
  const auto *memory =
      std::get_if<dataflow::MemoryContractPayload>(&actor.payload);
  if (!memory)
    return false;
  return std::visit(
      [&](const auto &projection) {
        using Projection = std::decay_t<decltype(projection)>;
        for (const MemoryActorContractClause &clause : clauses_) {
          if constexpr (std::is_same_v<Projection,
                                       dataflow::PlainAccessProjection>) {
            if (const auto *typed =
                    std::get_if<LoadStorePlainContractClause>(&clause))
              if (containsClause(*typed, projection))
                return true;
          } else if constexpr (std::is_same_v<
                                   Projection,
                                   dataflow::AtomicAccessProjection>) {
            if (const auto *typed =
                    std::get_if<LoadStoreAtomicContractClause>(&clause))
              if (containsClause(*typed, projection))
                return true;
          } else if constexpr (std::is_same_v<Projection,
                                              dataflow::AtomicRmwProjection>) {
            if (const auto *typed =
                    std::get_if<AtomicRmwContractClause>(&clause))
              if (containsClause(*typed, projection))
                return true;
          } else if constexpr (std::is_same_v<
                                   Projection,
                                   dataflow::CompareExchangeProjection>) {
            if (const auto *typed =
                    std::get_if<CompareExchangeContractClause>(&clause))
              if (containsClause(*typed, projection))
                return true;
          } else {
            if (const auto *typed = std::get_if<FenceContractClause>(&clause))
              if (containsClause(*typed, projection))
                return true;
          }
        }
        return false;
      },
      *memory);
}

llvm::Expected<std::vector<std::uint8_t>>
encodeMemoryActorContractDomain(const MemoryActorContractDomain &domain) {
  return encodeDomainRecord(domain.actorSchema(), domain.clauses());
}

llvm::Expected<MemoryActorContractDomain>
decodeMemoryActorContractDomain(llvm::ArrayRef<std::uint8_t> bytes,
                                mlir::MLIRContext *context) {
  if (!context)
    return invalid("memory actor contract import requires an MLIR context");
  Reader reader(bytes);
  auto schemaBytes = reader.readFrame("memory actor schema");
  if (!schemaBytes)
    return schemaBytes.takeError();
  auto schema = dataflow::decodeOperationSchemaId(*schemaBytes);
  if (!schema)
    return schema.takeError();
  auto clauseCount = reader.readU64("memory actor contract clause count");
  if (!clauseCount)
    return clauseCount.takeError();
  if (*clauseCount == 0 || *clauseCount > reader.remaining() / 8)
    return invalid("memory actor contract domain has an invalid clause count");
  std::vector<MemoryActorContractClause> clauses;
  clauses.reserve(*clauseCount);
  for (std::uint64_t index = 0; index < *clauseCount; ++index) {
    auto clauseBytes = reader.readFrame("memory actor contract clause");
    if (!clauseBytes)
      return clauseBytes.takeError();
    auto clause = decodeClause(*clauseBytes, context);
    if (!clause)
      return clause.takeError();
    clauses.push_back(std::move(*clause));
  }
  if (llvm::Error error = reader.finish("memory actor contract domain"))
    return std::move(error);
  auto result = MemoryActorContractDomain::fromCanonical(*schema, clauses);
  if (!result)
    return result.takeError();
  auto canonical = encodeMemoryActorContractDomain(*result);
  if (!canonical)
    return canonical.takeError();
  if (llvm::ArrayRef<std::uint8_t>(*canonical) != bytes)
    return invalid("memory actor contract domain bytes are not canonical");
  return result;
}

} // namespace fabric
