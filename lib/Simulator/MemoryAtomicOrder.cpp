#include "Simulator/MemoryAtomicOrder.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>
#include <utility>

namespace loom {
namespace sim {
namespace {

std::string describe(const AtomicObjectKey &key) {
  return ("atomic object (root " + llvm::Twine(key.logicalRootId) +
          ", offset " + llvm::Twine(key.canonicalByteOffset) + ", size " +
          llvm::Twine(key.accessByteSize) + ")")
      .str();
}

llvm::Error reject(MemoryAtomicOrderError::Kind kind,
                   const llvm::Twine &message) {
  return llvm::make_error<MemoryAtomicOrderError>(kind, message.str());
}

/// An access byte size is the exact width of one atomic object, so a zero size
/// names no object at all.
llvm::Error validateKey(const AtomicObjectKey &key) {
  if (key.accessByteSize == 0)
    return reject(MemoryAtomicOrderError::Kind::InvalidKey,
                  describe(key) + " has no access byte size");
  return llvm::Error::success();
}

/// The shared requirement of an rmw and a successful compare-exchange: both
/// read the immediately preceding version, so a selection the order has already
/// moved past cannot be the version they append behind.
llvm::Error requireCurrentTail(llvm::ArrayRef<AtomicVersionId> order,
                               const AtomicObjectKey &key,
                               AtomicVersionId version) {
  if (version != order.back())
    return reject(MemoryAtomicOrderError::Kind::StaleVersion,
                  "version " + llvm::Twine(version.value()) +
                      " is no longer the tail of " + describe(key));
  return llvm::Error::success();
}

} // namespace

char MemoryAtomicOrderError::ID = 0;

MemoryAtomicOrderError::MemoryAtomicOrderError(Kind kind, std::string message)
    : kind_(kind), message_(std::move(message)) {}

void MemoryAtomicOrderError::log(llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code MemoryAtomicOrderError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::Expected<MemoryAtomicOrder::VersionOrder *>
MemoryAtomicOrder::resolveObject(const AtomicObjectKey &key) {
  if (llvm::Error error = validateKey(key))
    return std::move(error);
  auto object = objects_.find(key);
  if (object == objects_.end())
    return reject(MemoryAtomicOrderError::Kind::UnknownObject,
                  describe(key) + " has no explicit initial version");
  return &object->second;
}

llvm::Error
MemoryAtomicOrder::validateSelection(const AtomicObjectKey &key,
                                     AtomicVersionId version) const {
  if (version.value() >= versions_.size())
    return reject(MemoryAtomicOrderError::Kind::UnknownVersion,
                  "version " + llvm::Twine(version.value()) +
                      " was never produced by this order");
  if (versions_[version.value()].key() != key)
    return reject(MemoryAtomicOrderError::Kind::ForeignVersion,
                  "version " + llvm::Twine(version.value()) +
                      " belongs to another atomic object than " +
                      describe(key));
  return llvm::Error::success();
}

AtomicVersionId
MemoryAtomicOrder::appendVersion(VersionOrder &order,
                                 const AtomicObjectKey &key,
                                 std::optional<AtomicVersionId> predecessor) {
  AtomicVersionId id(versions_.size());
  versions_.push_back(AtomicVersionRecord(id, key, predecessor));
  order.push_back(id);
  return id;
}

AtomicReadId MemoryAtomicOrder::recordRead(const AtomicObjectKey &key,
                                           AtomicVersionId version) {
  AtomicReadId id(reads_.size());
  reads_.push_back(AtomicReadRecord(id, key, version));
  return id;
}

llvm::Expected<AtomicVersionId>
MemoryAtomicOrder::initializeObject(const AtomicObjectKey &key) {
  if (llvm::Error error = validateKey(key))
    return std::move(error);
  // A key that already names an object keeps its order untouched: try_emplace
  // inserts only when this call owns the initial version.
  auto [object, initialized] = objects_.try_emplace(key);
  if (!initialized)
    return reject(MemoryAtomicOrderError::Kind::DuplicateInitialization,
                  describe(key) + " already has an initial version");
  return appendVersion(object->second, key, std::nullopt);
}

llvm::Expected<AtomicVersionId>
MemoryAtomicOrder::atomicStore(const AtomicObjectKey &key) {
  llvm::Expected<VersionOrder *> object = resolveObject(key);
  if (!object)
    return object.takeError();

  VersionOrder &order = **object;
  return appendVersion(order, key, order.back());
}

llvm::Expected<AtomicReadId>
MemoryAtomicOrder::atomicLoad(const AtomicObjectKey &key,
                              AtomicVersionId source) {
  llvm::Expected<VersionOrder *> object = resolveObject(key);
  if (!object)
    return object.takeError();
  if (llvm::Error error = validateSelection(key, source))
    return std::move(error);

  return recordRead(key, source);
}

llvm::Expected<AtomicRmwResult>
MemoryAtomicOrder::atomicRmw(const AtomicObjectKey &key,
                             AtomicVersionId predecessor) {
  llvm::Expected<VersionOrder *> object = resolveObject(key);
  if (!object)
    return object.takeError();
  VersionOrder &order = **object;
  if (llvm::Error error = validateSelection(key, predecessor))
    return std::move(error);
  if (llvm::Error error = requireCurrentTail(order, key, predecessor))
    return std::move(error);

  AtomicReadId read = recordRead(key, predecessor);
  AtomicVersionId version = appendVersion(order, key, predecessor);
  return AtomicRmwResult{read, version};
}

llvm::Expected<AtomicCompareExchangeResult>
MemoryAtomicOrder::compareExchange(const AtomicObjectKey &key,
                                   AtomicVersionId source,
                                   AtomicCompareExchangeDecision decision) {
  llvm::Expected<VersionOrder *> object = resolveObject(key);
  if (!object)
    return object.takeError();
  VersionOrder &order = **object;
  if (llvm::Error error = validateSelection(key, source))
    return std::move(error);

  switch (decision) {
  case AtomicCompareExchangeDecision::Success: {
    if (llvm::Error error = requireCurrentTail(order, key, source))
      return std::move(error);
    AtomicReadId read = recordRead(key, source);
    AtomicVersionId version = appendVersion(order, key, source);
    return AtomicCompareExchangeResult{read, version};
  }
  case AtomicCompareExchangeDecision::ComparisonFailure:
  case AtomicCompareExchangeDecision::SpuriousFailure:
    break;
  }
  return AtomicCompareExchangeResult{recordRead(key, source), std::nullopt};
}

std::optional<llvm::ArrayRef<AtomicVersionId>>
MemoryAtomicOrder::modificationOrder(const AtomicObjectKey &key) const {
  auto object = objects_.find(key);
  if (object == objects_.end())
    return std::nullopt;
  return llvm::ArrayRef<AtomicVersionId>(object->second);
}

std::optional<AtomicVersionRecord>
MemoryAtomicOrder::versionRecord(AtomicVersionId id) const {
  if (id.value() >= versions_.size())
    return std::nullopt;
  return versions_[id.value()];
}

std::optional<AtomicReadRecord>
MemoryAtomicOrder::readRecord(AtomicReadId id) const {
  if (id.value() >= reads_.size())
    return std::nullopt;
  return reads_[id.value()];
}

} // namespace sim
} // namespace loom
