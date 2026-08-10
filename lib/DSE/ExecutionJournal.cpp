#include "DSE/ExecutionJournal.h"

#include "Common/ArtifactLocalReference.h"
#include "DSE/ResolvedConfigView.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <system_error>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

constexpr llvm::StringLiteral kSnapshotName = "execution-journal.snapshot";
constexpr llvm::StringLiteral kSnapshotIdentity = "loom.dse.execution_journal";
constexpr SchemaVersion kSnapshotVersion{1, 0};
constexpr std::uint64_t kMaximumSnapshotBytes = 64ULL * 1024ULL * 1024ULL;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "execution_journal_invalid: " + message);
}

llvm::Error filesystemError(llvm::StringRef operation) {
  return llvm::createStringError(
      std::error_code(errno, std::generic_category()),
      "execution journal " + operation);
}

bool canonicalAscii(llvm::StringRef value) {
  if (value.empty())
    return false;
  return llvm::all_of(value, [](unsigned char character) {
    return character >= 0x21 && character <= 0x7e;
  });
}

class Encoder final {
public:
  void u32(std::uint32_t value) {
    for (int shift = 24; shift >= 0; shift -= 8)
      bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
  }
  void u64(std::uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8)
      bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
  }
  void bytes(llvm::ArrayRef<std::uint8_t> bytes) {
    bytes_.insert(bytes_.end(), bytes.begin(), bytes.end());
  }
  void string(llvm::StringRef value) {
    u64(value.size());
    bytes(llvm::ArrayRef<std::uint8_t>(
        reinterpret_cast<const std::uint8_t *>(value.data()), value.size()));
  }
  std::vector<std::uint8_t> take() { return std::move(bytes_); }

private:
  std::vector<std::uint8_t> bytes_;
};

class Decoder final {
public:
  explicit Decoder(llvm::ArrayRef<std::uint8_t> bytes) : bytes_(bytes) {}

  llvm::Expected<std::uint32_t> u32(llvm::StringRef field) {
    if (bytes_.size() - offset_ < 4)
      return invalid("truncated " + field);
    std::uint32_t value = 0;
    for (unsigned index = 0; index != 4; ++index)
      value = (value << 8) | bytes_[offset_++];
    return value;
  }
  llvm::Expected<std::uint64_t> u64(llvm::StringRef field) {
    if (bytes_.size() - offset_ < 8)
      return invalid("truncated " + field);
    std::uint64_t value = 0;
    for (unsigned index = 0; index != 8; ++index)
      value = (value << 8) | bytes_[offset_++];
    return value;
  }
  llvm::Expected<llvm::ArrayRef<std::uint8_t>> bytes(std::size_t count,
                                                     llvm::StringRef field) {
    if (bytes_.size() - offset_ < count)
      return invalid("truncated " + field);
    llvm::ArrayRef<std::uint8_t> result = bytes_.slice(offset_, count);
    offset_ += count;
    return result;
  }
  llvm::Expected<std::string> string(llvm::StringRef field) {
    auto count = u64((field + " length").str());
    if (!count)
      return count.takeError();
    if (*count > std::numeric_limits<std::size_t>::max())
      return invalid(field + " length exceeds size_t");
    auto value = bytes(static_cast<std::size_t>(*count), field);
    if (!value)
      return value.takeError();
    return std::string(reinterpret_cast<const char *>(value->data()),
                       value->size());
  }
  std::size_t remaining() const { return bytes_.size() - offset_; }
  bool empty() const { return offset_ == bytes_.size(); }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
  std::size_t offset_ = 0;
};

void encodeDescriptor(Encoder &encoder,
                      const WorkUnitDescriptorRef &descriptor) {
  encoder.string(descriptor.ownerRegistryIdentity());
  encoder.u32(descriptor.ownerRegistryVersion().major);
  encoder.u32(descriptor.ownerRegistryVersion().minor);
  encoder.u32(descriptor.ownerLocalKind());
}

llvm::Expected<WorkUnitDescriptorRef> decodeDescriptor(Decoder &decoder) {
  auto identity = decoder.string("work descriptor owner identity");
  if (!identity)
    return identity.takeError();
  auto major = decoder.u32("work descriptor owner major version");
  if (!major)
    return major.takeError();
  auto minor = decoder.u32("work descriptor owner minor version");
  if (!minor)
    return minor.takeError();
  auto kind = decoder.u32("work descriptor local kind");
  if (!kind)
    return kind.takeError();
  return WorkUnitDescriptorRef::get(*identity, {*major, *minor}, *kind);
}

void encodeKey(Encoder &encoder, const WorkUnitKey &key) {
  encoder.u64(key.planNodeOrdinal());
  encodeDescriptor(encoder, key.descriptor());
  encoder.u64(key.stableOrdinal());
}

llvm::Expected<WorkUnitKey> decodeKey(Decoder &decoder) {
  auto planNode = decoder.u64("work key plan node");
  if (!planNode)
    return planNode.takeError();
  auto descriptor = decodeDescriptor(decoder);
  if (!descriptor)
    return descriptor.takeError();
  auto ordinal = decoder.u64("work key stable ordinal");
  if (!ordinal)
    return ordinal.takeError();
  return WorkUnitKey::get(*planNode, std::move(*descriptor), *ordinal);
}

bool terminal(JournalWorkUnitStatus status) {
  return status == JournalWorkUnitStatus::Completed ||
         status == JournalWorkUnitStatus::Failed ||
         status == JournalWorkUnitStatus::TimedOut ||
         status == JournalWorkUnitStatus::Unsupported;
}

bool boundedCount(std::uint64_t count, std::size_t remaining,
                  std::size_t minimumEncodedWidth = 1) {
  return count <= kMaximumSnapshotBytes &&
         count <= std::numeric_limits<std::size_t>::max() &&
         count <= remaining / minimumEncodedWidth;
}

std::vector<ArtifactRootReference>
generateResultRoots(const CandidateGeneratorProviderResult &result) {
  std::vector<ArtifactRootReference> roots;
  std::visit(
      [&](const auto &outcome) {
        using T = std::decay_t<decltype(outcome)>;
        const auto &bindings = [&]() -> const auto & {
          if constexpr (std::is_same_v<T, CompletedCandidateGeneratorResult>)
            return outcome.outputBindings;
          else
            return outcome.retainedOutputBindings;
        }();
        for (const CandidateGeneratorOutputBinding &binding : bindings)
          roots.insert(roots.end(), binding.artifacts.begin(),
                       binding.artifacts.end());
      },
      result.outcome);
  llvm::sort(roots, artifactRootReferenceLess);
  roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
  return roots;
}

llvm::Expected<std::uint64_t> unixNanosecondsNow() {
  const auto elapsed = std::chrono::system_clock::now().time_since_epoch();
  const auto count =
      std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
  if (count <= 0)
    return invalid("system clock cannot represent a positive observation time");
  return static_cast<std::uint64_t>(count);
}

bool intervalLess(const JournalActiveWallInterval &lhs,
                  const JournalActiveWallInterval &rhs) {
  if (lhs.beginUnixTimeNanoseconds != rhs.beginUnixTimeNanoseconds)
    return lhs.beginUnixTimeNanoseconds < rhs.beginUnixTimeNanoseconds;
  return lhs.endUnixTimeNanoseconds < rhs.endUnixTimeNanoseconds;
}

llvm::Error
validateIntervals(llvm::ArrayRef<JournalActiveWallInterval> intervals) {
  std::uint64_t total = 0;
  for (std::size_t index = 0; index != intervals.size(); ++index) {
    const JournalActiveWallInterval &interval = intervals[index];
    if (interval.beginUnixTimeNanoseconds == 0 ||
        interval.beginUnixTimeNanoseconds >= interval.endUnixTimeNanoseconds)
      return invalid("active wall interval is empty or malformed");
    if (index != 0 && intervals[index - 1].endUnixTimeNanoseconds >=
                          interval.beginUnixTimeNanoseconds)
      return invalid("active wall intervals overlap or are not canonical");
    const std::uint64_t duration =
        interval.endUnixTimeNanoseconds - interval.beginUnixTimeNanoseconds;
    if (duration > std::numeric_limits<std::uint64_t>::max() - total)
      return invalid("active wall interval total overflows uint64");
    total += duration;
  }
  return llvm::Error::success();
}

llvm::Error addInterval(JournalWorkUnitRecord &record,
                        std::uint64_t activeWallTimeNanoseconds,
                        std::uint64_t observedUnixTimeNanoseconds) {
  if (record.activeAttemptStartUnixTimeNanoseconds != 0 &&
      observedUnixTimeNanoseconds <
          record.activeAttemptStartUnixTimeNanoseconds)
    return invalid("active interval ends before its recorded start");
  if (activeWallTimeNanoseconds == 0) {
    record.activeAttemptStartUnixTimeNanoseconds = 0;
    return llvm::Error::success();
  }
  if (observedUnixTimeNanoseconds == 0 ||
      activeWallTimeNanoseconds >= observedUnixTimeNanoseconds)
    return invalid("active wall interval predates the Unix clock epoch");
  record.activeWallIntervals.push_back(
      {observedUnixTimeNanoseconds - activeWallTimeNanoseconds,
       observedUnixTimeNanoseconds});
  llvm::sort(record.activeWallIntervals, intervalLess);
  std::vector<JournalActiveWallInterval> merged;
  merged.reserve(record.activeWallIntervals.size());
  for (const JournalActiveWallInterval &interval : record.activeWallIntervals) {
    if (merged.empty() || merged.back().endUnixTimeNanoseconds <
                              interval.beginUnixTimeNanoseconds) {
      merged.push_back(interval);
      continue;
    }
    merged.back().endUnixTimeNanoseconds = std::max(
        merged.back().endUnixTimeNanoseconds, interval.endUnixTimeNanoseconds);
  }
  record.activeWallIntervals = std::move(merged);
  record.activeAttemptStartUnixTimeNanoseconds = 0;
  return validateIntervals(record.activeWallIntervals);
}

llvm::Error settleStartedInterval(JournalWorkUnitRecord &record,
                                  std::uint64_t observedUnixTimeNanoseconds) {
  if (record.activeAttemptStartUnixTimeNanoseconds == 0)
    return llvm::Error::success();
  if (observedUnixTimeNanoseconds <
      record.activeAttemptStartUnixTimeNanoseconds)
    return invalid("system clock moved backwards during active work");
  const std::uint64_t duration = observedUnixTimeNanoseconds -
                                 record.activeAttemptStartUnixTimeNanoseconds;
  return addInterval(record, duration, observedUnixTimeNanoseconds);
}

llvm::Error validateRecord(const JournalWorkUnitRecord &record) {
  if (static_cast<std::uint32_t>(record.status) >
      static_cast<std::uint32_t>(JournalWorkUnitStatus::Unsupported))
    return invalid("work record has an unknown status");
  if (record.status == JournalWorkUnitStatus::Prepared &&
      !record.preparedInvocation)
    return invalid("prepared work has no invocation handle");
  if ((record.status == JournalWorkUnitStatus::Queued ||
       record.status == JournalWorkUnitStatus::Running) &&
      record.preparedInvocation)
    return invalid("unprepared work carries a prepared invocation");
  if (llvm::Error error = validateIntervals(record.activeWallIntervals))
    return error;
  if (record.status == JournalWorkUnitStatus::Running &&
      record.activeAttemptStartUnixTimeNanoseconds == 0)
    return invalid("running work has no active interval start");
  if (record.status == JournalWorkUnitStatus::Queued &&
      record.activeAttemptStartUnixTimeNanoseconds != 0)
    return invalid("queued work carries an active interval start");
  if (terminal(record.status) &&
      record.activeAttemptStartUnixTimeNanoseconds != 0)
    return invalid("terminal work carries an active interval start");
  if (!terminal(record.status) && (record.terminalUnixTimeNanoseconds != 0 ||
                                   !record.finalizedOutputs.empty()))
    return invalid("nonterminal work carries terminal observations");
  if (terminal(record.status) && record.terminalUnixTimeNanoseconds == 0)
    return invalid("terminal work has no completion timestamp");
  if (terminal(record.status) && !record.activeWallIntervals.empty() &&
      record.activeWallIntervals.back().endUnixTimeNanoseconds >
          record.terminalUnixTimeNanoseconds)
    return invalid("active wall interval ends after terminal completion");
  if (!llvm::is_sorted(record.finalizedOutputs, artifactRootReferenceLess) ||
      std::adjacent_find(record.finalizedOutputs.begin(),
                         record.finalizedOutputs.end()) !=
          record.finalizedOutputs.end())
    return invalid("finalized output roots are not canonical and unique");
  if (record.preparedInvocation &&
      record.preparedInvocation->bundleRoot.empty())
    return invalid("prepared invocation has an empty bundle root");
  return llvm::Error::success();
}

void encodeRecord(Encoder &encoder, const JournalWorkUnitRecord &record) {
  encodeKey(encoder, record.key);
  encoder.u32(static_cast<std::uint32_t>(record.status));
  encoder.u64(record.activeWallIntervals.size());
  for (const JournalActiveWallInterval &interval : record.activeWallIntervals) {
    encoder.u64(interval.beginUnixTimeNanoseconds);
    encoder.u64(interval.endUnixTimeNanoseconds);
  }
  encoder.u64(record.activeAttemptStartUnixTimeNanoseconds);
  encoder.u64(record.terminalUnixTimeNanoseconds);
  encoder.u64(record.finalizedOutputs.size());
  for (const ArtifactRootReference &output : record.finalizedOutputs) {
    const std::vector<std::uint8_t> encoded =
        encodeArtifactRootReference(output);
    encoder.u64(encoded.size());
    encoder.bytes(encoded);
  }
  encoder.u32(record.preparedInvocation ? 1 : 0);
  if (record.preparedInvocation) {
    encoder.string(record.preparedInvocation->bundleRoot);
    encoder.bytes(record.preparedInvocation->manifestDigest.bytes());
  }
}

llvm::Expected<JournalWorkUnitRecord> decodeRecord(Decoder &decoder) {
  auto key = decodeKey(decoder);
  if (!key)
    return key.takeError();
  auto rawStatus = decoder.u32("work status");
  if (!rawStatus)
    return rawStatus.takeError();
  if (*rawStatus >
      static_cast<std::uint32_t>(JournalWorkUnitStatus::Unsupported))
    return invalid("work record has an unknown status");
  auto intervalCount = decoder.u64("work active interval count");
  if (!intervalCount)
    return intervalCount.takeError();
  if (!boundedCount(*intervalCount, decoder.remaining(),
                    2 * sizeof(std::uint64_t)))
    return invalid("work active interval count exceeds the snapshot bound");
  std::vector<JournalActiveWallInterval> intervals;
  intervals.reserve(static_cast<std::size_t>(*intervalCount));
  for (std::uint64_t index = 0; index != *intervalCount; ++index) {
    auto begin = decoder.u64("work active interval begin");
    if (!begin)
      return begin.takeError();
    auto end = decoder.u64("work active interval end");
    if (!end)
      return end.takeError();
    intervals.push_back({*begin, *end});
  }
  auto activeStart = decoder.u64("work active interval start");
  if (!activeStart)
    return activeStart.takeError();
  auto terminalTime = decoder.u64("work terminal timestamp");
  if (!terminalTime)
    return terminalTime.takeError();
  auto outputCount = decoder.u64("finalized output count");
  if (!outputCount)
    return outputCount.takeError();
  if (!boundedCount(*outputCount, decoder.remaining(), sizeof(std::uint64_t)))
    return invalid("finalized output count exceeds the snapshot bound");
  std::vector<ArtifactRootReference> outputs;
  outputs.reserve(static_cast<std::size_t>(*outputCount));
  for (std::uint64_t index = 0; index != *outputCount; ++index) {
    auto encodedSize = decoder.u64("finalized output byte count");
    if (!encodedSize)
      return encodedSize.takeError();
    if (!boundedCount(*encodedSize, decoder.remaining()))
      return invalid("finalized output byte count exceeds the snapshot bound");
    auto encoded = decoder.bytes(static_cast<std::size_t>(*encodedSize),
                                 "finalized output");
    if (!encoded)
      return encoded.takeError();
    auto decoded = decodeArtifactRootReferencePrefix(*encoded);
    if (!decoded)
      return decoded.takeError();
    if (decoded->byteCount != encoded->size())
      return invalid("finalized output has trailing bytes");
    outputs.push_back(std::move(decoded->reference));
  }
  auto hasPrepared = decoder.u32("prepared invocation presence");
  if (!hasPrepared)
    return hasPrepared.takeError();
  if (*hasPrepared > 1)
    return invalid("prepared invocation presence is not boolean");
  std::optional<external_tool::PreparedExternalToolInvocation> prepared;
  if (*hasPrepared == 1) {
    auto root = decoder.string("prepared bundle root");
    if (!root)
      return root.takeError();
    auto digestBytes =
        decoder.bytes(BlobDigest::byteSize, "prepared manifest digest");
    if (!digestBytes)
      return digestBytes.takeError();
    auto digest = BlobDigest::fromBytes(*digestBytes);
    if (!digest)
      return digest.takeError();
    prepared = external_tool::PreparedExternalToolInvocation{
        std::move(*root), std::move(*digest)};
  }
  JournalWorkUnitRecord record{
      std::move(*key),      static_cast<JournalWorkUnitStatus>(*rawStatus),
      std::move(intervals), *activeStart,
      *terminalTime,        std::move(outputs),
      std::move(prepared)};
  if (llvm::Error error = validateRecord(record))
    return error;
  return record;
}

llvm::Expected<int> openRunDirectory(llvm::StringRef root) {
  const int directory = ::open(root.str().c_str(),
                               O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (directory < 0)
    return filesystemError("cannot open run directory");
  return directory;
}

llvm::Error closeDescriptor(int descriptor, llvm::StringRef description) {
  if (::close(descriptor) == 0)
    return llvm::Error::success();
  return filesystemError(("cannot close " + description).str());
}

llvm::Expected<std::optional<std::vector<std::uint8_t>>>
readSnapshot(int directory) {
  const int file = ::openat(directory, kSnapshotName.str().c_str(),
                            O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (file < 0) {
    if (errno == ENOENT)
      return std::optional<std::vector<std::uint8_t>>{};
    return filesystemError("cannot open snapshot");
  }
  struct stat status{};
  if (::fstat(file, &status) != 0) {
    llvm::consumeError(closeDescriptor(file, "snapshot"));
    return filesystemError("cannot stat snapshot");
  }
  if (!S_ISREG(status.st_mode) || status.st_size < 0 ||
      static_cast<std::uint64_t>(status.st_size) > kMaximumSnapshotBytes) {
    llvm::consumeError(closeDescriptor(file, "snapshot"));
    return invalid("snapshot is not a bounded regular file");
  }
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(status.st_size));
  std::size_t offset = 0;
  while (offset != bytes.size()) {
    const ssize_t amount =
        ::read(file, bytes.data() + offset, bytes.size() - offset);
    if (amount < 0) {
      if (errno == EINTR)
        continue;
      llvm::consumeError(closeDescriptor(file, "snapshot"));
      return filesystemError("cannot read snapshot");
    }
    if (amount == 0) {
      llvm::consumeError(closeDescriptor(file, "snapshot"));
      return invalid("snapshot was truncated while reading");
    }
    offset += static_cast<std::size_t>(amount);
  }
  if (llvm::Error error = closeDescriptor(file, "snapshot"))
    return error;
  return std::optional<std::vector<std::uint8_t>>(std::move(bytes));
}

llvm::Error writeAll(int file, llvm::ArrayRef<std::uint8_t> bytes) {
  std::size_t offset = 0;
  while (offset != bytes.size()) {
    const ssize_t amount =
        ::write(file, bytes.data() + offset, bytes.size() - offset);
    if (amount < 0) {
      if (errno == EINTR)
        continue;
      return filesystemError("cannot write snapshot");
    }
    offset += static_cast<std::size_t>(amount);
  }
  return llvm::Error::success();
}

llvm::Error publishSnapshot(int directory, llvm::ArrayRef<std::uint8_t> bytes) {
  static std::atomic<std::uint64_t> counter{0};
  constexpr std::uint64_t maximumNameAttempts = 1024;
  std::string temporary;
  int file = -1;
  for (std::uint64_t attempt = 0; attempt != maximumNameAttempts; ++attempt) {
    const std::uint64_t temporaryOrdinal =
        counter.fetch_add(1, std::memory_order_relaxed) + 1;
    temporary = (".execution-journal.snapshot.partial." +
                 llvm::Twine(::getpid()) + "." + llvm::Twine(temporaryOrdinal))
                    .str();
    file = ::openat(directory, temporary.c_str(),
                    O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (file >= 0)
      break;
    if (errno != EEXIST)
      return filesystemError("cannot create temporary snapshot");
  }
  if (file < 0)
    return invalid("cannot allocate a unique temporary snapshot name");
  llvm::Error error = writeAll(file, bytes);
  if (!error && ::fsync(file) != 0)
    error = filesystemError("cannot sync temporary snapshot");
  error = llvm::joinErrors(std::move(error),
                           closeDescriptor(file, "temporary snapshot"));
  if (error) {
    (void)::unlinkat(directory, temporary.c_str(), 0);
    return error;
  }
  if (::renameat(directory, temporary.c_str(), directory,
                 kSnapshotName.str().c_str()) != 0) {
    (void)::unlinkat(directory, temporary.c_str(), 0);
    return filesystemError("cannot publish snapshot atomically");
  }
  if (::fsync(directory) != 0)
    return filesystemError("cannot sync run directory");
  return llvm::Error::success();
}

} // namespace

struct ExecutionJournal::State final {
  State(std::string localRunRoot, DseRunKey runKey,
        ComponentViewDigest viewDigest)
      : localRunRoot(std::move(localRunRoot)), runKey(std::move(runKey)),
        viewDigest(viewDigest) {}

  std::string localRunRoot;
  DseRunKey runKey;
  ComponentViewDigest viewDigest;
  bool stopRequested = false;
  std::vector<JournalWorkUnitRecord> workUnits;
  std::map<WorkUnitKey, CandidateGeneratorProviderResult>
      transientGenerateResults;
  mutable std::mutex mutex;
};

llvm::Expected<WorkUnitDescriptorRef>
WorkUnitDescriptorRef::get(llvm::StringRef ownerRegistryIdentity,
                           SchemaVersion ownerRegistryVersion,
                           std::uint32_t ownerLocalKind) {
  if (!canonicalAscii(ownerRegistryIdentity))
    return invalid("work descriptor owner is not canonical ASCII");
  return WorkUnitDescriptorRef(ownerRegistryIdentity.str(),
                               ownerRegistryVersion, ownerLocalKind);
}

bool operator<(const WorkUnitDescriptorRef &lhs,
               const WorkUnitDescriptorRef &rhs) {
  if (lhs.ownerRegistryIdentity_ != rhs.ownerRegistryIdentity_)
    return lhs.ownerRegistryIdentity_ < rhs.ownerRegistryIdentity_;
  if (lhs.ownerRegistryVersion_.major != rhs.ownerRegistryVersion_.major)
    return lhs.ownerRegistryVersion_.major < rhs.ownerRegistryVersion_.major;
  if (lhs.ownerRegistryVersion_.minor != rhs.ownerRegistryVersion_.minor)
    return lhs.ownerRegistryVersion_.minor < rhs.ownerRegistryVersion_.minor;
  return lhs.ownerLocalKind_ < rhs.ownerLocalKind_;
}

llvm::Expected<WorkUnitKey> WorkUnitKey::get(std::uint64_t planNodeOrdinal,
                                             WorkUnitDescriptorRef descriptor,
                                             std::uint64_t stableOrdinal) {
  return WorkUnitKey(planNodeOrdinal, std::move(descriptor), stableOrdinal);
}

bool operator<(const WorkUnitKey &lhs, const WorkUnitKey &rhs) {
  if (lhs.planNodeOrdinal_ != rhs.planNodeOrdinal_)
    return lhs.planNodeOrdinal_ < rhs.planNodeOrdinal_;
  if (lhs.descriptor_ < rhs.descriptor_)
    return true;
  if (rhs.descriptor_ < lhs.descriptor_)
    return false;
  return lhs.stableOrdinal_ < rhs.stableOrdinal_;
}

std::uint64_t JournalWorkUnitRecord::activeWallTimeNanoseconds() const {
  std::uint64_t total = 0;
  for (const JournalActiveWallInterval &interval : activeWallIntervals)
    total +=
        interval.endUnixTimeNanoseconds - interval.beginUnixTimeNanoseconds;
  return total;
}

namespace {

template <typename StateT>
std::vector<std::uint8_t> encodeState(const StateT &state) {
  Encoder encoder;
  encoder.string(kSnapshotIdentity);
  encoder.u32(kSnapshotVersion.major);
  encoder.u32(kSnapshotVersion.minor);
  encoder.bytes(state.runKey.bytes());
  encoder.bytes(state.viewDigest.bytes());
  encoder.u32(state.stopRequested ? 1 : 0);
  encoder.u64(state.workUnits.size());
  for (const JournalWorkUnitRecord &record : state.workUnits)
    encodeRecord(encoder, record);
  return encoder.take();
}

template <typename StateT> llvm::Error flushLocked(const StateT &state) {
  auto directory = openRunDirectory(state.localRunRoot);
  if (!directory)
    return directory.takeError();
  std::vector<std::uint8_t> bytes = encodeState(state);
  llvm::Error error = bytes.size() > kMaximumSnapshotBytes
                          ? invalid("encoded snapshot exceeds the size bound")
                          : publishSnapshot(*directory, bytes);
  return llvm::joinErrors(std::move(error),
                          closeDescriptor(*directory, "run directory"));
}

llvm::Expected<std::pair<bool, std::vector<JournalWorkUnitRecord>>>
decodeState(llvm::ArrayRef<std::uint8_t> bytes, const DseRunKey &runKey,
            const ComponentViewDigest &viewDigest) {
  Decoder decoder(bytes);
  auto identity = decoder.string("journal identity");
  if (!identity)
    return identity.takeError();
  auto major = decoder.u32("journal major version");
  if (!major)
    return major.takeError();
  auto minor = decoder.u32("journal minor version");
  if (!minor)
    return minor.takeError();
  if (*identity != kSnapshotIdentity || *major != kSnapshotVersion.major ||
      *minor != kSnapshotVersion.minor)
    return invalid("snapshot schema is unsupported");
  auto runKeyBytes = decoder.bytes(DseRunKey::byteSize, "run key");
  if (!runKeyBytes)
    return runKeyBytes.takeError();
  auto decodedRunKey = DseRunKey::fromBytes(*runKeyBytes);
  if (!decodedRunKey)
    return decodedRunKey.takeError();
  if (*decodedRunKey != runKey)
    return invalid("snapshot belongs to another semantic run");
  auto digestBytes =
      decoder.bytes(ComponentViewDigest::byteSize, "resolved view digest");
  if (!digestBytes)
    return digestBytes.takeError();
  auto decodedDigest = ComponentViewDigest::fromBytes(*digestBytes);
  if (!decodedDigest)
    return decodedDigest.takeError();
  if (*decodedDigest != viewDigest)
    return invalid("snapshot belongs to another resolved DSE plan");
  auto stopRequested = decoder.u32("graceful stop flag");
  if (!stopRequested)
    return stopRequested.takeError();
  if (*stopRequested > 1)
    return invalid("graceful stop flag is not boolean");
  auto count = decoder.u64("work record count");
  if (!count)
    return count.takeError();
  constexpr std::size_t minimumRecordBytes =
      8 + 8 + 1 + 4 + 4 + 4 + 8 + 4 + 8 + 8 + 8 + 8 + 4;
  if (!boundedCount(*count, decoder.remaining(), minimumRecordBytes))
    return invalid("work record count exceeds the snapshot bound");
  std::vector<JournalWorkUnitRecord> records;
  records.reserve(static_cast<std::size_t>(*count));
  for (std::uint64_t index = 0; index != *count; ++index) {
    auto record = decodeRecord(decoder);
    if (!record)
      return record.takeError();
    if (!records.empty() && !(records.back().key < record->key))
      return invalid("work records are not strictly canonical");
    records.push_back(std::move(*record));
  }
  if (!decoder.empty())
    return invalid("snapshot has trailing bytes");
  return std::make_pair(*stopRequested == 1, std::move(records));
}

auto findRecord(std::vector<JournalWorkUnitRecord> &records,
                const WorkUnitKey &key) {
  return llvm::lower_bound(
      records, key,
      [](const JournalWorkUnitRecord &record, const WorkUnitKey &candidate) {
        return record.key < candidate;
      });
}

} // namespace

llvm::Expected<ExecutionJournal>
ExecutionJournal::open(llvm::StringRef localRunRoot,
                       const DseRunClosure &closure,
                       const ResolvedDseConfigView &view) {
  auto directory = openRunDirectory(localRunRoot);
  if (!directory)
    return directory.takeError();
  auto snapshot = readSnapshot(*directory);
  llvm::Error closeError = closeDescriptor(*directory, "run directory");
  if (!snapshot)
    return llvm::joinErrors(snapshot.takeError(), std::move(closeError));
  if (closeError)
    return std::move(closeError);

  auto state = std::make_shared<State>(localRunRoot.str(), closure.runKey(),
                                       view.digest());
  if (*snapshot) {
    auto decoded = decodeState(**snapshot, closure.runKey(), view.digest());
    if (!decoded)
      return decoded.takeError();
    state->stopRequested = decoded->first;
    state->workUnits = std::move(decoded->second);
    bool recoveredActiveWork = false;
    for (JournalWorkUnitRecord &record : state->workUnits) {
      if (record.activeAttemptStartUnixTimeNanoseconds != 0) {
        auto now = unixNanosecondsNow();
        if (!now)
          return now.takeError();
        if (llvm::Error error = settleStartedInterval(record, *now))
          return error;
        recoveredActiveWork = true;
      }
      if (record.status == JournalWorkUnitStatus::Running) {
        record.status = JournalWorkUnitStatus::Queued;
        recoveredActiveWork = true;
      }
    }
    if (recoveredActiveWork)
      if (llvm::Error error = flushLocked(*state))
        return error;
  } else if (llvm::Error error = flushLocked(*state)) {
    return error;
  }
  return ExecutionJournal(std::move(state));
}

const DseRunKey &ExecutionJournal::runKey() const { return state_->runKey; }

const ComponentViewDigest &
ExecutionJournal::resolvedDseConfigViewDigest() const {
  return state_->viewDigest;
}

llvm::StringRef ExecutionJournal::localRunRoot() const {
  return state_->localRunRoot;
}

llvm::Expected<std::vector<JournalWorkUnitRecord>>
ExecutionJournal::workUnits() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->workUnits;
}

llvm::Expected<std::optional<JournalWorkUnitRecord>>
ExecutionJournal::find(const WorkUnitKey &key) const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto found = findRecord(state_->workUnits, key);
  if (found == state_->workUnits.end() || !(found->key == key))
    return std::optional<JournalWorkUnitRecord>{};
  return std::optional<JournalWorkUnitRecord>(*found);
}

llvm::Error ExecutionJournal::queue(const WorkUnitKey &key) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto found = findRecord(state_->workUnits, key);
  if (found != state_->workUnits.end() && found->key == key) {
    if (terminal(found->status) ||
        found->status == JournalWorkUnitStatus::Prepared)
      return llvm::Error::success();
    if (found->status == JournalWorkUnitStatus::Running) {
      auto now = unixNanosecondsNow();
      if (!now)
        return now.takeError();
      if (llvm::Error error = settleStartedInterval(*found, *now))
        return error;
    }
    found->status = JournalWorkUnitStatus::Queued;
    found->activeAttemptStartUnixTimeNanoseconds = 0;
    found->terminalUnixTimeNanoseconds = 0;
    found->finalizedOutputs.clear();
    found->preparedInvocation.reset();
    state_->transientGenerateResults.erase(key);
  } else {
    state_->workUnits.insert(
        found,
        JournalWorkUnitRecord{
            key, JournalWorkUnitStatus::Queued, {}, 0, 0, {}, std::nullopt});
  }
  return flushLocked(*state_);
}

llvm::Error ExecutionJournal::markRunning(const WorkUnitKey &key) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto found = findRecord(state_->workUnits, key);
  if (found == state_->workUnits.end() || !(found->key == key))
    return invalid("cannot run an unqueued work unit");
  if (found->status != JournalWorkUnitStatus::Queued)
    return invalid("only queued work can enter running state");
  auto now = unixNanosecondsNow();
  if (!now)
    return now.takeError();
  found->status = JournalWorkUnitStatus::Running;
  found->activeAttemptStartUnixTimeNanoseconds = *now;
  return flushLocked(*state_);
}

llvm::Error ExecutionJournal::recordPrepared(
    const WorkUnitKey &key,
    const external_tool::PreparedExternalToolInvocation &prepared,
    std::uint64_t activeWallTimeNanoseconds,
    std::uint64_t observedUnixTimeNanoseconds) {
  if (prepared.bundleRoot.empty())
    return invalid("prepared invocation has an empty bundle root");
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto found = findRecord(state_->workUnits, key);
  if (found == state_->workUnits.end() || !(found->key == key))
    return invalid("cannot prepare an unknown work unit");
  if (found->status != JournalWorkUnitStatus::Running)
    return invalid("only running work can publish a prepared invocation");
  JournalWorkUnitRecord updated = *found;
  if (observedUnixTimeNanoseconds == 0) {
    auto now = unixNanosecondsNow();
    if (!now)
      return now.takeError();
    observedUnixTimeNanoseconds = *now;
    if (activeWallTimeNanoseconds == 0) {
      if (llvm::Error error =
              settleStartedInterval(updated, observedUnixTimeNanoseconds))
        return error;
    } else if (llvm::Error error =
                   addInterval(updated, activeWallTimeNanoseconds,
                               observedUnixTimeNanoseconds)) {
      return error;
    }
  } else if (llvm::Error error = addInterval(updated, activeWallTimeNanoseconds,
                                             observedUnixTimeNanoseconds)) {
    return error;
  }
  updated.status = JournalWorkUnitStatus::Prepared;
  updated.preparedInvocation = prepared;
  if (llvm::Error error = validateRecord(updated))
    return error;
  *found = std::move(updated);
  return flushLocked(*state_);
}

llvm::Error ExecutionJournal::beginPreparedExecution(const WorkUnitKey &key) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto found = findRecord(state_->workUnits, key);
  if (found == state_->workUnits.end() || !(found->key == key))
    return invalid("cannot execute an unknown prepared work unit");
  if (found->status != JournalWorkUnitStatus::Prepared ||
      !found->preparedInvocation)
    return invalid("only prepared work can enter external execution");
  if (found->activeAttemptStartUnixTimeNanoseconds != 0)
    return invalid("prepared work already has an active execution interval");
  auto now = unixNanosecondsNow();
  if (!now)
    return now.takeError();
  found->activeAttemptStartUnixTimeNanoseconds = *now;
  return flushLocked(*state_);
}

llvm::Error ExecutionJournal::recordPreparedExecutionInterval(
    const WorkUnitKey &key, std::uint64_t activeWallTimeNanoseconds,
    std::uint64_t observedUnixTimeNanoseconds) {
  if (observedUnixTimeNanoseconds == 0)
    return invalid("prepared execution interval requires an observation time");
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto found = findRecord(state_->workUnits, key);
  if (found == state_->workUnits.end() || !(found->key == key))
    return invalid("cannot observe an unknown prepared work unit");
  if (found->status != JournalWorkUnitStatus::Prepared ||
      !found->preparedInvocation)
    return invalid("only prepared work can record an execution interval");
  JournalWorkUnitRecord updated = *found;
  if (llvm::Error error = addInterval(updated, activeWallTimeNanoseconds,
                                      observedUnixTimeNanoseconds))
    return error;
  if (llvm::Error error = validateRecord(updated))
    return error;
  *found = std::move(updated);
  return flushLocked(*state_);
}

llvm::Error ExecutionJournal::markTerminal(
    const WorkUnitKey &key, JournalWorkUnitStatus status,
    std::uint64_t activeWallTimeNanoseconds,
    std::uint64_t terminalUnixTimeNanoseconds,
    llvm::ArrayRef<ArtifactRootReference> finalizedOutputs) {
  if (!terminal(status))
    return invalid("terminal transition requires a terminal status");
  if (terminalUnixTimeNanoseconds == 0)
    return invalid("terminal transition requires a completion timestamp");
  if (!llvm::is_sorted(finalizedOutputs, artifactRootReferenceLess) ||
      std::adjacent_find(finalizedOutputs.begin(), finalizedOutputs.end()) !=
          finalizedOutputs.end())
    return invalid("terminal outputs are not canonical and unique");
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto found = findRecord(state_->workUnits, key);
  if (found == state_->workUnits.end() || !(found->key == key))
    return invalid("cannot complete an unknown work unit");
  if (terminal(found->status)) {
    if (activeWallTimeNanoseconds != 0 || found->status != status ||
        found->terminalUnixTimeNanoseconds != terminalUnixTimeNanoseconds ||
        llvm::ArrayRef(found->finalizedOutputs) != finalizedOutputs)
      return invalid("terminal work record cannot be overwritten");
    return llvm::Error::success();
  }
  if (found->status != JournalWorkUnitStatus::Running &&
      found->status != JournalWorkUnitStatus::Prepared)
    return invalid("only running or prepared work can become terminal");
  JournalWorkUnitRecord updated = *found;
  if (activeWallTimeNanoseconds != 0) {
    if (llvm::Error error = addInterval(updated, activeWallTimeNanoseconds,
                                        terminalUnixTimeNanoseconds))
      return error;
  } else if (updated.activeAttemptStartUnixTimeNanoseconds != 0) {
    if (llvm::Error error =
            settleStartedInterval(updated, terminalUnixTimeNanoseconds))
      return error;
  }
  updated.status = status;
  updated.terminalUnixTimeNanoseconds = terminalUnixTimeNanoseconds;
  updated.finalizedOutputs.assign(finalizedOutputs.begin(),
                                  finalizedOutputs.end());
  if (llvm::Error error = validateRecord(updated))
    return error;
  *found = std::move(updated);
  return flushLocked(*state_);
}

llvm::Error ExecutionJournal::rememberTransientGenerateResult(
    const WorkUnitKey &key, const CandidateGeneratorProviderResult &result) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto found = findRecord(state_->workUnits, key);
  if (found == state_->workUnits.end() || !(found->key == key) ||
      !terminal(found->status))
    return invalid("Generate report requires a terminal work record");
  if (generateResultRoots(result) != found->finalizedOutputs)
    return invalid("Generate report roots disagree with finalized roots");
  const bool inserted =
      state_->transientGenerateResults.emplace(key, result).second;
  if (!inserted)
    return invalid("Generate report cache cannot be overwritten");
  return llvm::Error::success();
}

llvm::Expected<std::optional<CandidateGeneratorProviderResult>>
ExecutionJournal::findTransientGenerateResult(const WorkUnitKey &key) const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto found = state_->transientGenerateResults.find(key);
  if (found == state_->transientGenerateResults.end())
    return std::optional<CandidateGeneratorProviderResult>{};
  return std::optional<CandidateGeneratorProviderResult>(found->second);
}

llvm::Error ExecutionJournal::requestGracefulStop() {
  std::lock_guard<std::mutex> lock(state_->mutex);
  state_->stopRequested = true;
  return flushLocked(*state_);
}

llvm::Error ExecutionJournal::beginResume() {
  std::lock_guard<std::mutex> lock(state_->mutex);
  state_->stopRequested = false;
  for (JournalWorkUnitRecord &record : state_->workUnits) {
    if (record.activeAttemptStartUnixTimeNanoseconds != 0) {
      auto now = unixNanosecondsNow();
      if (!now)
        return now.takeError();
      if (llvm::Error error = settleStartedInterval(record, *now))
        return error;
    }
    if (record.status == JournalWorkUnitStatus::Running)
      record.status = JournalWorkUnitStatus::Queued;
  }
  return flushLocked(*state_);
}

bool ExecutionJournal::gracefulStopRequested() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->stopRequested;
}

llvm::Error ExecutionJournal::flush() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return flushLocked(*state_);
}

llvm::Expected<ExecutionJournal>
openExecutionJournal(llvm::StringRef localRunRoot, const DseRunClosure &closure,
                     const ResolvedDseConfigView &view) {
  return ExecutionJournal::open(localRunRoot, closure, view);
}

} // namespace loom::dse
