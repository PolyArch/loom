#ifndef LOOM_DSE_EXECUTIONJOURNAL_H
#define LOOM_DSE_EXECUTIONJOURNAL_H

#include "Common/Artifact.h"
#include "DSE/ExternalToolWorkLedger.h"
#include "DSE/InvocationManifest.h"
#include "ExternalTool/InvocationBundle.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

namespace loom::dse {

class ResolvedDseConfigView;

/// Exact owner-local work descriptor used only to derive recoverable work
/// keys. The referenced owner registry remains the authority for the kind.
class WorkUnitDescriptorRef final {
public:
  static llvm::Expected<WorkUnitDescriptorRef>
  get(llvm::StringRef ownerRegistryIdentity, SchemaVersion ownerRegistryVersion,
      std::uint32_t ownerLocalKind);

  llvm::StringRef ownerRegistryIdentity() const {
    return ownerRegistryIdentity_;
  }
  SchemaVersion ownerRegistryVersion() const { return ownerRegistryVersion_; }
  std::uint32_t ownerLocalKind() const { return ownerLocalKind_; }

  friend bool operator==(const WorkUnitDescriptorRef &lhs,
                         const WorkUnitDescriptorRef &rhs) {
    return lhs.ownerRegistryIdentity_ == rhs.ownerRegistryIdentity_ &&
           lhs.ownerRegistryVersion_ == rhs.ownerRegistryVersion_ &&
           lhs.ownerLocalKind_ == rhs.ownerLocalKind_;
  }
  friend bool operator<(const WorkUnitDescriptorRef &lhs,
                        const WorkUnitDescriptorRef &rhs);

private:
  WorkUnitDescriptorRef(std::string ownerRegistryIdentity,
                        SchemaVersion ownerRegistryVersion,
                        std::uint32_t ownerLocalKind)
      : ownerRegistryIdentity_(std::move(ownerRegistryIdentity)),
        ownerRegistryVersion_(ownerRegistryVersion),
        ownerLocalKind_(ownerLocalKind) {}

  std::string ownerRegistryIdentity_;
  SchemaVersion ownerRegistryVersion_;
  std::uint32_t ownerLocalKind_ = 0;
};

/// Stable recovery identity. Attempt, checkpoint, dispatch, and completion
/// order are deliberately absent.
class WorkUnitKey final {
public:
  static llvm::Expected<WorkUnitKey> get(std::uint64_t planNodeOrdinal,
                                         WorkUnitDescriptorRef descriptor,
                                         std::uint64_t stableOrdinal);

  std::uint64_t planNodeOrdinal() const { return planNodeOrdinal_; }
  const WorkUnitDescriptorRef &descriptor() const { return descriptor_; }
  std::uint64_t stableOrdinal() const { return stableOrdinal_; }

  friend bool operator==(const WorkUnitKey &lhs, const WorkUnitKey &rhs) {
    return lhs.planNodeOrdinal_ == rhs.planNodeOrdinal_ &&
           lhs.descriptor_ == rhs.descriptor_ &&
           lhs.stableOrdinal_ == rhs.stableOrdinal_;
  }
  friend bool operator<(const WorkUnitKey &lhs, const WorkUnitKey &rhs);

private:
  WorkUnitKey(std::uint64_t planNodeOrdinal, WorkUnitDescriptorRef descriptor,
              std::uint64_t stableOrdinal)
      : planNodeOrdinal_(planNodeOrdinal), descriptor_(std::move(descriptor)),
        stableOrdinal_(stableOrdinal) {}

  std::uint64_t planNodeOrdinal_ = 0;
  WorkUnitDescriptorRef descriptor_;
  std::uint64_t stableOrdinal_ = 0;
};

enum class JournalWorkUnitStatus : std::uint32_t {
  Queued = 0,
  Running = 1,
  Prepared = 2,
  Completed = 3,
  Failed = 4,
  TimedOut = 5,
  Unsupported = 6,
};

enum class ExecutionJournalPersistenceErrorReason : std::uint8_t {
  PublishedDirectorySyncPending = 0,
};

/// A snapshot rename completed, so the visible journal already owns the new
/// state, but the directory durability barrier failed. A later journal
/// operation first retries that barrier; callers must not assume rollback.
/// Invocation begin and manifest commit are specifically idempotent across
/// this outcome.
class ExecutionJournalPersistenceError final
    : public llvm::ErrorInfo<ExecutionJournalPersistenceError> {
public:
  static char ID;

  ExecutionJournalPersistenceError(
      ExecutionJournalPersistenceErrorReason reason, std::error_code error)
      : reason_(reason), error_(error) {}

  ExecutionJournalPersistenceErrorReason reason() const { return reason_; }
  const std::error_code &underlyingError() const { return error_; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  ExecutionJournalPersistenceErrorReason reason_;
  std::error_code error_;
};

struct JournalActiveWallInterval final {
  std::uint64_t beginUnixTimeNanoseconds = 0;
  std::uint64_t endUnixTimeNanoseconds = 0;

  friend bool operator==(const JournalActiveWallInterval &lhs,
                         const JournalActiveWallInterval &rhs) {
    return lhs.beginUnixTimeNanoseconds == rhs.beginUnixTimeNanoseconds &&
           lhs.endUnixTimeNanoseconds == rhs.endUnixTimeNanoseconds;
  }
};

struct OwnerFinalizedWorkRecordRef final {
  std::string schemaIdentity;
  SchemaVersion schemaVersion;
  BlobDigest payloadDigest;

  friend bool operator==(const OwnerFinalizedWorkRecordRef &lhs,
                         const OwnerFinalizedWorkRecordRef &rhs) {
    return lhs.schemaIdentity == rhs.schemaIdentity &&
           lhs.schemaVersion == rhs.schemaVersion &&
           lhs.payloadDigest == rhs.payloadDigest;
  }
  friend bool operator!=(const OwnerFinalizedWorkRecordRef &lhs,
                         const OwnerFinalizedWorkRecordRef &rhs) {
    return !(lhs == rhs);
  }
};

struct JournalWorkUnitRecord final {
  WorkUnitKey key;
  JournalWorkUnitStatus status = JournalWorkUnitStatus::Queued;
  std::vector<JournalActiveWallInterval> activeWallIntervals;
  std::uint64_t activeAttemptStartUnixTimeNanoseconds = 0;
  std::uint64_t terminalUnixTimeNanoseconds = 0;
  std::vector<ArtifactRootReference> finalizedOutputs;
  std::optional<external_tool::PreparedExternalToolInvocation>
      preparedInvocation;
  std::optional<OwnerFinalizedWorkRecordRef> finalizedWorkRecord;
  ExternalToolWorkLedger externalToolWork;

  std::uint64_t activeWallTimeNanoseconds() const;
};

/// Mutable, nonsemantic recovery state stored as one atomic canonical
/// snapshot in a caller-owned run directory.
class ExecutionJournal final {
public:
  static llvm::Expected<ExecutionJournal>
  open(llvm::StringRef localRunRoot, const DseRunClosure &closure,
       const ResolvedDseConfigView &view);

  const DseRunKey &runKey() const;
  const ComponentViewDigest &resolvedDseConfigViewDigest() const;
  llvm::StringRef localRunRoot() const;

  /// The occurrence opened by the most recent beginResume transaction and
  /// its durable predecessor, if any. The ordinal remains a transient lease
  /// until commitInvocationManifest atomically publishes its manifest receipt.
  llvm::Expected<std::pair<InvocationOccurrenceRef,
                           std::optional<InvocationOccurrenceRef>>>
  currentInvocationOccurrence() const;
  llvm::Expected<std::optional<InvocationManifestReceipt>>
  lastCommittedInvocationManifest() const;

  llvm::Expected<std::vector<JournalWorkUnitRecord>> workUnits() const;
  llvm::Expected<InvocationExternalToolWorkLedger>
  externalToolWorkLedger() const;
  llvm::Expected<std::optional<JournalWorkUnitRecord>>
  find(const WorkUnitKey &key) const;

  llvm::Error queue(const WorkUnitKey &key);
  llvm::Error markRunning(const WorkUnitKey &key);
  llvm::Error
  recordPrepared(const WorkUnitKey &key,
                 const external_tool::PreparedExternalToolInvocation &prepared,
                 std::uint64_t activeWallTimeNanoseconds = 0,
                 std::uint64_t observedUnixTimeNanoseconds = 0);
  llvm::Error beginPreparedExecution(const WorkUnitKey &key);
  llvm::Error recordPreparedExecutionInterval(
      const WorkUnitKey &key, std::uint64_t activeWallTimeNanoseconds,
      std::uint64_t observedUnixTimeNanoseconds,
      std::optional<external_tool::ExternalToolInvocationExecutionObservation>
          executionObservation = std::nullopt);
  llvm::Error
  markTerminal(const WorkUnitKey &key, JournalWorkUnitStatus status,
               std::uint64_t activeWallTimeNanoseconds,
               std::uint64_t terminalUnixTimeNanoseconds,
               llvm::ArrayRef<ArtifactRootReference> finalizedOutputs = {},
               std::optional<OwnerFinalizedWorkRecordRef> finalizedWorkRecord =
                   std::nullopt);

  llvm::Error requestGracefulStop();
  llvm::Error beginResume();
  llvm::Error releaseInvocationOccurrence();
  llvm::Error
  commitInvocationManifest(const InvocationOccurrenceRef &occurrence,
                           const BlobDigest &manifestDigest);
  bool gracefulStopRequested() const;
  llvm::Error flush() const;

private:
  struct State;
  explicit ExecutionJournal(std::shared_ptr<State> state)
      : state_(std::move(state)) {}

  llvm::Error validateProcessOwner() const;

  std::shared_ptr<State> state_;
};

llvm::Expected<ExecutionJournal>
openExecutionJournal(llvm::StringRef localRunRoot, const DseRunClosure &closure,
                     const ResolvedDseConfigView &view);

} // namespace loom::dse

#endif // LOOM_DSE_EXECUTIONJOURNAL_H
