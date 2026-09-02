#ifndef LOOM_DSE_SITESCHEDULER_H
#define LOOM_DSE_SITESCHEDULER_H

#include "Common/BlobDigest.h"
#include "Common/ExecutionControl.h"
#include "DSE/ExecutionJournal.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace loom::dse {

enum class SiteResourceKind : std::uint32_t {
  Cpu = 0,
  Memory = 1,
  Scratch = 2,
  ExternalTool = 3,
  License = 4,
};

/// Mechanically derived identity for one executable provider or its license
/// pool. Human-authored resource labels are deliberately not admitted.
class SiteResourceKey final {
public:
  static SiteResourceKey externalToolBinding(const BlobDigest &binding);
  static SiteResourceKey licenseBinding(const BlobDigest &binding);

  SiteResourceKind kind() const { return kind_; }
  const BlobDigest &digest() const { return digest_; }

  friend bool operator==(const SiteResourceKey &lhs,
                         const SiteResourceKey &rhs) {
    return lhs.kind_ == rhs.kind_ && lhs.digest_ == rhs.digest_;
  }
  friend bool operator<(const SiteResourceKey &lhs, const SiteResourceKey &rhs);

private:
  SiteResourceKey(SiteResourceKind kind, BlobDigest digest)
      : kind_(kind), digest_(std::move(digest)) {}

  SiteResourceKind kind_;
  BlobDigest digest_;
};

struct CountedSiteResource final {
  SiteResourceKey key;
  std::uint64_t units = 0;

  friend bool operator==(const CountedSiteResource &lhs,
                         const CountedSiteResource &rhs) {
    return lhs.key == rhs.key && lhs.units == rhs.units;
  }
};

/// One exact operational claim. Counted resource arrays are canonical and
/// unique; zero-unit entries are rejected rather than treated as absence.
class SiteResourceClaim final {
public:
  static llvm::Expected<SiteResourceClaim>
  get(std::uint64_t cpuCores, std::uint64_t memoryBytes,
      std::uint64_t scratchBytes,
      llvm::ArrayRef<CountedSiteResource> externalTools = {},
      llvm::ArrayRef<CountedSiteResource> licenses = {});

  std::uint64_t cpuCores() const { return cpuCores_; }
  std::uint64_t memoryBytes() const { return memoryBytes_; }
  std::uint64_t scratchBytes() const { return scratchBytes_; }
  llvm::ArrayRef<CountedSiteResource> externalTools() const {
    return externalTools_;
  }
  llvm::ArrayRef<CountedSiteResource> licenses() const { return licenses_; }

private:
  SiteResourceClaim(std::uint64_t cpuCores, std::uint64_t memoryBytes,
                    std::uint64_t scratchBytes,
                    std::vector<CountedSiteResource> externalTools,
                    std::vector<CountedSiteResource> licenses)
      : cpuCores_(cpuCores), memoryBytes_(memoryBytes),
        scratchBytes_(scratchBytes), externalTools_(std::move(externalTools)),
        licenses_(std::move(licenses)) {}

  std::uint64_t cpuCores_ = 0;
  std::uint64_t memoryBytes_ = 0;
  std::uint64_t scratchBytes_ = 0;
  std::vector<CountedSiteResource> externalTools_;
  std::vector<CountedSiteResource> licenses_;
};

class SiteCapacity final {
public:
  /// `undeclaredExternalToolUnits` is the capacity granted to every external
  /// tool binding that has no explicit entry in `externalTools`; zero keeps
  /// such bindings inadmissible. Licenses are never granted implicitly.
  static llvm::Expected<SiteCapacity>
  get(std::uint64_t cpuCores, std::uint64_t memoryBytes,
      std::uint64_t scratchBytes,
      llvm::ArrayRef<CountedSiteResource> externalTools = {},
      llvm::ArrayRef<CountedSiteResource> licenses = {},
      std::uint64_t undeclaredExternalToolUnits = 0);

  std::uint64_t cpuCores() const { return cpuCores_; }
  std::uint64_t memoryBytes() const { return memoryBytes_; }
  std::uint64_t scratchBytes() const { return scratchBytes_; }
  llvm::ArrayRef<CountedSiteResource> externalTools() const {
    return externalTools_;
  }
  llvm::ArrayRef<CountedSiteResource> licenses() const { return licenses_; }
  std::uint64_t undeclaredExternalToolUnits() const {
    return undeclaredExternalToolUnits_;
  }

private:
  SiteCapacity(std::uint64_t cpuCores, std::uint64_t memoryBytes,
               std::uint64_t scratchBytes,
               std::vector<CountedSiteResource> externalTools,
               std::vector<CountedSiteResource> licenses,
               std::uint64_t undeclaredExternalToolUnits)
      : cpuCores_(cpuCores), memoryBytes_(memoryBytes),
        scratchBytes_(scratchBytes), externalTools_(std::move(externalTools)),
        licenses_(std::move(licenses)),
        undeclaredExternalToolUnits_(undeclaredExternalToolUnits) {}

  std::uint64_t cpuCores_ = 0;
  std::uint64_t memoryBytes_ = 0;
  std::uint64_t scratchBytes_ = 0;
  std::vector<CountedSiteResource> externalTools_;
  std::vector<CountedSiteResource> licenses_;
  std::uint64_t undeclaredExternalToolUnits_ = 0;
};

struct ScheduledWorkUnit final {
  WorkUnitKey key;
  SiteResourceClaim claim;
};

struct SiteSchedulerSnapshot final {
  SiteCapacity capacity;
  SiteResourceClaim allocated;
  std::vector<ScheduledWorkUnit> running;
  std::vector<ScheduledWorkUnit> queued;
};

class SiteSchedulerState;

/// RAII ownership of one admitted operational claim. Moving transfers the
/// claim; destruction releases it and wakes blocked independent work.
class SiteResourceLease final {
public:
  SiteResourceLease(SiteResourceLease &&other) noexcept;
  SiteResourceLease &operator=(SiteResourceLease &&other) noexcept;
  SiteResourceLease(const SiteResourceLease &) = delete;
  SiteResourceLease &operator=(const SiteResourceLease &) = delete;
  ~SiteResourceLease();

  const WorkUnitKey &workUnit() const { return key_; }
  const SiteResourceClaim &claim() const { return claim_; }
  explicit operator bool() const { return state_ != nullptr; }
  void release();

private:
  SiteResourceLease(std::shared_ptr<SiteSchedulerState> state, WorkUnitKey key,
                    SiteResourceClaim claim)
      : state_(std::move(state)), key_(std::move(key)),
        claim_(std::move(claim)) {}

  std::shared_ptr<SiteSchedulerState> state_;
  WorkUnitKey key_;
  SiteResourceClaim claim_;

  friend class SiteScheduler;
};

/// Caller-owned resource coordinator. It does not execute work, infer tool
/// liveness, or own semantic scheduling policy.
class SiteScheduler final {
public:
  static llvm::Expected<SiteScheduler> create(SiteCapacity capacity);

  llvm::Expected<std::optional<SiteResourceLease>>
  tryAcquire(const WorkUnitKey &key, const SiteResourceClaim &claim);
  llvm::Expected<SiteResourceLease> acquire(const WorkUnitKey &key,
                                            const SiteResourceClaim &claim);
  /// Preserves queue fairness while observing invocation-local execution
  /// control. An empty lease means control stopped before admission.
  llvm::Expected<std::optional<SiteResourceLease>>
  acquire(const WorkUnitKey &key, const SiteResourceClaim &claim,
          ExecutionControlView executionControl);
  /// Transitions one lease to prepare-discovered tool and license resources
  /// while retaining its CPU, memory, and scratch reservation. Prior counted
  /// resources are released before the target is queued, preventing
  /// hold-and-wait cycles. The borrowed lease must not be used concurrently;
  /// execution-control callbacks may re-enter the scheduler.
  llvm::Expected<bool>
  bindCountedResources(SiteResourceLease &lease,
                       const SiteResourceClaim &target,
                       ExecutionControlView executionControl = {});
  llvm::Expected<SiteSchedulerSnapshot> snapshot() const;

private:
  explicit SiteScheduler(std::shared_ptr<SiteSchedulerState> state)
      : state_(std::move(state)) {}

  std::shared_ptr<SiteSchedulerState> state_;
};

} // namespace loom::dse

#endif // LOOM_DSE_SITESCHEDULER_H
