#ifndef LOOM_TEST_SYSTEM_EXECUTIONMATRIXLIFECYCLE_H
#define LOOM_TEST_SYSTEM_EXECUTIONMATRIXLIFECYCLE_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <memory>

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::system_test {

enum class ExecutionMatrixLifecycleOperation : std::uint8_t {
  Setup,
  HostLifecycle,
  Gem5Binding,
  RequestConstruction,
  OrdinaryPrepare,
  OrdinaryExternalExecution,
  OrdinaryImportAndEvidenceAssembly,
  OrdinaryEvidencePublication,
  OrdinaryExecutionImport,
  DiagnosticPrepare,
  DiagnosticExternalExecution,
  DiagnosticImportAndEvidenceAssembly,
  DiagnosticEvidencePublication,
  DiagnosticExecutionImport,
  OracleVerification,
  Cleanup,
};

class ExecutionMatrixLifecycleRecorder final {
public:
  class Impl;

  ExecutionMatrixLifecycleRecorder();
  ~ExecutionMatrixLifecycleRecorder();

  ExecutionMatrixLifecycleRecorder(const ExecutionMatrixLifecycleRecorder &) =
      delete;
  ExecutionMatrixLifecycleRecorder &
  operator=(const ExecutionMatrixLifecycleRecorder &) = delete;

  void emit(llvm::StringRef cell) const;

private:
  std::unique_ptr<Impl> impl_;
  friend class ExecutionMatrixLifecycleTimer;
};

class ExecutionMatrixLifecycleTimer final {
public:
  class Impl;

  ExecutionMatrixLifecycleTimer(ExecutionMatrixLifecycleRecorder &recorder,
                                ExecutionMatrixLifecycleOperation operation);
  ~ExecutionMatrixLifecycleTimer();

  ExecutionMatrixLifecycleTimer(const ExecutionMatrixLifecycleTimer &) =
      delete;
  ExecutionMatrixLifecycleTimer &
  operator=(const ExecutionMatrixLifecycleTimer &) = delete;

private:
  std::unique_ptr<Impl> impl_;
};

struct ExecutionMatrixImportSummary final {
  std::uint64_t gem5FactsRequests = 0;
  std::uint64_t gem5FactsHits = 0;
  std::uint64_t gem5FactsMisses = 0;
  std::uint64_t gem5FactsUniqueConstructions = 0;
};

/// Owns every removable immutable import/projection session for one exact
/// execution-matrix store domain. Independent replay cells construct isolated
/// instances and never share session attachments.
class ExecutionMatrixImportSessions final {
public:
  class Impl;

  ExecutionMatrixImportSessions(const ArtifactStore &artifacts,
                                const BlobStore &blobs);
  ~ExecutionMatrixImportSessions();

  ExecutionMatrixImportSessions(const ExecutionMatrixImportSessions &) =
      delete;
  ExecutionMatrixImportSessions &
  operator=(const ExecutionMatrixImportSessions &) = delete;

  ExecutionMatrixImportSummary summary() const;
  void emitStatistics(llvm::StringRef cell) const;

private:
  std::unique_ptr<Impl> impl_;
};

} // namespace loom::system_test

#endif // LOOM_TEST_SYSTEM_EXECUTIONMATRIXLIFECYCLE_H
