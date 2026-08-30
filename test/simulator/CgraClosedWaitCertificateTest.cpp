#include "Simulator/CGRASimulator.h"

#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <limits>

namespace {

using loom::sim::CgraClosedWaitSetDiagnostic;

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "CGRA closed-wait certificate test: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(bool condition, llvm::StringRef message) {
  if (!condition)
    fail(message);
}

using Diagnostic = CgraClosedWaitSetDiagnostic;
using OwnerKey = Diagnostic::WaitOwnerKey;

OwnerKey actor(std::uint64_t ordinal, std::uint64_t occurrence) {
  return OwnerKey{Diagnostic::WaitActorFiringKey{ordinal, occurrence}};
}

OwnerKey storage(std::uint64_t ordinal) {
  return OwnerKey{Diagnostic::WaitStorageQueueKey{
      Diagnostic::WaitStorageDomain::TraversalStorage, ordinal,
      Diagnostic::WaitQueueClass::global()}};
}

OwnerKey taggedStorage(std::uint64_t ordinal, std::uint64_t tag) {
  return OwnerKey{Diagnostic::WaitStorageQueueKey{
      Diagnostic::WaitStorageDomain::TraversalStorage, ordinal,
      Diagnostic::WaitQueueClass::tag(llvm::APInt(4, tag))}};
}

Diagnostic::WaitEdge edge(OwnerKey from, OwnerKey to,
                          Diagnostic::WaitEdgeKind kind) {
  Diagnostic::WaitEdge result;
  result.from = std::move(from);
  result.to = std::move(to);
  result.kind = kind;
  return result;
}

/// One closed fork/join component: two consumers join at one storage whose
/// head is owed to both, and both wait behind the head's queue class.
void closedForkJoinIsAccepted() {
  Diagnostic diagnostic;
  diagnostic.waitCertificate = {
      edge(actor(7, 1), storage(3), Diagnostic::WaitEdgeKind::StorageOrder),
      edge(actor(9, 2), storage(3), Diagnostic::WaitEdgeKind::StorageOrder),
      edge(storage(3), actor(7, 1),
           Diagnostic::WaitEdgeKind::StorageConsumer),
      edge(storage(3), actor(9, 2),
           Diagnostic::WaitEdgeKind::StorageConsumer),
  };
  require(loom::sim::verifyClosedWaitCertificateClosure(diagnostic),
          "a closed fork/join wait component was rejected");
}

/// A tag-local class is a distinct owner: the same storage at another tag is
/// a different queue, and the global class is distinct from every tag class.
void queueClassesDistinguishTagValues() {
  Diagnostic diagnostic;
  diagnostic.waitCertificate = {
      edge(actor(7, 1), taggedStorage(3, 1),
           Diagnostic::WaitEdgeKind::StorageOrder),
      edge(taggedStorage(3, 1), storage(4),
           Diagnostic::WaitEdgeKind::StorageDownstream),
      edge(storage(4), actor(7, 1),
           Diagnostic::WaitEdgeKind::StorageConsumer),
  };
  require(loom::sim::verifyClosedWaitCertificateClosure(diagnostic),
          "a tag-local closed wait was rejected");

  Diagnostic open = diagnostic;
  open.waitCertificate.push_back(edge(actor(7, 1), taggedStorage(3, 2),
                                      Diagnostic::WaitEdgeKind::StorageOrder));
  require(!loom::sim::verifyClosedWaitCertificateClosure(open),
          "a certificate with an open branch was accepted");
}

void openChainsAndProofFailuresAreRejected() {
  Diagnostic dangling;
  dangling.waitCertificate = {
      edge(actor(7, 1), storage(3), Diagnostic::WaitEdgeKind::StorageOrder),
      edge(storage(3), actor(7, 1),
           Diagnostic::WaitEdgeKind::StorageConsumer),
      edge(actor(9, 0), storage(3), Diagnostic::WaitEdgeKind::StorageOrder),
  };
  require(!loom::sim::verifyClosedWaitCertificateClosure(dangling),
          "a certificate with a node lacking an internal in-edge passed");

  Diagnostic absent;
  require(!loom::sim::verifyClosedWaitCertificateClosure(absent),
          "an absent certificate passed");
  absent.waitProofFailure =
      Diagnostic::WaitProofFailure::NoClosedComponent;
  require(!loom::sim::verifyClosedWaitCertificateClosure(absent),
          "a proof failure passed the closure check");
}

} // namespace

int main() {
  closedForkJoinIsAccepted();
  queueClassesDistinguishTagValues();
  openChainsAndProofFailuresAreRejected();
  return EXIT_SUCCESS;
}
