#include "Simulator/CgraClosedWaitCertificate.h"

#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <array>
#include <limits>
#include <utility>

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

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

loom::ArtifactIdentity identity(std::uint8_t value) {
  loom::ArtifactIdentity::Storage bytes{};
  bytes.fill(value);
  return take(loom::ArtifactIdentity::fromBytes(bytes));
}

loom::ArtifactRootReference root(llvm::StringRef schema,
                                 loom::SchemaVersion version,
                                 std::uint8_t value) {
  return {schema.str(), version, identity(value)};
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

void durableCertificateRoundTripsAsOneMinimalOwner() {
  Diagnostic diagnostic;
  diagnostic.ownerReferences = loom::sim::CgraExecutionOwnerReferences{
      root("dataflow.canonical", {1, 0}, 1),
      root("loom.fabric", {7, 1}, 2),
      root("loom.mapping", {4, 0}, 3),
      root("loom.mapping", {4, 0}, 4)};
  Diagnostic::Transfer transfer;
  transfer.bindingOrdinal = 11;
  transfer.occurrenceOrdinal = 2;
  transfer.physicalTagOrdinal = 0;
  transfer.physicalTagValue = llvm::APInt(4, 3);
  transfer.producer = dataflow::ActorTokenResultRef{
      dataflow::ActorRef{diagnostic.ownerReferences->dataflow.artifact,
                         dataflow::ActorId(5)},
      0};
  transfer.physicalTagOwner = loom::sim::CgraRoutePhysicalTagOwner{
      *transfer.producer, 3};
  transfer.blockingStorageOrdinal = 7;
  diagnostic.transfers.push_back(std::move(transfer));

  Diagnostic::WaitEdge order =
      edge(actor(5, 2), taggedStorage(7, 3),
           Diagnostic::WaitEdgeKind::StorageOrder);
  order.bindingOrdinal = 11;
  order.occurrenceOrdinal = 2;
  order.storageOrdinal = 7;
  order.headBindingOrdinal = 11;
  order.headOccurrenceOrdinal = 2;
  Diagnostic::WaitEdge consume =
      edge(taggedStorage(7, 3), actor(5, 2),
           Diagnostic::WaitEdgeKind::StorageConsumer);
  consume.bindingOrdinal = 11;
  consume.occurrenceOrdinal = 2;
  consume.storageOrdinal = 7;
  consume.headBindingOrdinal = 11;
  consume.headOccurrenceOrdinal = 2;
  diagnostic.waitCertificate = {std::move(consume), std::move(order)};

  auto certificate =
      take(loom::sim::buildCgraClosedWaitCertificate(diagnostic));
  const auto bytes = take(loom::sim::encodeCgraClosedWaitCertificate(
      certificate));
  auto adopted = take(loom::sim::decodeCgraClosedWaitCertificate(bytes));
  const auto reencoded =
      take(loom::sim::encodeCgraClosedWaitCertificate(adopted));
  require(bytes == reencoded,
          "durable certificate changed across strict adoption");
  require(adopted.transfers.size() == 1 &&
              adopted.transfers.front().bindingOrdinal == 11 &&
              adopted.transfers.front().physicalTagValue == llvm::APInt(4, 3) &&
              adopted.transfers.front().physicalTagOwner &&
              std::get<loom::sim::CgraRoutePhysicalTagOwner>(
                  *adopted.transfers.front().physicalTagOwner)
                      .segmentOrdinal == 3,
          "durable transfer lost an encoded semantic field");
  require(take(loom::sim::digestCgraClosedWaitCertificate(certificate)) ==
              take(loom::sim::digestCgraClosedWaitCertificate(adopted)),
          "certificate digest changed across strict adoption");

  adopted.transfers.emplace_back(
      12, 0, llvm::APInt(1, 0), false,
      dataflow::ActorTokenResultRef{
          dataflow::ActorRef{diagnostic.ownerReferences->dataflow.artifact,
                             dataflow::ActorId(6)},
          0});
  llvm::Error extra = loom::sim::verifyCgraClosedWaitCertificate(adopted);
  require(static_cast<bool>(extra),
          "certificate admitted a transfer no edge references");
  llvm::consumeError(std::move(extra));
}

} // namespace

int main() {
  closedForkJoinIsAccepted();
  queueClassesDistinguishTagValues();
  openChainsAndProofFailuresAreRejected();
  durableCertificateRoundTripsAsOneMinimalOwner();
  return EXIT_SUCCESS;
}
