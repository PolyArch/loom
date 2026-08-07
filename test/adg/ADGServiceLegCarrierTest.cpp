#include "ADGBuilderTestSupport.h"

#include "ADG/Builtin.h"

#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/MemoryServiceContract.h"
#include "FabricArtifactBytecodeInternal.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace loom::adg::test {
namespace {

template <typename Configure>
llvm::Expected<FinalizedFabricDesign>
buildReadServiceSystemInStore(llvm::StringRef test, ArtifactStore &store,
                              llvm::ArrayRef<std::uint32_t> inputWidths,
                              llvm::ArrayRef<std::uint32_t> outputWidths,
                              Configure configure) {
  DesignBuilder moduleDesign(store);
  auto spatial =
      take(test, moduleDesign.createSpatialCore("service-leg-spatial", {}, {}));
  if (llvm::Error error = spatial.close({}))
    fail(test, llvm::toString(std::move(error)));
  FinalizedFabricDesign moduleClosure =
      take(test, std::move(moduleDesign).finalize());

  DesignBuilder systemDesign(store);
  auto system = take(test, systemDesign.createSystem("service-leg-system"));
  auto imported =
      take(test, system.importSpatialCore(moduleClosure.roots().front()));
  auto architecture = instructionArchitecture(test);
  auto microarchitecture = inOrderMicroarchitecture(test);
  auto host = take(test, system.addHostCore(architecture, microarchitecture));
  auto core =
      take(test, system.addAccCore(architecture, microarchitecture, imported));

  std::vector<PortType> inputs;
  std::vector<PortType> outputs;
  for (std::uint32_t width : inputWidths)
    inputs.push_back(take(test, PortType::bits(width)));
  for (std::uint32_t width : outputWidths)
    outputs.push_back(take(test, PortType::bits(width)));
  auto transport = take(
      test, system.addTransportResource({std::move(inputs), std::move(outputs),
                                         singleUseResourceContract(test)}));

  auto clock = take(test, system.createHardwareDomain());
  auto rate = take(test, system.createServiceRate(
                             clock, 1, 1, 4,
                             fabric::ServiceProgress(
                                 std::in_place_type<::fabric::FairEventual>)));
  mlir::MLIRContext contractContext(mlir::MLIRContext::Threading::DISABLED);
  auto memoryService = take(test, system.addMemoryService(systemMemoryContract(
                                      test, contractContext)));
  auto serviceEndpoint =
      take(test,
           system.addServiceEndpoint(
               memoryService, systemMemoryCapabilities(test, std::move(rate))));
  auto memoryEndpoint = take(test, serviceEndpoint.memory());
  if (llvm::Error error = configure(system, memoryEndpoint, transport))
    fail(test, llvm::toString(std::move(error)));

  if (llvm::Error error = clock.close(
          {host.domainMember(), core.instructionCoreDomainMember(),
           core.spatialCoreDomainMember(), transport.domainMember(),
           memoryService.domainMember(), serviceEndpoint.domainMember()},
          take(test, fabric::ClockDomainContractRecord::create(1'000, 0))))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = system.close())
    fail(test, llvm::toString(std::move(error)));
  return std::move(systemDesign).finalize();
}

template <typename Configure>
llvm::Expected<FinalizedFabricDesign> buildReadServiceSystem(
    llvm::StringRef test, llvm::ArrayRef<std::uint32_t> inputWidths,
    llvm::ArrayRef<std::uint32_t> outputWidths, Configure configure) {
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  return buildReadServiceSystemInStore(test, store, inputWidths, outputWidths,
                                       std::move(configure));
}

std::vector<std::uint8_t> unsignedBytes(mlir::DenseI8ArrayAttr attribute) {
  std::vector<std::uint8_t> result;
  result.reserve(attribute.size());
  for (std::int8_t byte : attribute.asArrayRef())
    result.push_back(static_cast<std::uint8_t>(byte));
  return result;
}

mlir::DenseI8ArrayAttr denseBytes(mlir::MLIRContext *context,
                                  llvm::ArrayRef<std::uint8_t> bytes) {
  std::vector<std::int8_t> signedBytes;
  signedBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    signedBytes.push_back(static_cast<std::int8_t>(byte));
  return mlir::DenseI8ArrayAttr::get(context, signedBytes);
}

template <typename Mutate>
void expectStoredSystemMutationRejected(
    llvm::StringRef test, const fabric::FinalizedFabricRoot &valid,
    ArtifactStore &store, Mutate mutate, llvm::StringRef diagnostic) {
  fabric::DecodedFabricArtifact decoded = take(
      test,
      fabric::decodeFabricArtifactEnvelope(valid.canonicalBytes().bytes()));
  fabric::detail::ParsedFabricBytecodeModule parsed = take(
      test,
      fabric::detail::parseFabricBytecodeModule(decoded.canonicalMlirBytecode));
  auto systems = parsed.module->getOps<::fabric::SystemOp>();
  require(test, llvm::hasSingleElement(systems),
          "canonical fixture does not contain one System root");
  mutate(*systems.begin());
  decoded.canonicalMlirBytecode = take(
      test, fabric::detail::writeCanonicalFabricBytecode(parsed.module.get()));
  CanonicalSemanticBytes canonical =
      take(test, fabric::encodeFabricArtifactEnvelope(
                     fabric::FabricRootKind::System, decoded.dependencies,
                     decoded.canonicalMlirBytecode));
  ArtifactIdentity identity =
      take(test, store.put(fabric::fabricArtifactSchema, canonical));
  expectError(test,
              fabric::importEntireFabricRoot(
                  {fabric::fabricArtifactSchema.identity.str(),
                   fabric::fabricArtifactSchema.version, identity},
                  store),
              diagnostic);
}

llvm::SmallVector<::fabric::SystemServiceLegCarrierAttachmentOp>
attachments(::fabric::SystemOp system) {
  return llvm::to_vector(
      system.getOps<::fabric::SystemServiceLegCarrierAttachmentOp>());
}

void missingServiceLegAttachmentIsRejected() {
  const llvm::StringRef test = __func__;
  auto finalized = buildReadServiceSystem(
      test, {64}, {32},
      [&](SystemBuilder &system, const SystemMemoryEndpoint &endpoint,
          const SystemTransportResource &transport) {
        return system.attachServiceLegCarriers(
            endpoint, dataflow::semantics::ServiceKind::MemoryRead, 0,
            {take(test, transport.input(0))});
      });
  expectError(test, std::move(finalized),
              "does not attach every admitted memory service leg");
}

llvm::Error
attachCompleteRead(llvm::StringRef test, SystemBuilder &system,
                   const SystemMemoryEndpoint &endpoint,
                   const SystemTransportResource &transport,
                   llvm::ArrayRef<std::size_t> requestCarriers = {0},
                   llvm::ArrayRef<std::size_t> responseCarriers = {0}) {
  std::vector<SystemTransportEndpoint> requests;
  std::vector<SystemTransportEndpoint> responses;
  for (std::size_t ordinal : requestCarriers)
    requests.push_back(take(test, transport.input(ordinal)));
  for (std::size_t ordinal : responseCarriers)
    responses.push_back(take(test, transport.output(ordinal)));
  if (llvm::Error error = system.attachServiceLegCarriers(
          endpoint, dataflow::semantics::ServiceKind::MemoryRead, 0, requests))
    return error;
  return system.attachServiceLegCarriers(
      endpoint, dataflow::semantics::ServiceKind::MemoryRead, 1, responses);
}

void outOfRangeServiceLegIsRejected() {
  const llvm::StringRef test = __func__;
  auto finalized = buildReadServiceSystem(
      test, {64}, {32},
      [&](SystemBuilder &system, const SystemMemoryEndpoint &endpoint,
          const SystemTransportResource &transport) -> llvm::Error {
        if (llvm::Error error =
                attachCompleteRead(test, system, endpoint, transport))
          return error;
        return system.attachServiceLegCarriers(
            endpoint, dataflow::semantics::ServiceKind::MemoryRead, 2,
            {take(test, transport.input(0))});
      });
  expectError(test, std::move(finalized),
              "service-leg attachment ordinal is out of range");
}

void unsupportedServiceKindIsRejected() {
  const llvm::StringRef test = __func__;
  auto finalized = buildReadServiceSystem(
      test, {64}, {32},
      [&](SystemBuilder &system, const SystemMemoryEndpoint &endpoint,
          const SystemTransportResource &transport) -> llvm::Error {
        if (llvm::Error error =
                attachCompleteRead(test, system, endpoint, transport))
          return error;
        return system.attachServiceLegCarriers(
            endpoint, dataflow::semantics::ServiceKind::MemoryWrite, 0,
            {take(test, transport.input(0))});
      });
  expectError(test, std::move(finalized),
              "service-leg attachment selects an unsupported kind");
}

void wrongCarrierDirectionIsRejected() {
  const llvm::StringRef test = __func__;
  auto finalized = buildReadServiceSystem(
      test, {64}, {64},
      [&](SystemBuilder &system, const SystemMemoryEndpoint &endpoint,
          const SystemTransportResource &transport) -> llvm::Error {
        if (llvm::Error error = system.attachServiceLegCarriers(
                endpoint, dataflow::semantics::ServiceKind::MemoryRead, 0,
                {take(test, transport.output(0))}))
          return error;
        return system.attachServiceLegCarriers(
            endpoint, dataflow::semantics::ServiceKind::MemoryRead, 1,
            {take(test, transport.output(0))});
      });
  expectError(test, std::move(finalized),
              "service-leg carrier has the wrong direction");
}

void narrowCarrierIsRejected() {
  const llvm::StringRef test = __func__;
  auto finalized = buildReadServiceSystem(
      test, {32}, {32},
      [&](SystemBuilder &system, const SystemMemoryEndpoint &endpoint,
          const SystemTransportResource &transport) {
        return attachCompleteRead(test, system, endpoint, transport);
      });
  expectError(test, std::move(finalized),
              "service-leg carrier payload is too narrow");
}

void multipleCarrierAlternativesArePreserved() {
  const llvm::StringRef test = __func__;
  FinalizedFabricDesign finalized = take(
      test, buildReadServiceSystem(
                test, {64, 128}, {32, 64},
                [&](SystemBuilder &system, const SystemMemoryEndpoint &endpoint,
                    const SystemTransportResource &transport) {
                  const std::size_t alternatives[] = {0, 1};
                  return attachCompleteRead(test, system, endpoint, transport,
                                            alternatives, alternatives);
                }));
  auto system =
      take(test, fabric::requireSystemRoot(finalized.roots().front().view()));
  require(test, system.serviceLegCarrierAttachments().size() == 2,
          "complete read service did not retain exactly two leg rows");
  for (const fabric::ServiceLegCarrierAttachmentRecord &attachment :
       system.serviceLegCarrierAttachments())
    require(test, attachment.carriers().size() == 2,
            "service-leg carrier alternatives were not preserved");
}

void authoringOrderAndDuplicateRowsConverge() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  const std::size_t alternatives[] = {0, 1};
  FinalizedFabricDesign direct = take(
      test, buildReadServiceSystemInStore(
                test, store, {64, 128}, {32, 64},
                [&](SystemBuilder &system, const SystemMemoryEndpoint &endpoint,
                    const SystemTransportResource &transport) {
                  return attachCompleteRead(test, system, endpoint, transport,
                                            alternatives, alternatives);
                }));
  FinalizedFabricDesign split =
      take(test,
           buildReadServiceSystemInStore(
               test, store, {64, 128}, {32, 64},
               [&](SystemBuilder &system, const SystemMemoryEndpoint &endpoint,
                   const SystemTransportResource &transport) -> llvm::Error {
                 if (llvm::Error error = system.attachServiceLegCarriers(
                         endpoint, dataflow::semantics::ServiceKind::MemoryRead,
                         1, {take(test, transport.output(1))}))
                   return error;
                 if (llvm::Error error = system.attachServiceLegCarriers(
                         endpoint, dataflow::semantics::ServiceKind::MemoryRead,
                         0, {take(test, transport.input(1))}))
                   return error;
                 if (llvm::Error error = system.attachServiceLegCarriers(
                         endpoint, dataflow::semantics::ServiceKind::MemoryRead,
                         0, {take(test, transport.input(0))}))
                   return error;
                 return system.attachServiceLegCarriers(
                     endpoint, dataflow::semantics::ServiceKind::MemoryRead, 1,
                     {take(test, transport.output(0))});
               }));
  require(test,
          direct.roots().front().reference() ==
              split.roots().front().reference(),
          "attachment authoring order or duplicate rows changed identity");
  auto view =
      take(test, fabric::requireSystemRoot(split.roots().front().view()));
  require(test,
          view.serviceLegCarrierAttachments().size() == 2 &&
              view.serviceLegCarrierAttachments()[0].legOrdinal() == 0 &&
              view.serviceLegCarrierAttachments()[1].legOrdinal() == 1,
          "attachment rows are not in canonical key order");
}

FinalizedFabricDesign buildCanonicalReadFixture(llvm::StringRef test,
                                                ArtifactStore &store) {
  return take(
      test, buildReadServiceSystemInStore(
                test, store, {64}, {32},
                [&](SystemBuilder &system, const SystemMemoryEndpoint &endpoint,
                    const SystemTransportResource &transport) {
                  return attachCompleteRead(test, system, endpoint, transport);
                }));
}

void persistedAttachmentOrderIsStrict() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  FinalizedFabricDesign fixture = buildCanonicalReadFixture(test, store);
  expectStoredSystemMutationRejected(
      test, fixture.roots().front(), store,
      [&](::fabric::SystemOp system) {
        auto rows = attachments(system);
        require(test, rows.size() == 2,
                "strict-order fixture does not contain two attachment rows");
        rows.front()->moveAfter(rows.back());
      },
      "canonical System child operation order is not canonical");
}

void persistedDuplicateAttachmentKeyIsRejected() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  FinalizedFabricDesign fixture = buildCanonicalReadFixture(test, store);
  expectStoredSystemMutationRejected(
      test, fixture.roots().front(), store,
      [&](::fabric::SystemOp system) {
        auto rows = attachments(system);
        require(test, !rows.empty(),
                "duplicate-key fixture has no attachment row");
        mlir::OpBuilder builder(system.getContext());
        builder.setInsertionPointAfter(rows.front());
        builder.insert(rows.front()->clone());
      },
      "service-leg attachment relation repeats one key");
}

void persistedIncompleteAttachmentRelationIsRejected() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  FinalizedFabricDesign fixture = buildCanonicalReadFixture(test, store);
  expectStoredSystemMutationRejected(
      test, fixture.roots().front(), store,
      [&](::fabric::SystemOp system) {
        auto rows = attachments(system);
        require(test, rows.size() == 2,
                "incomplete-relation fixture has the wrong row count");
        rows.back().erase();
      },
      "does not attach every admitted memory service leg exactly once");
}

FinalizedFabricDesign buildCanonicalBuiltinFixture(llvm::StringRef test,
                                                   ArtifactStore &store) {
  return take(test, buildBuiltinTarget(store, BuiltinTargetPreset::Small));
}

void persistedIncompletePairMemberRelationIsRejected() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  FinalizedFabricDesign fixture = buildCanonicalBuiltinFixture(test, store);
  expectStoredSystemMutationRejected(
      test, fixture.roots().front(), store,
      [&](::fabric::SystemOp system) {
        for (auto row : attachments(system)) {
          auto record =
              take(test, fabric::decodeServiceLegCarrierAttachmentRecord(
                             unsignedBytes(row.getRecordAttr())));
          if (record.endpoint().owner.kind() ==
              fabric::FabricMemoryEndpointOwnerKind::SpatialCoreOccurrence) {
            row.erase();
            return;
          }
        }
        fail(test, "builtin fixture has no occurrence-side carrier row");
      },
      "does not attach every admitted memory service leg exactly once");
}

void persistedOccurrenceCarrierDirectionIsChecked() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  FinalizedFabricDesign fixture = buildCanonicalBuiltinFixture(test, store);
  expectStoredSystemMutationRejected(
      test, fixture.roots().front(), store,
      [&](::fabric::SystemOp system) {
        const auto rows = attachments(system);
        std::optional<fabric::FabricMemoryEndpointRef> endpoint;
        for (auto row : rows) {
          auto record =
              take(test, fabric::decodeServiceLegCarrierAttachmentRecord(
                             unsignedBytes(row.getRecordAttr())));
          if (record.endpoint().owner.kind() ==
              fabric::FabricMemoryEndpointOwnerKind::SpatialCoreOccurrence) {
            endpoint = record.endpoint();
            break;
          }
        }
        require(test, endpoint.has_value(),
                "builtin fixture has no occurrence-side endpoint");

        ::fabric::SystemServiceLegCarrierAttachmentOp requestRow;
        std::vector<fabric::FabricTransportEndpointRef> responseCarriers;
        for (auto row : rows) {
          auto record =
              take(test, fabric::decodeServiceLegCarrierAttachmentRecord(
                             unsignedBytes(row.getRecordAttr())));
          if (record.endpoint() != *endpoint ||
              record.kind() != dataflow::semantics::ServiceKind::MemoryRead)
            continue;
          if (record.legOrdinal() == 0)
            requestRow = row;
          if (record.legOrdinal() == 1)
            responseCarriers.assign(record.carriers().begin(),
                                    record.carriers().end());
        }
        require(test, requestRow && !responseCarriers.empty(),
                "builtin fixture has no complete occurrence read pair");
        auto changed = take(
            test, fabric::ServiceLegCarrierAttachmentRecord::create(
                      *endpoint, dataflow::semantics::ServiceKind::MemoryRead,
                      0, std::move(responseCarriers)));
        auto bytes = take(
            test, fabric::encodeServiceLegCarrierAttachmentRecord(changed));
        requestRow.setRecordAttr(denseBytes(system.getContext(), bytes));
      },
      "service-leg carrier has the wrong direction");
}

void persistedUnknownAttachmentEndpointIsRejected() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  FinalizedFabricDesign fixture = buildCanonicalReadFixture(test, store);
  expectStoredSystemMutationRejected(
      test, fixture.roots().front(), store,
      [&](::fabric::SystemOp system) {
        auto rows = attachments(system);
        require(test, !rows.empty(),
                "unknown-endpoint fixture has no attachment row");
        auto record =
            take(test, fabric::decodeServiceLegCarrierAttachmentRecord(
                           unsignedBytes(rows.front().getRecordAttr())));
        std::vector<fabric::FabricTransportEndpointRef> carriers(
            record.carriers().begin(), record.carriers().end());
        auto changed = take(
            test, fabric::ServiceLegCarrierAttachmentRecord::create(
                      {fabric::FabricMemoryEndpointOwnerRef::of(
                           fabric::SystemServiceEndpointRef(999)),
                       0},
                      record.kind(), record.legOrdinal(), std::move(carriers)));
        auto bytes = take(
            test, fabric::encodeServiceLegCarrierAttachmentRecord(changed));
        rows.front().setRecordAttr(denseBytes(system.getContext(), bytes));
      },
      "System relation references an unknown entity");
}

} // namespace

void runServiceLegCarrierTests() {
  missingServiceLegAttachmentIsRejected();
  outOfRangeServiceLegIsRejected();
  unsupportedServiceKindIsRejected();
  wrongCarrierDirectionIsRejected();
  narrowCarrierIsRejected();
  multipleCarrierAlternativesArePreserved();
  authoringOrderAndDuplicateRowsConverge();
  persistedAttachmentOrderIsStrict();
  persistedDuplicateAttachmentKeyIsRejected();
  persistedIncompleteAttachmentRelationIsRejected();
  persistedIncompletePairMemberRelationIsRejected();
  persistedOccurrenceCarrierDirectionIsChecked();
  persistedUnknownAttachmentEndpointIsRejected();
}

} // namespace loom::adg::test
