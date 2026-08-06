#include "ADGBuilderTestSupport.h"

#include "ADG/Export.h"
#include "ADG/FuLibrary.h"

#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricSystemRootView.h"

#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <vector>

namespace loom::adg::test {

void fuBackedgesAreExplicitAndResolved() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  const PortType bits32 = take(test, PortType::bits(32));

  {
    DesignBuilder incomplete(store);
    auto spatial = take(test, incomplete.createSpatialCore(
                                  "unresolved-feedback", {bits32}, {bits32}));
    auto pe = take(test, spatial.addPe({take(test, spatial.input(0))},
                                       PeSpec::spatial({bits32}, {bits32})));
    auto fu = take(
        test, pe.addFu({take(test, pe.input(0))}, FuSpec{{bits32}, {bits32}}));
    auto backedge = take(test, fu.createBackedge(bits32));
    expectError(test, fu.close({backedge.value()}), "unresolved backedge");
  }

  DesignBuilder design(store);
  auto spatial = take(
      test, design.createSpatialCore("resolved-feedback", {bits32}, {bits32}));
  auto pe = take(test, spatial.addPe({take(test, spatial.input(0))},
                                     PeSpec::spatial({bits32}, {bits32})));
  auto fu = take(
      test, pe.addFu({take(test, pe.input(0))}, FuSpec{{bits32}, {bits32}}));
  auto backedge = take(test, fu.createBackedge(bits32));
  auto sum = take(
      test,
      fu.addOperation({take(test, fu.input(0)), backedge.value()},
                      integerCapability(
                          ::fabric::ImplementationFamilyId::ScalarIntegerAddSub,
                          ::dataflow::OperationSchemaId::ArithAddI, bits32)));
  FuValue sumValue = take(test, sum.output(0));
  if (llvm::Error error = fu.resolveBackedge(std::move(backedge), sumValue))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error =
          fu.addCapabilityTemplate(FuCapabilityTemplateSpec{{sum}, {}}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = fu.close({sumValue}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = pe.close())
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = spatial.close({take(test, pe.output(0))}))
    fail(test, llvm::toString(std::move(error)));
  auto finalized = take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 1,
          "resolved FU backedge did not finalize");
}

void spatialBackedgesEnableCyclicTopology() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  const PortType bits32 = take(test, PortType::bits(32));

  {
    DesignBuilder incomplete(store);
    auto spatial = take(test, incomplete.createSpatialCore("unresolved-cycle",
                                                           {bits32}, {bits32}));
    auto backedge = take(test, spatial.createBackedge(bits32));
    expectError(test, spatial.close({backedge.value()}), "unresolved backedge");
  }

  DesignBuilder design(store);
  auto spatial =
      take(test, design.createSpatialCore("cyclic-switch", {bits32}, {bits32}));
  auto backedge = take(test, spatial.createBackedge(bits32));
  auto routed = take(
      test,
      spatial.addSwitch({take(test, spatial.input(0)), backedge.value()},
                        SwitchSpec::spatial({bits32, bits32}, {bits32, bits32},
                                            {{0, 1}, {0, 1}})));
  SpatialValue buffered =
      take(test, spatial.addFifo(routed[0], FifoSpec{bits32, 2, true}))
          .value();
  if (llvm::Error error =
          spatial.resolveBackedge(std::move(backedge), buffered))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = spatial.close({routed[1]}))
    fail(test, llvm::toString(std::move(error)));

  auto finalized = take(test, std::move(design).finalize());
  require(test,
          finalized.roots().size() == 1 &&
              !finalized.roots().front().view().admittedTraversals().empty(),
          "resolved SpatialCore cycle did not finalize as explicit topology");
}

void routedFuLibraryBuildsHeterogeneousBoundaries() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);
  const PortType bits128 = take(test, PortType::bits(128));

  auto adapter = take(test, design.createSpatialCore(
                                "vector-adapter", {bits128, bits128, bits128},
                                {bits128, bits128, bits128}));
  auto adapterPe =
      take(test, adapter.addPe({take(test, adapter.input(0)),
                                take(test, adapter.input(1)),
                                take(test, adapter.input(2))},
                               PeSpec::spatial({bits128, bits128, bits128},
                                               {bits128, bits128, bits128})));
  std::vector<loom::adg::PeValue> adapterInputs;
  for (std::size_t ordinal = 0; ordinal != 3; ++ordinal)
    adapterInputs.push_back(take(test, adapterPe.input(ordinal)));
  if (llvm::Error error =
          loom::adg::addVectorAdapterFu(adapterPe, adapterInputs))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = adapterPe.close())
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = adapter.close({take(test, adapterPe.output(0)),
                                         take(test, adapterPe.output(1)),
                                         take(test, adapterPe.output(2))}))
    fail(test, llvm::toString(std::move(error)));

  auto token = take(test, design.createSpatialCore(
                              "token-control",
                              {bits128, bits128, bits128, bits128, bits128},
                              {bits128, bits128, bits128, bits128}));
  auto tokenPe = take(
      test,
      token.addPe({take(test, token.input(0)), take(test, token.input(1)),
                   take(test, token.input(2)), take(test, token.input(3)),
                   take(test, token.input(4))},
                  PeSpec::spatial({bits128, bits128, bits128, bits128, bits128},
                                  {bits128, bits128, bits128, bits128})));
  std::vector<loom::adg::PeValue> tokenInputs;
  for (std::size_t ordinal = 0; ordinal != 5; ++ordinal)
    tokenInputs.push_back(take(test, tokenPe.input(ordinal)));
  if (llvm::Error error = loom::adg::addTokenControlFu(
          tokenPe, tokenInputs, loom::adg::TokenControlFuParameters{128, 64}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = tokenPe.close())
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = token.close(
          {take(test, tokenPe.output(0)), take(test, tokenPe.output(1)),
           take(test, tokenPe.output(2)), take(test, tokenPe.output(3))}))
    fail(test, llvm::toString(std::move(error)));

  auto finalized = take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 2,
          "routed FU helpers did not finalize both boundary shapes");
  for (const auto &root : finalized.roots())
    require(test,
            root.view()
                    .fuCapabilityTemplates(uniqueFuTemplate(test, root.view()))
                    .size() == 4,
            "routed FU helper did not derive four complete templates");
  std::string text;
  llvm::raw_string_ostream stream(text);
  for (const auto &root : finalized.roots())
    if (llvm::Error error = loom::fabric::writeFabricMlir(root, stream))
      fail(test, llvm::toString(std::move(error)));
  stream.flush();
  require(test,
          llvm::StringRef(text).contains("FixedVectorParallelize") &&
              llvm::StringRef(text).contains("FixedVectorSerialize") &&
              llvm::StringRef(text).contains("TokenSync") &&
              llvm::StringRef(text).contains("TokenDemux"),
          "routed FU helpers lost heterogeneous operation capabilities");
}

void heterogeneousSystemFinalizes() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  const PortType bits32 = take(test, PortType::bits(32));
  const PortType bits64 = take(test, PortType::bits(64));

  DesignBuilder moduleDesign(store);
  auto firstSpatial =
      take(test, moduleDesign.createSpatialCore("system-spatial-buffered",
                                                {bits32}, {bits32}));
  SpatialValue firstBuffered =
      take(test, firstSpatial.addFifo(take(test, firstSpatial.input(0)),
                                      FifoSpec{bits32, 2, true}))
          .value();
  if (llvm::Error error = firstSpatial.close({firstBuffered}))
    fail(test, llvm::toString(std::move(error)));
  auto secondSpatial =
      take(test, moduleDesign.createSpatialCore("system-spatial-deep-buffered",
                                                {bits32}, {bits32}));
  SpatialValue secondBuffered =
      take(test, secondSpatial.addFifo(take(test, secondSpatial.input(0)),
                                       FifoSpec{bits32, 3, false}))
          .value();
  if (llvm::Error error = secondSpatial.close({secondBuffered}))
    fail(test, llvm::toString(std::move(error)));
  loom::adg::FinalizedFabricDesign moduleClosure =
      take(test, std::move(moduleDesign).finalize());
  require(test, moduleClosure.roots().size() == 2,
          "heterogeneous System fixture did not publish both SpatialCores");

  DesignBuilder systemDesign(store);
  auto system = take(test, systemDesign.createSystem("heterogeneous-system"));
  auto firstImported =
      take(test, system.importSpatialCore(moduleClosure.roots()[0]));
  auto secondImported =
      take(test, system.importSpatialCore(moduleClosure.roots()[1]));
  auto architecture = instructionArchitecture(test);
  auto inOrder = inOrderMicroarchitecture(test);
  auto host = take(test, system.addHostCore(architecture, inOrder));
  auto firstCore =
      take(test, system.addAccCore(architecture, inOrder, firstImported));
  auto secondCore = take(
      test, system.addAccCore(architecture, outOfOrderMicroarchitecture(test),
                              secondImported));

  auto transport =
      take(test, system.addTransportResource(
                     {{bits64}, {bits64}, singleUseResourceContract(test)}));
  auto pattern = take(test, system.addTransferPattern(transport, 0, {0}, 0));
  if (llvm::Error error =
          system.connect(take(test, firstCore.spatialTransportOutput(0)),
                         take(test, transport.input(0))))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error =
          system.connect(take(test, transport.output(0)),
                         take(test, secondCore.spatialTransportInput(0))))
    fail(test, llvm::toString(std::move(error)));

  auto clock = take(test, system.createHardwareDomain());
  auto serviceRate =
      take(test, system.createServiceRate(
                     clock, 1, 1, 4,
                     loom::fabric::ServiceProgress(
                         std::in_place_type<::fabric::FairEventual>)));
  mlir::MLIRContext contractContext(mlir::MLIRContext::Threading::DISABLED);
  auto memoryService = take(test, system.addMemoryService(systemMemoryContract(
                                      test, contractContext)));
  auto memoryEndpoint =
      take(test, system.addServiceEndpoint(
                     memoryService,
                     systemMemoryCapabilities(test, std::move(serviceRate))));
  auto memoryEndpointRef = take(test, memoryEndpoint.memory());
  if (llvm::Error error = system.attachServiceLegCarriers(
          memoryEndpointRef, ::dataflow::semantics::ServiceKind::MemoryRead, 0,
          {take(test, transport.input(0))}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = system.attachServiceLegCarriers(
          memoryEndpointRef, ::dataflow::semantics::ServiceKind::MemoryRead, 0,
          {take(test, transport.input(0))}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = system.attachServiceLegCarriers(
          memoryEndpointRef, ::dataflow::semantics::ServiceKind::MemoryRead, 1,
          {take(test, transport.output(0))}))
    fail(test, llvm::toString(std::move(error)));
  auto clockContract =
      take(test, loom::fabric::ClockDomainContractRecord::create(1'000, 0));
  if (llvm::Error error = clock.close(
          {host.domainMember(), firstCore.instructionCoreDomainMember(),
           firstCore.spatialCoreDomainMember(),
           secondCore.instructionCoreDomainMember(),
           secondCore.spatialCoreDomainMember(), transport.domainMember(),
           pattern.domainMember(), memoryService.domainMember(),
           memoryEndpoint.domainMember()},
          std::move(clockContract)))
    fail(test, llvm::toString(std::move(error)));

  if (llvm::Error error = system.close())
    fail(test, llvm::toString(std::move(error)));
  loom::adg::FinalizedFabricDesign finalized =
      take(test, std::move(systemDesign).finalize());
  require(test, finalized.roots().size() == 1,
          "System design did not publish one root");
  const auto &root = finalized.roots().front();
  require(test, root.view().rootKind() == loom::fabric::FabricRootKind::System,
          "System Builder published the wrong root kind");
  require(test, root.directDependencies().size() == 2,
          "heterogeneous System did not retain both SpatialCore types");
  for (const auto &module : moduleClosure.roots()) {
    bool found = false;
    for (const auto &dependency : root.directDependencies())
      found |= dependency.root == module.reference();
    require(test, found,
            "heterogeneous System changed an exact SpatialCore dependency");
  }
  require(test,
          entityCount(root.view(),
                      loom::fabric::FabricEntityKind::AccCoreOccurrence) == 2,
          "heterogeneous System lost an AccCore occurrence");
  require(test,
          entityCount(root.view(),
                      loom::fabric::FabricEntityKind::SystemMemoryService) == 1,
          "heterogeneous System lost its memory service");
  require(test,
          entityCount(root.view(),
                      loom::fabric::FabricEntityKind::SystemServiceEndpoint) ==
              1,
          "heterogeneous System lost its service endpoint");
  auto systemView = take(test, loom::fabric::requireSystemRoot(root.view()));
  require(test, systemView.spatialAttachments().size() == 4,
          "System Builder did not attach every SpatialCore boundary");
  require(
      test,
      systemView.transportResources().size() == 1 &&
          systemView.transferPatterns(systemView.transportResources().front())
                  .size() == 1,
      "System Builder lost its explicit transport resource or pattern");
  require(test, systemView.serviceLegCarrierAttachments().size() == 2,
          "System Builder lost the complete memory service carrier relation");
  require(test, root.view().pointConnections().size() == 2,
          "System Builder lost its arbitrary directed transport path");

  std::string mlirText;
  llvm::raw_string_ostream stream(mlirText);
  if (llvm::Error error = loom::fabric::writeFabricMlir(root, stream))
    fail(test, llvm::toString(std::move(error)));
  stream.flush();
  require(test, llvm::StringRef(mlirText).contains("fabric.system"),
          "finalized System did not export Fabric MLIR");

  llvm::SmallString<128> outputBase(directory.path());
  llvm::sys::path::append(outputBase, "heterogeneous-system");
  if (llvm::Error error =
          loom::adg::exportFabricDesign(root, store, outputBase))
    fail(test, llvm::toString(std::move(error)));

  llvm::SmallString<128> mlirPath(outputBase);
  mlirPath.append(".mlir");
  llvm::SmallString<128> htmlPath(outputBase);
  htmlPath.append(".html");
  auto exportedMlir = llvm::MemoryBuffer::getFile(mlirPath);
  if (!exportedMlir)
    fail(test, exportedMlir.getError().message());
  auto exportedHtml = llvm::MemoryBuffer::getFile(htmlPath);
  if (!exportedHtml)
    fail(test, exportedHtml.getError().message());
  require(test, exportedMlir.get()->getBuffer().contains("fabric.system"),
          "paired export did not write the canonical Fabric MLIR projection");
  const llvm::StringRef html = exportedHtml.get()->getBuffer();
  const std::size_t firstSpatialView =
      html.find("data-view-kind=\"spatial-core\"");
  const std::size_t secondSpatialView =
      html.find("data-view-kind=\"spatial-core\"", firstSpatialView + 1);
  require(
      test,
      html.contains("data-layout-engine=\"loom-layered-v1\"") &&
          html.contains("data-view-kind=\"system-overview\"") &&
          html.contains("data-view-kind=\"system-noc\"") &&
          html.contains("data-view-kind=\"system\"") &&
          html.contains("data-view-kind=\"spatial-core\"") &&
          html.contains("data-entity-kind=\"fabric.acc_core_occurrence\"") &&
          html.contains("data-entity-kind=\"fabric.fifo_occurrence\"") &&
          html.contains("data-x=\"") && html.contains("data-y=\""),
      "Fabric HTML did not contain the precomputed two-level topology");
  require(test,
          firstSpatialView != llvm::StringRef::npos &&
              secondSpatialView != llvm::StringRef::npos,
          "Fabric HTML did not preserve both heterogeneous SpatialCore views");
  const std::size_t overviewBegin =
      html.find("data-view-kind=\"system-overview\"");
  const std::size_t overviewEnd = html.find("</svg>", overviewBegin);
  require(test,
          overviewBegin != llvm::StringRef::npos &&
              overviewEnd != llvm::StringRef::npos,
          "Fabric HTML has no bounded System overview");
  const llvm::StringRef overview =
      html.slice(overviewBegin, overviewEnd + llvm::StringRef("</svg>").size());
  require(
      test,
      overview.contains("data-entity-kind=\"visual.noc_summary\"") &&
          overview.contains("data-entity-kind=\"fabric.acc_core_occurrence\""),
      "System overview lost its AccCore or NoC architecture summary");
  require(test,
          !overview.contains(
              "data-entity-kind=\"fabric.system_transport_resource\""),
          "System overview exposed individual NoC transport resources");
  const std::size_t nocBegin = html.find(" data-view-kind=\"system-noc\"");
  const std::size_t nocEnd = html.find("</svg>", nocBegin);
  require(test,
          nocBegin != llvm::StringRef::npos && nocEnd != llvm::StringRef::npos,
          "Fabric HTML has no bounded NoC topology view");
  const llvm::StringRef noc =
      html.slice(nocBegin, nocEnd + llvm::StringRef("</svg>").size());
  require(test,
          noc.contains("data-entity-kind=\"fabric.acc_core_occurrence\"") &&
              noc.contains("data-entity-kind=\"fabric.system_memory_service\""),
          "NoC topology lost an architecture participant");
  require(test,
          !noc.contains("data-entity-kind=\"fabric.module_dependency\"") &&
              !noc.contains(
                  "data-entity-kind=\"fabric.system_service_endpoint\"") &&
              !noc.contains("data-entity-kind=\"fabric.hardware_domain\""),
          "NoC topology exposed a detail-only node");
  require(test,
          !html.contains("forceSimulation") && !html.contains("dagre.layout") &&
              !html.contains("elk.layout"),
          "Fabric HTML contains a browser-side graph layout engine");
}

void runTopologyTests() {
  fuBackedgesAreExplicitAndResolved();
  spatialBackedgesEnableCyclicTopology();
  routedFuLibraryBuildsHeterogeneousBoundaries();
  heterogeneousSystemFinalizes();
}

} // namespace loom::adg::test
