#include "DSE/FpaCampaign.h"

#include "Common/ArtifactStore.h"
#include "DSE/PortableSpatialCoreRtlCandidateGenerator.h"
#include "DSE/ResolvedConfigView.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "EDA/Adapters/OpenSource/OpenRoadRouted.h"
#include "EDA/Adapters/OpenSource/YosysGateNetlist.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricDialect.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Hardware/Configuration/PackedConfigurationABI.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <optional>
#include <utility>

namespace loom::dse {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fpa_physical_implementation_invalid: " +
                                     message);
}

constexpr llvm::StringLiteral kTechnologyLefLogicalName = "technology";
constexpr llvm::StringLiteral kCellLefLogicalName = "cells";
constexpr llvm::StringLiteral kLibertyLogicalName = "timing";

} // namespace

llvm::Expected<FpaPhysicalImplementationPlan>
buildFpaPhysicalImplementationPlan(FpaPhysicalImplementationRequest request,
                                   const ResolvedConfig &baseConfig,
                                   const ArtifactStore &artifactStore) {
  if (request.systems.empty() && request.rtlImplementations.empty())
    return invalid("physical implementation requires at least one System or "
                   "RTL implementation");
  if (!baseConfig.dse.modelAuthorizations.empty() ||
      !baseConfig.dse.evidenceObligationTemplates.empty() ||
      !baseConfig.dse.qualityGatePolicies.empty() ||
      !baseConfig.dse.planNodes.empty())
    return invalid("base ResolvedConfig already owns a DSE invocation plan");
  if (llvm::Error error = registerPortableSpatialCoreRtlCandidateGenerator())
    return std::move(error);
  if (llvm::Error error =
          eda::open_source::registerYosysGateNetlistCandidateGenerator())
    return std::move(error);
  if (llvm::Error error =
          eda::open_source::registerOpenRoadRoutedCandidateGenerator())
    return std::move(error);

  for (std::vector<ArtifactRootReference> *roots :
       {&request.systems, &request.rtlImplementations}) {
    llvm::sort(*roots, artifactRootReferenceLess);
    roots->erase(std::unique(roots->begin(), roots->end()), roots->end());
  }

  auto platform = platform::finalizeImplementationPlatform(
      platform::ImplementationPlatformDraft{request.asicTarget,
                                            request.technologyCornerKeys},
      artifactStore);
  if (!platform)
    return platform.takeError();
  std::optional<platform::TechnologyCornerRef> corner;
  for (const platform::TechnologyCorner &candidate :
       platform->platform().technologyCorners())
    if (candidate.key == request.selectedTechnologyCornerKey)
      corner = platform::TechnologyCornerRef{platform->reference().artifact,
                                             candidate.id};
  if (!corner)
    return invalid("selected technology corner is not a platform corner");

  ResolvedConfig planConfig = baseConfig;
  FpaPhysicalImplementationPlan plan{
      {}, platform->reference(), *corner, {}, {}, {platform->reference()}};

  if (!request.systems.empty()) {
    auto rtlConfig = resolvePortableSpatialCoreRtlConfig();
    if (!rtlConfig)
      return rtlConfig.takeError();
    mlir::DialectRegistry registry;
    registry.insert<::dataflow::DataflowDialect, ::fabric::FabricDialect,
                    mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::LLVM::LLVMDialect, mlir::memref::MemRefDialect>();
    mlir::MLIRContext context(registry,
                              mlir::MLIRContext::Threading::DISABLED);
    context.loadAllAvailableDialects();
    for (const ArtifactRootReference &systemReference : request.systems) {
      auto system =
          fabric::importEntireFabricRoot(systemReference, artifactStore);
      if (!system)
        return system.takeError();
      if (system->view().rootKind() != fabric::FabricRootKind::System)
        return invalid("physical implementation input is not a System root");
      auto abiDraft =
          hardware::derivePackedConfigurationABIDraft(*system, context);
      if (!abiDraft)
        return abiDraft.takeError();
      auto abi = hardware::finalizeConfigurationABI(std::move(*abiDraft),
                                                    artifactStore);
      if (!abi)
        return abi.takeError();
      const std::uint64_t rtlNode = planConfig.dse.planNodes.size();
      planConfig.dse.planNodes.push_back(GeneratePlanNodeDefinition{
          portableSpatialCoreRtlCandidateGeneratorDescriptor().reference(),
          {ExactPlanArtifacts{{systemReference}},
           ExactPlanArtifacts{{abi->reference()}},
           ExactPlanArtifacts{{platform->reference()}}},
          rtlConfig->canonicalViewBytes().vec(),
          rtlConfig->digest()});
      plan.rtlStages.push_back(
          {systemReference, abi->reference(), PlanOutputRef{rtlNode, 0}});
      plan.semanticInputs.push_back(systemReference);
      plan.semanticInputs.push_back(abi->reference());
    }
  }

  if (!request.rtlImplementations.empty()) {
    auto yosysConfig =
        eda::open_source::createResolvedYosysGateNetlistConfigView(
            request.yosysProviderBuild, *corner, request.liberty);
    if (!yosysConfig)
      return yosysConfig.takeError();
    eda::open_source::OpenRoadPlacedConfig routedConfig{
        request.openRoadProviderBuild,
        *corner,
        request.placement,
        {{eda::open_source::OpenRoadExternalFileKind::TechnologyLef,
          kTechnologyLefLogicalName.str(), request.technologyLef},
         {eda::open_source::OpenRoadExternalFileKind::CellLef,
          kCellLefLogicalName.str(), request.cellLef},
         {eda::open_source::OpenRoadExternalFileKind::Liberty,
          kLibertyLogicalName.str(), request.liberty}}};
    auto routedBytes =
        eda::open_source::encodeOpenRoadPlacedConfig(routedConfig);
    if (!routedBytes)
      return routedBytes.takeError();
    auto routedDigest = computeComponentViewDigest(
        eda::open_source::openRoadPlacedConfigSchemaDescriptorBytes(),
        *routedBytes);
    if (!routedDigest)
      return routedDigest.takeError();
    auto routedBinding = ResolvedCandidateGeneratorBinding::get(
        eda::open_source::openRoadRoutedCandidateGeneratorDescriptor()
            .reference(),
        *routedBytes, *routedDigest);
    if (!routedBinding)
      return routedBinding.takeError();
    for (const ArtifactRootReference &rtl : request.rtlImplementations) {
      const std::uint64_t gateNode = planConfig.dse.planNodes.size();
      planConfig.dse.planNodes.push_back(GeneratePlanNodeDefinition{
          eda::open_source::yosysGateNetlistCandidateGeneratorDescriptor()
              .reference(),
          {ExactPlanArtifacts{{rtl}},
           ExactPlanArtifacts{{platform->reference()}}},
          yosysConfig->canonicalViewBytes().vec(),
          yosysConfig->digest()});
      const std::uint64_t routedNode = planConfig.dse.planNodes.size();
      planConfig.dse.planNodes.push_back(GeneratePlanNodeDefinition{
          eda::open_source::openRoadRoutedCandidateGeneratorDescriptor()
              .reference(),
          {PlanOutputRef{gateNode, 0}},
          *routedBytes,
          *routedDigest});
      plan.physicalStages.push_back(
          {rtl, PlanOutputRef{gateNode, 0}, PlanOutputRef{routedNode, 0}});
      plan.semanticInputs.push_back(rtl);
    }
  }
  auto admitted = projectResolvedDseConfigView(planConfig);
  if (!admitted)
    return admitted.takeError();
  llvm::sort(plan.semanticInputs, artifactRootReferenceLess);
  plan.semanticInputs.erase(
      std::unique(plan.semanticInputs.begin(), plan.semanticInputs.end()),
      plan.semanticInputs.end());
  plan.resolvedConfig = std::move(planConfig);
  return plan;
}

} // namespace loom::dse
