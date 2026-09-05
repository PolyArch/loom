#include "SystemCandidateFixture.h"
#include "ADG/Builtin.h"
#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/CandidateGenerator.h"
#include "DSE/ResourceTimeFrontier.h"
#include "DSE/SystemCompositionCandidateGenerator.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowEventDerivation.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Deployment/Deployment.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/ResourceContract.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"
#include "Mapping/Artifact/SystemMappingHardwareDemand.h"
#include "Mapping/IR/MappingAttrs.h"
#include "Mapping/IR/MappingDialect.h"
#include "Mapping/Tech/TechMappingConfig.h"
#include "Mapping/Tech/TechMappingGenerator.h"
#include "PnR/MappingObjective.h"
#include "PnR/PnrConfig.h"
#include "PnR/SpatialPhysicalTiming.h"
#include "PnR/SpatialPnrGenerator.h"
#include "PnR/System/SystemCandidateState.h"
#include "PnR/System/SystemMappingMaterializer.h"
#include "PnR/System/SystemMappingMigration.h"
#include "PnR/System/SystemPnrGenerator.h"
#include "PnR/System/SystemPnrProblem.h"
#include "PnR/System/SystemPnrSearchDomain.h"
#include "SystemCandidateStateTestSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace loom::pnr::test::fixture {

using loom::pnr::test::byteList;
using loom::pnr::test::bytesAttr;
using loom::pnr::test::unsignedBytes;

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "System CandidateState anchor failed: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(message);
}

void requireVerificationFailureContains(mlir::Operation *operation,
                                        llvm::StringRef expected) {
  std::vector<std::string> diagnostics;
  mlir::ScopedDiagnosticHandler capture(
      operation->getContext(), [&](mlir::Diagnostic &diagnostic) {
        diagnostics.push_back(diagnostic.str());
        return mlir::success();
      });
  require(mlir::failed(mlir::verify(operation)),
          "adverse SystemMapping operation unexpectedly verified");
  require(llvm::any_of(diagnostics,
                       [&](const std::string &diagnostic) {
                         return llvm::StringRef(diagnostic).contains(expected);
                       }),
          "adverse SystemMapping diagnostic changed");
}

std::string identityAttr(const loom::ArtifactIdentity &identity) {
  return "#mapping.artifact_identity<" + byteList(identity.bytes()) + ">";
}

template <typename Ref>
std::string dataflowAttr(llvm::StringRef spelling,
                         const loom::ArtifactIdentity &identity,
                         const Ref &reference) {
  return "#mapping." + spelling.str() + "<" +
         byteList(
             take(dataflow::encodeDataflowReference(identity, reference))) +
         ">";
}

::mapping::ArtifactRootReferenceAttr
rootReferenceAttr(mlir::MLIRContext *context,
                  const loom::ArtifactRootReference &reference) {
  return ::mapping::ArtifactRootReferenceAttr::get(
      context,
      bytesAttr(context, loom::encodeArtifactRootReference(reference)));
}

::mapping::SystemServiceObligationKeyAttr
serviceObligationAttr(mlir::MLIRContext *context,
                      const loom::ArtifactIdentity &owner,
                      const loom::mapping::SystemServiceObligationKey &key) {
  return ::mapping::SystemServiceObligationKeyAttr::get(
      context,
      bytesAttr(context, take(loom::mapping::encodeSystemServiceObligationKey(
                             owner, key))));
}

::mapping::SystemTransferTerminalKeyAttr
transferTerminalAttr(mlir::MLIRContext *context,
                     const loom::ArtifactIdentity &owner,
                     const loom::mapping::SystemTransferTerminalKey &key) {
  return ::mapping::SystemTransferTerminalKeyAttr::get(
      context,
      bytesAttr(context, take(loom::mapping::encodeSystemTransferTerminalKey(
                             owner, key))));
}

mlir::OwningOpRef<mlir::ModuleOp> buildSystemConstraintModule(
    mlir::MLIRContext &context, const loom::ArtifactIdentity &dataflowIdentity,
    const loom::ArtifactIdentity &fabricIdentity,
    llvm::ArrayRef<dataflow::RootThreadLaunchRef> roots) {
  context.loadDialect<::mapping::MappingDialect>();
  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module.getBody());
  std::vector<mlir::Attribute> rootAttributes;
  rootAttributes.reserve(roots.size());
  for (const auto root : roots)
    rootAttributes.push_back(
        constraintDataflowAttr<::mapping::RootThreadLaunchRefAttr>(
            &context, dataflowIdentity, root));
  auto constraint = ::mapping::ConstraintsSystemOp::create(
      builder, builder.getUnknownLoc(),
      ::mapping::ArtifactIdentityAttr::get(
          &context, bytesAttr(&context, dataflowIdentity.bytes())),
      ::mapping::ArtifactIdentityAttr::get(
          &context, bytesAttr(&context, fabricIdentity.bytes())),
      builder.getArrayAttr(rootAttributes), builder.getArrayAttr({}));
  constraint.getBody().emplaceBlock();
  return module;
}

void addSystemRestriction(mlir::OpBuilder &builder,
                          ::mapping::ConstraintsSystemOp root,
                          ::mapping::SystemConstraintProjection projection,
                          mlir::Attribute subject,
                          llvm::ArrayRef<mlir::Attribute> domain) {
  builder.setInsertionPointToEnd(&root.getBody().front());
  mlir::OperationState state(
      builder.getUnknownLoc(),
      ::mapping::ConstraintDomainRestrictionOp::getOperationName());
  state.addAttribute(
      "projection",
      ::mapping::SystemConstraintProjectionKeyAttr::get(
          builder.getContext(), static_cast<std::uint32_t>(projection)));
  state.addAttribute("subject", subject);
  state.addAttribute("admissible_domain", builder.getArrayAttr(domain));
  builder.create(state);
}

void addSystemEquality(mlir::OpBuilder &builder,
                       ::mapping::ConstraintsSystemOp root,
                       ::mapping::SystemConstraintProjection projection,
                       llvm::ArrayRef<mlir::Attribute> subjects) {
  builder.setInsertionPointToEnd(&root.getBody().front());
  mlir::OperationState state(builder.getUnknownLoc(),
                             ::mapping::ConstraintEqualOp::getOperationName());
  state.addAttribute(
      "projection",
      ::mapping::SystemConstraintProjectionKeyAttr::get(
          builder.getContext(), static_cast<std::uint32_t>(projection)));
  state.addAttribute("subjects", builder.getArrayAttr(subjects));
  builder.create(state);
}

mlir::MLIRContext makeContext() {
  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, mlir::arith::ArithDialect,
                  ::mapping::MappingDialect, mlir::DLTIDialect,
                  mlir::func::FuncDialect, mlir::LLVM::LLVMDialect>();
  return mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
}

dataflow::CanonicalDataflowArtifact buildDataflow(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @sync(%start: none, %value: i32) -> i32
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %result:2 = dataflow.sync %start, %value
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%result#1 : i32) streams() memories()
        complete(%result#0 : none)
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)(
      %value: i32) ctrl (%ctrl: none) iv (%iv: index) {
    %first_result, %first_done = dataflow.graph.launch @sync deps(%ctrl)
        values(%value) stream_inputs() memories() stream_outputs()
        : (none, i32) -> (i32, none)
    %second_result, %second_done = dataflow.graph.launch @sync deps(%first_done)
        values(%first_result) stream_inputs() memories() stream_outputs()
        : (none, i32) -> (i32, none)
    dataflow.thread.yield %second_done : none
  }
  llvm.func internal @host() {
    %value = arith.constant 7 : i32
    %extent = arith.constant 8 : index
    %first = dataflow.thread.launch @worker(%value) grid(%extent)
        : (i32) -> !dataflow.thread_token
    %second = dataflow.thread.launch @worker(%value) grid(%extent)
        : (i32) -> !dataflow.thread_token
    llvm.return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

dataflow::CanonicalDataflowArtifact
buildCapacityPressureDataflow(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @first(%start: none) -> ()
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    %done = dataflow.sync %start : (none) -> none
    dataflow.graph.return values() streams() memories()
        complete(%done : none)
  }
  dataflow.graph private @second(%start: none) -> ()
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    %done = dataflow.sync %start : (none) -> none
    dataflow.graph.return values() streams() memories()
        complete(%done : none)
  }
  dataflow.thread private @first_worker domain(#dataflow.thread_domain<dense>)()
      ctrl (%ctrl: none) {
    %first_done = dataflow.graph.launch @first deps(%ctrl)
        values() stream_inputs() memories() stream_outputs()
        : (none) -> none
    dataflow.thread.yield %first_done : none
  }
  dataflow.thread private @second_worker domain(#dataflow.thread_domain<dense>)()
      ctrl (%ctrl: none) {
    %second_done = dataflow.graph.launch @second deps(%ctrl)
        values() stream_inputs() memories() stream_outputs()
        : (none) -> none
    dataflow.thread.yield %second_done : none
  }
  func.func private @host() {
    %first_completion = dataflow.thread.launch @first_worker()
        : () -> !dataflow.thread_token
    %second_completion = dataflow.thread.launch @second_worker()
        : () -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse imported-capacity pressure Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

dataflow::CanonicalDataflowArtifact
buildMemoryDataflow(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @load(
      %ctrl: none, %index: index, %memory: memref<4xi32>) -> i32
      attributes {input_segments = array<i32: 1, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value, %loaded = dataflow.load %memory[%index] %ctrl : memref<4xi32>
    %done = dataflow.store %memory[%index] %value %loaded : memref<4xi32>
    dataflow.graph.return values(%value : i32) streams() memories()
        complete(%done : none)
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)(
      %index: index, %memory: memref<4xi32>) ctrl (%ctrl: none) {
    %value, %done = dataflow.graph.launch @load deps(%ctrl)
        values(%index) stream_inputs() memories(%memory) stream_outputs()
        : (none, index, memref<4xi32>) -> (i32, none)
    dataflow.thread.yield %done : none
  }
  func.func private @host(%index: index, %memory: memref<4xi32>) {
    %completion = dataflow.thread.launch @worker(%index, %memory)
        : (index, memref<4xi32>) -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse memory Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

loom::ArtifactRootReference
generateSpatialMapping(const dataflow::CanonicalDataflowProgramView &dataflow,
                       const loom::fabric::FinalizedFabricRoot &module,
                       const loom::ResolvedConfig &resolved,
                       loom::ArtifactStore &store, mlir::MLIRContext *context,
                       std::optional<dataflow::GraphRef> cover) {
  const auto techConfig =
      take(loom::mapping::projectResolvedTechMappingConfigView(resolved));
  const std::array<dataflow::GraphRef, 1> covers = {
      cover.value_or(dataflow.graphs().front().ref)};
  auto techOutcome = loom::mapping::generateTechMappings(
      {dataflow, covers, module.view(), techConfig, store});
  const auto *techCandidates =
      std::get_if<loom::mapping::GeneratedTechMappings>(&techOutcome);
  if (!techCandidates) {
    if (const auto *invalid =
            std::get_if<loom::mapping::InvalidTechMappingGeneration>(
                &techOutcome))
      fail("TechMapping fixture is invalid: " + invalid->diagnostic);
    if (const auto *internal =
            std::get_if<loom::mapping::InternalTechMappingGeneration>(
                &techOutcome))
      fail("TechMapping fixture failed internally: " + internal->diagnostic);
    if (std::holds_alternative<loom::mapping::ProvenInfeasibleTechMapping>(
            techOutcome))
      fail("TechMapping fixture is proven infeasible");
    fail("TechMapping fixture ended without a proof or candidate");
  }
  require(techCandidates->candidates.size() == 1,
          "TechMapping fixture did not produce one candidate");
  auto tech = take(loom::mapping::importTechMapping(
      techCandidates->candidates.front(), store));
  auto constraints = [&]() {
    if (!context)
      return take(loom::mapping::finalizeEmptySpatialMappingConstraintSet(
          dataflow, tech.view(), module.view(), store));
    require(dataflow.logicalMemoryRoots().size() == 1,
            "boundary-only Mapping fixture requires one logical memory root");
    const std::string text =
        "module {\n  mapping.constraints.spatial dataflow(" +
        identityAttr(dataflow.identity()) + ") tech_mapping(" +
        identityAttr(tech.view().identity()) + ") fabric(" +
        identityAttr(module.view().identity()) +
        ") {\n    mapping.constraint.domain_restriction "
        "projection(memory_bound_services) subject(" +
        dataflowAttr("logical_memory_root_ref", dataflow.identity(),
                     dataflow.logicalMemoryRoots().front().ref) +
        ") admissible_domain([])\n  }\n}\n";
    auto parsed = mlir::parseSourceString<mlir::ModuleOp>(text, context);
    if (!parsed)
      fail("cannot parse boundary-only Spatial MappingConstraintSet");
    auto roots = parsed->getOps<::mapping::ConstraintsSpatialOp>();
    return take(loom::mapping::finalizeSpatialMappingConstraintSet(
        *roots.begin(), dataflow, tech.view(), module.view(), store));
  }();
  const auto spatialConfig =
      take(loom::pnr::projectResolvedSpatialPnrConfigView(resolved));
  const auto physicalTiming =
      take(loom::fabric::projectNormalizedFabricPhysicalTimingProfile(
          module.view()));
  auto spatialOutcome = loom::pnr::generateSpatialMappings(
      {dataflow, tech.view(), module.view(), physicalTiming, spatialConfig,
       constraints.view(), store});
  const auto *spatialCandidates =
      std::get_if<loom::pnr::GeneratedSpatialMappings>(&spatialOutcome);
  if (!spatialCandidates)
    std::visit(
        [&](const auto &outcome) {
          using Outcome = std::decay_t<decltype(outcome)>;
          if constexpr (std::is_same_v<
                            Outcome,
                            loom::pnr::InterruptedSpatialPnrGeneration>)
            fail("SpatialMapping fixture was interrupted at " +
                 loom::pnr::spatialPnrInterruptionStageSpelling(
                     outcome.snapshot.stage));
          else if constexpr (!std::is_same_v<
                                 Outcome, loom::pnr::GeneratedSpatialMappings>)
            fail("SpatialMapping fixture did not produce one candidate: " +
                 outcome.diagnostic);
        },
        spatialOutcome);
  require(!spatialCandidates->candidates.empty(),
          "SpatialMapping fixture did not produce a candidate");
  return spatialCandidates->candidates.front();
}

} // namespace loom::pnr::test::fixture
