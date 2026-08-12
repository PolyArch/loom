#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/JointDesignExploration.h"
#include "DSE/ResolvedConfigView.h"
#include "DSE/RootCompleteTechMappingCandidateGenerator.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Frontend/IR/LoomOps.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingExecutionProjection.h"
#include "Simulator/SimulationArtifacts.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "joint design exploration anchor failed: " << message
               << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-joint-design", path_))
      fail("cannot create test directory: " + error.message());
  }
  ~TemporaryDirectory() { llvm::sys::fs::remove_directories(path_); }
  llvm::StringRef path() const { return path_; }

private:
  llvm::SmallString<128> path_;
};

mlir::MLIRContext makeContext() {
  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, mlir::arith::ArithDialect,
                  mlir::DLTIDialect, mlir::func::FuncDialect,
                  loom::LoomDialect>();
  return mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
}

dataflow::CanonicalDataflowArtifact buildDataflow(mlir::MLIRContext &context,
                                                  std::int32_t constant) {
  const std::string source = R"mlir(
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
      %value: i32) ctrl (%ctrl: none) {
    %result, %done = dataflow.graph.launch @sync deps(%ctrl)
        values(%value) stream_inputs() memories() stream_outputs()
        : (none, i32) -> (i32, none)
    dataflow.thread.yield %done : none
  }
  func.func private @host() {
    %value = arith.constant )mlir" +
                             std::to_string(constant) + R"mlir( : i32
    %thread = dataflow.thread.launch @worker(%value)
        : (i32) -> !dataflow.thread_token
    return
  }
}

)mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module)
    fail("cannot parse Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

loom::ArtifactRootReference publishApplicationWorkload(
    const dataflow::CanonicalDataflowArtifact &artifact,
    const loom::ArtifactStore &store) {
  auto view = take(artifact.view());
  if (view.rootThreadLaunches().size() != 1 ||
      view.staticGraphLaunches().size() != 1)
    fail("application fixture does not have one rooted graph launch");
  dataflow::RootedGraphLaunchRef launch{
      view.rootThreadLaunches().front().ref,
      view.staticGraphLaunches().front().ref};
  loom::sim::SpatialSimulationWorkload draft{launch};
  auto shapes = take(loom::sim::projectSpatialSimulationBoundaryShapes(
      view, launch));
  draft.valueInputPlan.assign(shapes.valueInputs.size(),
                              loom::sim::RuntimeValueInput{});
  auto workload = take(loom::sim::finalizeSimulationWorkload(draft, view));
  return take(loom::sim::publishSimulationWorkload(workload, store));
}

std::string key(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

bool everyCoreIsUsed(const loom::ArtifactRootReference &systemReference,
                     llvm::ArrayRef<loom::ArtifactRootReference> mappings,
                     const loom::ArtifactStore &store) {
  auto systemArtifact =
      take(loom::fabric::importEntireFabricRoot(systemReference, store));
  auto system = take(loom::fabric::requireSystemRoot(systemArtifact.view()));
  std::set<std::string> used;
  for (const loom::ArtifactRootReference &reference : mappings) {
    auto mapping = take(loom::mapping::importSystemMapping(reference, store));
    loom::ArtifactRootReference dataflowReference{
        dataflow::canonicalDataflowSchema.identity.str(),
        dataflow::canonicalDataflowSchema.version,
        mapping.view().dataflowIdentity()};
    auto dataflowArtifact =
        take(dataflow::importCanonicalDataflow(dataflowReference, store));
    auto dataflowView = take(dataflowArtifact.view());
    auto projection = take(loom::mapping::projectSystemExecutionContexts(
        dataflowView, mapping.view().executionBindings()));
    for (const auto &domain : projection.instructionDomains)
      used.insert(key(loom::fabric::canonicalFabricBytes(
          domain.context.accCore)));
  }
  return llvm::all_of(system.artifact().accCoreOccurrences(),
                      [&](loom::fabric::AccCoreOccurrenceRef core) {
                        return used.count(key(
                                   loom::fabric::canonicalFabricBytes(core))) !=
                               0;
                      });
}

void exerciseJointExploration() {
  TemporaryDirectory temporary;
  llvm::SmallString<128> blobPath(temporary.path());
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  loom::ArtifactStore store(temporary.path());
  loom::BlobStore blobs(blobPath);
  mlir::MLIRContext context = makeContext();

  auto first = buildDataflow(context, 7);
  auto second = buildDataflow(context, 11);
  take(dataflow::publishCanonicalDataflow(first, store));
  take(dataflow::publishCanonicalDataflow(second, store));
  const loom::ArtifactRootReference firstWorkload =
      publishApplicationWorkload(first, store);
  const loom::ArtifactRootReference secondWorkload =
      publishApplicationWorkload(second, store);
  auto small = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));
  auto alternate = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Default));
  if (small.roots().size() != 1 || alternate.roots().size() != 1)
    fail("builtin fixture did not publish one complete System");
  const loom::ArtifactRootReference system = small.roots().front().reference();
  const loom::ArtifactRootReference alternateSystem =
      alternate.roots().front().reference();

  const loom::dse::JointDesignPolicy policy =
      take(loom::dse::JointDesignPolicy::get(2, 1, 1, 32));
  loom::ResolvedConfig config = loom::defaultResolvedConfig();
  config.dse.techMapping.candidatePublicationLimit = 2;
  auto plan = take(loom::dse::buildJointDesignExplorationPlan(
      {{{firstWorkload}, {secondWorkload}}, {system}}, policy, config, store));
  if (plan.frontier.eligiblePairCount != 2 || !plan.frontier.truncated ||
      plan.frontier.pairs.size() != 1 || plan.pairOutputs.size() != 1)
    fail("bounded pair frontier did not declare deterministic truncation");
  if (plan.pairOutputs.front().techMappings.empty() ||
      plan.pairOutputs.front().spatialMappings.empty())
    fail("joint Mapping plan lost an intermediate result projection");
  const auto &systemNode = std::get<loom::dse::GeneratePlanNodeDefinition>(
      plan.resolvedConfig.dse.planNodes
          [plan.pairOutputs.front().systemMappings.producerNodeOrdinal]);
  const auto &join = std::get<loom::dse::BoundedPlanOutputJoin>(
      systemNode.inputBindings[1]);
  if (join.outputs.empty() || join.maximumArtifacts != 32)
    fail("joint Mapping plan lost its explicit SpatialMapping bound");
  for (const loom::dse::PlanOutputRef &spatialOutput : join.outputs) {
    const auto &spatialNode = std::get<loom::dse::GeneratePlanNodeDefinition>(
        plan.resolvedConfig.dse.planNodes[spatialOutput.producerNodeOrdinal]);
    const auto &techOutput =
        std::get<loom::dse::PlanOutputRef>(spatialNode.inputBindings.front());
    const auto &techNode = std::get<loom::dse::GeneratePlanNodeDefinition>(
        plan.resolvedConfig.dse.planNodes[techOutput.producerNodeOrdinal]);
    if (techNode.descriptor !=
        loom::dse::applicationGraphTechMappingCandidateGeneratorDescriptor()
            .reference())
      fail("joint Mapping plan used a whole-program TechMapping cover");
  }

  auto view = take(loom::dse::projectResolvedDseConfigView(
      plan.resolvedConfig));
  auto execution = take(loom::dse::executeDsePlan(view, store, blobs));
  const auto *completed =
      std::get_if<loom::dse::CompletedDsePlanExecution>(&execution);
  if (!completed)
    fail("joint Mapping plan remained incomplete: " +
         loom::dse::toString(
             std::get<loom::dse::IncompleteDsePlanExecution>(execution)
                 .reason()));
  const std::vector<loom::ArtifactRootReference> mappings =
      completed->resolve(plan.pairOutputs.front().systemMappings).vec();
  if (mappings.empty())
    fail("joint Mapping plan produced no complete SystemMapping");
  for (const loom::ArtifactRootReference &reference : mappings) {
    auto mapping = take(loom::mapping::importSystemMapping(reference, store));
    if (mapping.view().dataflowIdentity() !=
            plan.frontier.pairs.front().software.dataflow.artifact ||
        mapping.view().fabricIdentity() != system.artifact)
      fail("joint Mapping output lost its exact pair owners");
  }

  const std::vector<loom::ArtifactRootReference> systems = {system,
                                                             alternateSystem};
  const std::vector<loom::dse::JointMemberPromotion> memberPromotions = {
      {plan.frontier.pairs.front().software.dataflow,
       loom::dse::CompletedSelection{mappings, {}}}};
  auto selected = take(loom::dse::selectJointDesignSystems(
      systems, memberPromotions, {}, loom::dse::AllPassingSelection{}, nullptr,
      store));
  const bool covered = everyCoreIsUsed(system, mappings, store);
  bool sawMissingAlternate = false;
  bool sawUnusedPrimary = false;
  std::vector<loom::dse::JointSystemGateOutcome> *outcomes = nullptr;
  if (auto *completedSelection =
          std::get_if<loom::dse::JointDesignSelection>(&selected)) {
    outcomes = &completedSelection->systemOutcomes;
    if (!covered || completedSelection->selectedSystems !=
                        std::vector<loom::ArtifactRootReference>{system} ||
        completedSelection->acceptedMappings != mappings)
      fail("aggregate selection bypassed member-local System gates");
  } else {
    auto &noFeasible =
        std::get<loom::dse::JointDesignNoFeasibleSystem>(selected);
    outcomes = &noFeasible.systemOutcomes;
    if (covered)
      fail("fully covered System was rejected before aggregate selection");
  }
  for (const loom::dse::JointSystemGateOutcome &outcome : *outcomes) {
    if (const auto *missing =
            std::get_if<loom::dse::JointSystemMissingMember>(&outcome))
      sawMissingAlternate |= missing->system == alternateSystem;
    if (const auto *unused =
            std::get_if<loom::dse::JointSystemUnusedAccCore>(&outcome))
      sawUnusedPrimary |= unused->system == system;
  }
  if (!sawMissingAlternate || sawUnusedPrimary == covered)
    fail("typed System dispositions lost missing-member or AccCore coverage");

  auto oversized = loom::dse::buildBoundedJointFrontier(
      {{{firstWorkload}, {secondWorkload}}, {system}},
      take(loom::dse::JointDesignPolicy::get(1, 1, 1, 1)), store);
  if (oversized)
    fail("joint frontier accepted a software set beyond its resolved bound");
  const std::string oversizedMessage =
      llvm::toString(oversized.takeError());
  if (!llvm::StringRef(oversizedMessage).contains("exceeds"))
    fail("frontier-bound rejection lost its diagnostic");
}

} // namespace

int main() {
  exerciseJointExploration();
  return 0;
}
