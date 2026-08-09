#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"
#include "Frontend/Compilation/StructuredExecutionShape.h"
#include "Frontend/Compilation/StructuredMemoryCommunication.h"
#include "Frontend/Compilation/StructuredSpecialMathAccuracy.h"
#include "Frontend/Lowering/CanonicalDataflowLowering.h"
#include "Simulator/DFGSimulator.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <string>
#include <utility>
#include <variant>

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "source-backed attention channel anchor failed: " << message
               << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

loom::frontend::StructuredEntityRef
findCall(const loom::frontend::StructuredProgramCandidate &candidate,
         llvm::StringRef callee) {
  auto view = take(candidate.view());
  std::optional<loom::frontend::StructuredEntityRef> result;
  for (const loom::frontend::StructuredEntity &entity :
       view.entities(loom::frontend::StructuredEntityKind::Operation)) {
    auto call = llvm::dyn_cast_or_null<mlir::LLVM::CallOp>(entity.operation);
    if (!call || !call.getCalleeAttr() ||
        call.getCalleeAttr().getValue() != callee)
      continue;
    if (result)
      fail("callee has more than one direct call site: " + callee);
    result = entity.reference;
  }
  if (!result)
    fail("callee has no direct call site: " + callee);
  return *result;
}

loom::frontend::StructuredProgramCandidate
materializeCallLeaf(const loom::frontend::StructuredProgramCandidate &parent,
                    llvm::StringRef callee) {
  loom::frontend::StructuredEntityRef call = findCall(parent, callee);
  auto domain = take(
      loom::frontend::enumerateSpatialOwnershipDecisionDomain(parent, call));
  const auto selected = llvm::find_if(
      domain, [&](const loom::frontend::SpatialOwnershipDecisionPoint &point) {
        if (!point.directCallInlining ||
            point.directCallInlining->callSite != call)
          return false;
        if (!point.addressProjection)
          return true;
        return std::holds_alternative<
            loom::frontend::PointerAddressedAddressProjection>(
            *point.addressProjection);
      });
  if (selected == domain.end())
    fail("direct-call leaf has no exact pointer-addressed inline decision: " +
         callee);
  auto child =
      take(loom::frontend::materializeStructuredSpatialOwnershipDecision(
          parent, {call}, *selected));
  return std::move(child.structuredProgram);
}

loom::frontend::StructuredProgramCandidate
promoteUniqueChannel(const loom::frontend::StructuredProgramCandidate &parent) {
  auto domain =
      take(loom::frontend::enumerateStructuredMemoryCommunicationDecisions(
          parent, 64));
  const loom::frontend::StructuredMemoryCommunicationDecision *selected =
      nullptr;
  for (const auto &decision : domain.decisions) {
    if (loom::frontend::structuredMemoryCommunicationDecisionKind(decision) !=
        loom::frontend::StructuredMemoryCommunicationDecisionKind::
            PromoteSpscBufferToChannel) {
      continue;
    }
    if (selected)
      fail("attention exposed more than one channel-promotion decision");
    selected = &decision;
  }
  if (!selected)
    fail("attention exposed no channel-promotion decision");
  auto child =
      take(loom::frontend::materializeStructuredMemoryCommunicationDecision(
          parent, *selected));
  return std::move(child.structuredProgram);
}

loom::frontend::StructuredProgramCandidate closeExecutionSemantics(
    const loom::frontend::StructuredProgramCandidate &parent) {
  using Shape = loom::raising::FMulAddExecutionShape;
  auto shapes =
      take(loom::frontend::enumerateStructuredExecutionShapeDecisions(parent));
  if (shapes.size() != 2 || shapes[0].fmuladdShape != Shape::Fused ||
      shapes[1].fmuladdShape != Shape::Split)
    fail("attention did not expose the canonical Fused/Split shape domain");
  auto shaped =
      take(loom::frontend::materializeStructuredExecutionShapeDecision(
          parent, shapes[0]));
  auto candidate = std::move(shaped.structuredProgram);

  while (true) {
    auto accuracy =
        take(loom::frontend::enumerateStructuredSpecialMathAccuracyDecisions(
            candidate));
    if (accuracy.empty())
      return candidate;
    if (accuracy.size() != 1 ||
        accuracy.front().accuracy !=
            loom::SpecialMathAccuracyTier::CorrectlyRounded)
      fail("attention exposed an unproved relaxed special-math tier");
    auto child =
        take(loom::frontend::materializeStructuredSpecialMathAccuracyDecision(
            candidate, accuracy.front()));
    candidate = std::move(child.structuredProgram);
  }
}

dataflow::RootedGraphLaunchRef
rootedLaunchForGraph(const dataflow::CanonicalDataflowProgramView &view,
                     dataflow::GraphOp graph) {
  std::optional<dataflow::RootedGraphLaunchRef> result;
  view.forEachRootedGraphLaunch([&](dataflow::RootedGraphLaunchRef candidate) {
    dataflow::GraphRef graphRef = take(view.resolve(candidate));
    if (take(view.resolve(graphRef)).op != graph.getOperation())
      return;
    if (result)
      fail("attention graph has more than one rooted launch");
    result = candidate;
  });
  if (!result)
    fail("attention graph has no rooted launch");
  return *result;
}

std::vector<dataflow::LogicalMemoryRootRef>
launchMemoryRoots(const dataflow::CanonicalDataflowProgramView &view,
                  dataflow::RootedGraphLaunchRef rooted) {
  auto site = take(view.resolve(rooted.staticGraphLaunch));
  auto launch = llvm::cast<dataflow::GraphLaunchOp>(site.op);
  std::vector<dataflow::LogicalMemoryRootRef> result;
  result.reserve(launch.getMemoryInputs().size());
  for (mlir::Value input : launch.getMemoryInputs()) {
    mlir::Operation *owner = input.getDefiningOp();
    auto found = llvm::find_if(
        view.logicalMemoryRoots(),
        [&](const dataflow::CanonicalLogicalMemoryRootView &root) {
          return root.op == owner;
        });
    if (!owner || found == view.logicalMemoryRoots().end())
      fail("attention graph memory input has no canonical service root");
    result.push_back(found->ref);
  }
  return result;
}

loom::sim::RuntimeMemoryObject
f32Memory(std::initializer_list<std::uint32_t> values) {
  loom::sim::RuntimeMemoryObject object;
  object.initialBytes.reserve(values.size() * sizeof(std::uint32_t));
  for (std::uint32_t value : values)
    for (unsigned byte = 0; byte != sizeof(value); ++byte)
      object.initialBytes.push_back(
          {loom::sim::SemanticState::Defined,
           static_cast<std::uint8_t>(value >> (byte * 8))});
  return object;
}

loom::sim::CanonicalValueSequence pointerValue(std::uint64_t objectOrdinal) {
  return {1,
          {loom::sim::SemanticLane::definedPointer(
              llvm::APInt(64, 0x1000 + objectOrdinal * 0x1000), objectOrdinal,
              llvm::APInt(64, 0))}};
}

loom::sim::RetiredDFGSimulation
simulateProducer(const dataflow::CanonicalDataflowArtifact &artifact,
                 const dataflow::CanonicalDataflowProgramView &view,
                 dataflow::GraphOp producer) {
  dataflow::RootedGraphLaunchRef launch = rootedLaunchForGraph(view, producer);
  std::vector<dataflow::LogicalMemoryRootRef> roots =
      launchMemoryRoots(view, launch);
  if (roots.size() != 1)
    fail("attention producer does not have one score service root");

  loom::sim::SpatialSimulationWorkload workloadModel{launch};
  workloadModel.valueInputPlan = {loom::sim::RuntimeValueInput{}};
  workloadModel.observableContract.streamOutputs = {0};
  auto workload =
      take(loom::sim::finalizeSimulationWorkload(workloadModel, view));

  loom::sim::SpatialSimulationRuntimeInputDraft input{workload.identity()};
  input.runtimeValues = {{0, pointerValue(0)}};
  input.memoryObjects = {
      f32Memory({0x3f000000, 0x3f000000, 0x3f000000, 0x3f800000, 0x3f800000,
                 0x3f800000, 0xbf000000, 0xbf000000, 0xbf000000})};
  input.memoryRootBindings = {{roots.front(), 0, 0}};
  auto runtime =
      take(loom::sim::finalizeSimulationRuntimeInput(input, workload, view));
  return take(
      loom::sim::simulateRetiredDfgWorkload(artifact, workload, runtime));
}

loom::sim::RetiredDFGSimulation
simulateConsumer(const dataflow::CanonicalDataflowArtifact &artifact,
                 const dataflow::CanonicalDataflowProgramView &view,
                 dataflow::GraphOp consumer) {
  dataflow::RootedGraphLaunchRef launch = rootedLaunchForGraph(view, consumer);
  std::vector<dataflow::LogicalMemoryRootRef> roots =
      launchMemoryRoots(view, launch);
  if (roots.size() != 2)
    fail("attention consumer does not have V and output service roots");

  loom::sim::SpatialSimulationWorkload workloadModel{launch};
  workloadModel.valueInputPlan = {loom::sim::RuntimeValueInput{},
                                  loom::sim::RuntimeValueInput{}};
  workloadModel.observableContract.memories.push_back(
      {dataflow::LogicalMemoryRootOrViewRef{roots[1]},
       loom::sim::MemoryObservationForm::FullState});
  auto workload =
      take(loom::sim::finalizeSimulationWorkload(workloadModel, view));

  loom::sim::CanonicalStreamSequence probabilities;
  probabilities.values.tokenCount = 9;
  for (unsigned ordinal = 0; ordinal != 9; ++ordinal)
    probabilities.values.lanes.push_back(
        loom::sim::SemanticLane::defined(llvm::APInt(32, 0x3eaaaaab)));
  probabilities.termination = loom::sim::StreamTermination::ClosedAfterLast;

  loom::sim::SpatialSimulationRuntimeInputDraft input{workload.identity()};
  input.runtimeValues = {{0, pointerValue(0)}, {1, pointerValue(1)}};
  input.runtimeStreams = {std::move(probabilities)};
  input.memoryObjects = {f32Memory({0x3f800000, 0x40000000, 0x40400000,
                                    0x40800000, 0x40a00000, 0x40c00000}),
                         f32Memory({0, 0, 0, 0, 0, 0})};
  input.memoryRootBindings = {{roots[0], 0, 0}, {roots[1], 1, 0}};
  auto runtime =
      take(loom::sim::finalizeSimulationRuntimeInput(input, workload, view));
  return take(
      loom::sim::simulateRetiredDfgWorkload(artifact, workload, runtime));
}

void verifyCanonicalAttention(dataflow::CanonicalDataflowArtifact &artifact) {
  dataflow::GraphOp producer;
  dataflow::GraphOp consumer;
  std::size_t creates = 0;
  artifact.module().walk([&](dataflow::ChannelCreateOp) { ++creates; });
  artifact.module().walk([&](dataflow::GraphOp graph) {
    if (graph.getResultSegmentSizes()[1] == 1)
      producer = graph;
    if (graph.getInputSegmentSizes()[1] == 1)
      consumer = graph;
  });
  if (creates != 1 || !producer || !consumer)
    fail("canonical attention lost its unique producer/consumer channel");

  auto view = take(artifact.view());
  loom::sim::RetiredDFGSimulation produced =
      simulateProducer(artifact, view, producer);
  if (produced.report.status != "pass" ||
      produced.observations.streamOutputs.size() != 1)
    fail("attention producer changed the ordered probability payload");
  const loom::sim::CanonicalStreamSequence &probabilities =
      produced.observations.streamOutputs.front();
  if (probabilities.termination !=
          loom::sim::StreamTermination::ClosedAfterLast ||
      probabilities.values.tokenCount != 9 ||
      probabilities.values.lanes.size() != 9 ||
      llvm::any_of(probabilities.values.lanes,
                   [](const loom::sim::SemanticLane &lane) {
                     return lane.state != loom::sim::SemanticState::Defined ||
                            lane.bits != llvm::APInt(32, 0x3eaaaaab);
                   }))
    fail("attention producer changed the ordered probability payload");

  loom::sim::RetiredDFGSimulation consumed =
      simulateConsumer(artifact, view, consumer);
  loom::sim::RuntimeMemoryObject expected = f32Memory(
      {0x40400000, 0x40800000, 0x40400000, 0x40800000, 0x40400000, 0x40800000});
  const auto *output = consumed.observations.memories.size() == 1
                           ? std::get_if<loom::sim::FullMemoryObservation>(
                                 &consumed.observations.memories.front())
                           : nullptr;
  if (consumed.report.status != "pass" || !output ||
      output->bytes.size() != expected.initialBytes.size() ||
      !llvm::equal(output->bytes, expected.initialBytes,
                   [](const loom::sim::SemanticMemoryByte &lhs,
                      const loom::sim::SemanticMemoryByte &rhs) {
                     return lhs.state == rhs.state && lhs.value == rhs.value;
                   }))
    fail("attention consumer changed the terminal output sequence");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3)
    fail("expected <structured-input.mlir> <canonical-output.mlir>");
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  if (!module)
    fail("cannot parse the Structured input");

  auto candidate =
      take(loom::frontend::finalizeStructuredProgram(module.get()));
  candidate = materializeCallLeaf(candidate, "attention_softmax");
  candidate = materializeCallLeaf(candidate, "attention_values");
  candidate = closeExecutionSemantics(candidate);
  candidate = promoteUniqueChannel(candidate);
  auto canonical = take(
      loom::lowering::lowerStructuredProgramToCanonicalDataflow(candidate));
  verifyCanonicalAttention(canonical);

  std::error_code error;
  llvm::raw_fd_ostream output(argv[2], error, llvm::sys::fs::OF_Text);
  if (error)
    fail("cannot open canonical output: " + error.message());
  canonical.module().print(output);
  output << '\n';
  return EXIT_SUCCESS;
}
