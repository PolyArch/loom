#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "DSE/CandidateGenerator.h"
#include "DSE/StructuredExecutionShapeCandidateGenerator.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Frontend/Compilation/PreMappingCompilation.h"
#include "Frontend/Compilation/StructuredExecutionShape.h"
#include "Frontend/IR/LoomDialect.h"
#include "Frontend/IR/LoomOps.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Frontend/Raising/StructuredRaising.h"
#include "Simulator/NativeSimulationOracle.h"
#include "Simulator/SimulationInputCapture.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

namespace {

[[noreturn]] void fail(const std::string &message) {
  llvm::errs() << "structuredExecutionShapeGenerator: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

mlir::MLIRContext &context() {
  static mlir::MLIRContext *result = [] {
    mlir::DialectRegistry registry;
    registry.insert<dataflow::DataflowDialect, loom::LoomDialect,
                    mlir::arith::ArithDialect, mlir::LLVM::LLVMDialect,
                    mlir::math::MathDialect>();
    auto *created =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    created->loadAllAvailableDialects();
    return created;
  }();
  return *result;
}

std::unique_ptr<llvm::Module> parseFmuladd(llvm::LLVMContext &context,
                                           bool riscvTarget = true) {
  constexpr llvm::StringLiteral source = R"llvm(
declare float @llvm.fmuladd.f32(float, float, float)

define void @fma_kernel(ptr %a, ptr %b, ptr %c, ptr %output) {
entry:
  %av = load float, ptr %a, align 4
  %bv = load float, ptr %b, align 4
  %cv = load float, ptr %c, align 4
  %result = call float @llvm.fmuladd.f32(float %av, float %bv, float %cv)
  store float %result, ptr %output, align 4
  ret void
}

define i32 @main() {
entry:
  %a = alloca float, align 4
  %b = alloca float, align 4
  %c = alloca float, align 4
  %output = alloca float, align 4
  %one_plus_epsilon = bitcast i32 1065353217 to float
  %negative_rounded_product = bitcast i32 -1082130430 to float
  store float %one_plus_epsilon, ptr %a, align 4
  store float %one_plus_epsilon, ptr %b, align 4
  store float %negative_rounded_product, ptr %c, align 4
  store float 0.000000e+00, ptr %output, align 4
  call void @fma_kernel(ptr %a, ptr %b, ptr %c, ptr %output)
  ret i32 0
}
)llvm";
  llvm::SMDiagnostic diagnostic;
  auto buffer = llvm::MemoryBuffer::getMemBuffer(source, "<fmuladd>");
  auto module = llvm::parseIR(buffer->getMemBufferRef(), diagnostic, context);
  if (!module) {
    std::string message;
    llvm::raw_string_ostream stream(message);
    diagnostic.print("structuredExecutionShapeGenerator", stream);
    fail(stream.str());
  }
  if (riscvTarget) {
    module->setDataLayout("e-m:e-p:64:64-i64:64-n32:64-S128");
    module->setTargetTriple(llvm::Triple("riscv64-unknown-unknown-elf"));
  }
  return module;
}

void configureHostModule(llvm::Module &module) {
  static const bool initializationFailed = [] {
    return llvm::InitializeNativeTarget() ||
           llvm::InitializeNativeTargetAsmPrinter();
  }();
  if (initializationFailed)
    fail("cannot initialize the native target");
  auto target = take(llvm::orc::JITTargetMachineBuilder::detectHost());
  module.setTargetTriple(target.getTargetTriple());
  module.setDataLayout(take(target.getDefaultDataLayoutForTarget()));
}

loom::frontend::StructuredEntityRef
findCallable(const loom::frontend::StructuredProgramCandidate &candidate,
             llvm::StringRef name) {
  auto view = take(candidate.view());
  for (const loom::frontend::StructuredEntity &entity :
       view.entities(loom::frontend::StructuredEntityKind::Operation)) {
    auto callable =
        llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(entity.operation);
    if (callable && callable.getSymName() == name)
      return entity.reference;
  }
  fail("Structured callable does not resolve");
}

dataflow::RootedGraphLaunchRef
onlyLaunch(const dataflow::CanonicalDataflowProgramView &view) {
  if (view.rootThreadLaunches().size() != 1 ||
      view.staticGraphLaunches().size() != 1)
    fail("fmuladd candidate does not have one rooted graph launch");
  return dataflow::RootedGraphLaunchRef{view.rootThreadLaunches().front().ref,
                                        view.staticGraphLaunches().front().ref};
}

dataflow::LogicalMemoryRootRef
memoryRoot(const dataflow::CanonicalDataflowProgramView &view,
           unsigned threadFormal) {
  for (const dataflow::CanonicalLogicalMemoryRootView &root :
       view.logicalMemoryRoots()) {
    if (root.formalArgIndex && *root.formalArgIndex == threadFormal)
      return root.ref;
    auto service = llvm::dyn_cast_or_null<dataflow::MemoryServiceOp>(root.op);
    auto source =
        service ? llvm::dyn_cast<mlir::BlockArgument>(service.getPointer())
                : mlir::BlockArgument{};
    if (source && source.getArgNumber() == threadFormal)
      return root.ref;
  }
  fail("fmuladd candidate is missing its output memory root");
}

mlir::LLVM::CallOp findHostCall(mlir::ModuleOp module, llvm::StringRef caller,
                                llvm::StringRef callee) {
  mlir::LLVM::CallOp result;
  module.walk([&](mlir::LLVM::CallOp call) {
    auto function = call->getParentOfType<mlir::LLVM::LLVMFuncOp>();
    if (function && function.getSymName() == caller && call.getCalleeAttr() &&
        call.getCalleeAttr().getValue() == callee)
      result = call;
  });
  if (!result)
    fail("fmuladd candidate has no host call site");
  return result;
}

const loom::sim::SimulationMemoryRootCapture &
captureBinding(const loom::sim::SimulationInputCapturePlan &plan,
               dataflow::LogicalMemoryRootRef root) {
  for (const loom::sim::SimulationMemoryRootCapture &binding :
       plan.memoryRootBindings)
    if (binding.root == root)
      return binding;
  fail("fmuladd capture plan is missing its output memory root");
}

loom::frontend::StructuredProgramCandidate parseProgram() {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  llvm.func @host_choice(%x: f32, %y: f32, %z: f32) -> f32 {
    %r = llvm.intr.fmuladd(%x, %y, %z)
        : (f32, f32, f32) -> f32
    llvm.return %r : f32
  }

  dataflow.thread private @selected domain(#dataflow.thread_domain<dense>)(
      %lhs: f32, %rhs: f32, %acc: f32) ctrl (%start: none) {
    %result = "loom.spatial_region"(%lhs, %rhs, %acc)
        <{operandSegmentSizes = array<i32: 3, 0, 0, 0>,
          resultSegmentSizes = array<i32: 1, 0>}> ({
      ^bb0(%a: f32, %b: f32, %c: f32):
        %first = llvm.intr.fmuladd(%a, %b, %c)
            : (f32, f32, f32) -> f32
        %second = llvm.intr.fmuladd(%first, %b, %c)
            : (f32, f32, f32) -> f32
        "loom.spatial_yield"(%second)
            <{operandSegmentSizes = array<i32: 1, 0>}> : (f32) -> ()
    }) {graph_name = "selected_graph", source_maps = []} :
        (f32, f32, f32) -> f32
    dataflow.thread.yield
  }

  llvm.func @entry(%lhs: f32, %rhs: f32, %acc: f32) {
    %token = dataflow.thread.launch @selected(%lhs, %rhs, %acc)
        : (f32, f32, f32) -> !dataflow.thread_token
    dataflow.thread.wait %token : !dataflow.thread_token
    llvm.return
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail("cannot parse Structured execution-shape fixture");
  return take(loom::frontend::finalizeStructuredProgram(module.get()));
}

void finiteUniformDomainAndScopedMaterialization() {
  auto parent = parseProgram();
  auto decisions =
      take(loom::frontend::enumerateStructuredExecutionShapeDecisions(parent));
  using Shape = loom::raising::FMulAddExecutionShape;
  if (decisions.size() != 2 || decisions[0].fmuladdShape != Shape::Fused ||
      decisions[1].fmuladdShape != Shape::Split)
    fail("execution-shape domain is not the canonical Fused/Split pair");

  for (const loom::frontend::StructuredExecutionShapeDecision &decision :
       decisions) {
    auto child =
        take(loom::frontend::materializeStructuredExecutionShapeDecision(
            parent, decision));
    std::size_t spatialFmuladd = 0;
    std::size_t spatialFma = 0;
    std::size_t spatialMulf = 0;
    std::size_t spatialAddf = 0;
    child.structuredProgram.module().walk([&](loom::SpatialRegionOp spatial) {
      spatial.walk([&](mlir::LLVM::FMulAddOp) { ++spatialFmuladd; });
      spatial.walk([&](mlir::math::FmaOp) { ++spatialFma; });
      spatial.walk([&](mlir::arith::MulFOp) { ++spatialMulf; });
      spatial.walk([&](mlir::arith::AddFOp) { ++spatialAddf; });
    });
    if (spatialFmuladd != 0)
      fail("selected Spatial region retained an unresolved fmuladd");
    if (decision.fmuladdShape == Shape::Fused &&
        (spatialFma != 2 || spatialMulf != 0 || spatialAddf != 0))
      fail("Fused decision did not materialize every selected operation");
    if (decision.fmuladdShape == Shape::Split &&
        (spatialFma != 0 || spatialMulf != 2 || spatialAddf != 2))
      fail("Split decision did not materialize every selected operation");

    auto host =
        child.structuredProgram.module().lookupSymbol<mlir::LLVM::LLVMFuncOp>(
            "host_choice");
    std::size_t hostFmuladd = 0;
    host.walk([&](mlir::LLVM::FMulAddOp) { ++hostFmuladd; });
    if (hostFmuladd != 1)
      fail("execution-shape decision escaped its selected Spatial region");
  }
}

void selectedShapesRemainNativeObservable() {
  llvm::SmallString<128> directory;
  std::error_code error =
      llvm::sys::fs::createUniqueDirectory("loom-fmuladd-oracle", directory);
  if (error)
    fail("cannot create ArtifactStore directory: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));

  llvm::LLVMContext targetContext;
  auto compiled = take(loom::frontend::compileLlvmModuleToPreMapping(
      parseFmuladd(targetContext), design.roots().front().reference(), store));
  const loom::frontend::StructuredEntityRef callable =
      findCallable(compiled.structuredProgram, "fma_kernel");
  auto ownershipDomain =
      take(loom::frontend::enumerateSpatialOwnershipDecisionDomain(
          compiled.structuredProgram, callable));
  auto owned =
      take(loom::frontend::materializeStructuredSpatialOwnershipDecision(
          compiled.structuredProgram, {callable}, ownershipDomain.front()));
  auto shaped =
      take(loom::frontend::materializeStructuredExecutionShapeDecision(
          std::move(owned), {loom::raising::FMulAddExecutionShape::Fused}));
  auto candidate = take(loom::frontend::finalizeSpatialOwnershipCandidate(
      std::move(shaped), design.roots().front()));
  auto view = take(candidate.canonicalDataflow.view());
  dataflow::RootedGraphLaunchRef launch = onlyLaunch(view);
  auto plan = take(loom::sim::deriveSimulationInputCapturePlan(
      view, launch,
      findHostCall(candidate.canonicalDataflow.module(), "main",
                   "fma_kernel")));
  const auto &outputBinding = captureBinding(plan.input, memoryRoot(view, 3));
  if (outputBinding.floatingWriteLaneType !=
      mlir::Float32Type::get(candidate.canonicalDataflow.module().getContext()))
    fail("fmuladd output is not a uniform floating write root");

  llvm::LLVMContext hostContext;
  std::unique_ptr<llvm::Module> hostModule = parseFmuladd(hostContext, false);
  configureHostModule(*hostModule);
  auto host = take(
      loom::raising::raiseLlvmModuleToStructuredProgram(std::move(hostModule)));
  loom::frontend::SpatialOwnershipScope hostScope{
      findCallable(host, "fma_kernel")};
  auto captureShape = [&](loom::raising::FMulAddExecutionShape shape) {
    auto prepared = take(loom::frontend::prepareSpatialOwnershipSelection(
        host, hostScope,
        loom::frontend::SpatialOwnershipDecisionPoint{std::nullopt}));
    loom::raising::materializeFMulAddInOperation(*prepared.operation, shape);
    auto nativeContext = std::make_unique<llvm::LLVMContext>();
    std::unique_ptr<llvm::Module> nativeModule =
        parseFmuladd(*nativeContext, false);
    configureHostModule(*nativeModule);
    return take(loom::sim::executeStructuredDirectCallSimulationInputCapture(
        llvm::orc::ThreadSafeModule(std::move(nativeModule),
                                    std::move(nativeContext)),
        std::move(prepared.module), plan));
  };
  loom::sim::NativeSimulationInputCapture fused =
      captureShape(loom::raising::FMulAddExecutionShape::Fused);
  loom::sim::NativeSimulationInputCapture split =
      captureShape(loom::raising::FMulAddExecutionShape::Split);
  if (fused.entryResult != 0 || split.entryResult != 0 ||
      fused.calls.size() != 1 || split.calls.size() != 1 ||
      outputBinding.objectIndex >= fused.calls.front().objects.size() ||
      outputBinding.objectIndex >= split.calls.front().objects.size())
    fail("fmuladd native captures are malformed");
  const auto &fusedOutput =
      fused.calls.front().objects[outputBinding.objectIndex];
  const auto &splitOutput =
      split.calls.front().objects[outputBinding.objectIndex];
  if (fusedOutput.finalBytes == splitOutput.finalBytes)
    fail("typed fmuladd decisions did not produce distinct results");

  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail("cannot remove ArtifactStore directory: " + cleanup.message());
}

void ownershipLineageIsMechanicallyReprojected() {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  dataflow.thread private @selected domain(#dataflow.thread_domain<dense>)(
      %lhs: f32, %rhs: f32, %acc: f32) ctrl (%start: none) {
    %result = "loom.spatial_region"(%lhs, %rhs, %acc)
        <{operandSegmentSizes = array<i32: 3, 0, 0, 0>,
          resultSegmentSizes = array<i32: 1, 0>}> ({
      ^bb0(%a: f32, %b: f32, %c: f32):
        %value = llvm.intr.fmuladd(%a, %b, %c)
            : (f32, f32, f32) -> f32
        "loom.spatial_yield"(%value)
            <{operandSegmentSizes = array<i32: 1, 0>}> : (f32) -> ()
    }) {graph_name = "selected_graph", source_maps = []} :
        (f32, f32, f32) -> f32
    dataflow.thread.yield
  }

  llvm.func @entry(%lhs: f32, %rhs: f32, %acc: f32) {
    %token = dataflow.thread.launch @selected(%lhs, %rhs, %acc)
        : (f32, f32, f32) -> !dataflow.thread_token
    dataflow.thread.wait %token : !dataflow.thread_token
    llvm.return
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail("cannot parse execution-shape lineage fixture");
  module->walk([&](mlir::LLVM::FMulAddOp operation) {
    operation->setLoc(
        mlir::FileLineColLoc::get(&context(), "operator.c", 17, 5));
  });

  std::vector<mlir::Block *> blocks;
  module->walk([&](mlir::Operation *operation) {
    for (mlir::Region &region : operation->getRegions())
      for (mlir::Block &block : region)
        blocks.push_back(&block);
  });
  auto projected =
      take(loom::frontend::finalizeStructuredProgramWithTrackedBlocks(
          module.get(), blocks));
  if (projected.sourceProvenance.empty() ||
      projected.trackedBlocks.size() != blocks.size())
    fail("lineage fixture did not retain its source projections");

  std::vector<loom::frontend::StructuredBlockActivityLineage> lineage;
  lineage.reserve(projected.trackedBlocks.size());
  for (const loom::frontend::StructuredEntityRef &block :
       projected.trackedBlocks)
    lineage.push_back({block, block});
  const auto expectedParents = lineage;
  loom::frontend::MaterializedStructuredOwnershipCandidate parent{
      std::move(projected.artifact), std::move(lineage),
      std::move(projected.sourceProvenance)};
  auto child = take(loom::frontend::materializeStructuredExecutionShapeDecision(
      std::move(parent), {loom::raising::FMulAddExecutionShape::Fused}));

  if (!llvm::any_of(child.sourceProvenance, [](const auto &provenance) {
        return provenance.sourceFiles == std::vector<std::string>{"operator.c"};
      }))
    fail("execution-shape materialization lost source provenance");
  if (child.blockActivityLineage.size() != expectedParents.size())
    fail("execution-shape materialization changed block-lineage cardinality");
  auto view = take(child.structuredProgram.view());
  for (auto [actual, expected] :
       llvm::zip_equal(child.blockActivityLineage, expectedParents)) {
    if (actual.parentBlock != expected.parentBlock ||
        actual.childBlock.parent != child.structuredProgram.identity() ||
        actual.childBlock.kind != loom::frontend::StructuredEntityKind::Block)
      fail("execution-shape materialization changed block lineage");
    if (!take(view.resolve(actual.childBlock)).block)
      fail("execution-shape block lineage does not resolve in the child");
  }
}

void centralGeneratorPublishesOnlyAdmittedUniformShapes() {
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-execution-shape-generator", directory);
  if (error)
    fail("cannot create ArtifactStore directory: " + error.message());
  loom::ArtifactStore store(directory);
  llvm::SmallString<128> blobPath(directory);
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  auto design = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));
  auto parent = parseProgram();
  auto parentReference =
      take(loom::frontend::publishStructuredProgram(parent, store));
  auto config = take(
      loom::dse::projectResolvedStructuredExecutionShapeGeneratorConfigView());
  auto inputs =
      take(loom::dse::bindStructuredExecutionShapeCandidateGeneratorInputs(
          {parentReference}, design.roots().front().reference()));
  auto binding =
      take(loom::dse::resolveStructuredExecutionShapeCandidateGeneratorBinding(
          config));
  auto outcome =
      take(loom::dse::invokeCandidateGenerator(inputs, binding, store, blobs));
  auto *completed = std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
      &outcome.outcome);
  if (!completed || completed->outputBindings.size() != 1 ||
      completed->outputBindings.front().artifacts.size() != 2)
    fail("central generator did not publish the Fused/Split candidate pair");
  if (completed->lineageEdges.size() != 2)
    fail("central generator lost Fused/Split lineage");
  for (const loom::dse::CandidateGeneratorLineageEdge &edge :
       completed->lineageEdges) {
    if (edge.parents !=
        std::vector<loom::ArtifactRootReference>{parentReference})
      fail("execution-shape lineage changed its exact parent");
    take(loom::frontend::adoptStructuredExecutionShapeDecision(
        edge.ownerPayload));
  }

  std::size_t fused = 0;
  std::size_t split = 0;
  for (const loom::ArtifactRootReference &reference :
       completed->outputBindings.front().artifacts) {
    auto child =
        take(loom::frontend::importStructuredProgram(reference, store));
    std::size_t fma = 0;
    std::size_t mulf = 0;
    std::size_t addf = 0;
    child.module().walk([&](loom::SpatialRegionOp spatial) {
      spatial.walk([&](mlir::math::FmaOp) { ++fma; });
      spatial.walk([&](mlir::arith::MulFOp) { ++mulf; });
      spatial.walk([&](mlir::arith::AddFOp) { ++addf; });
    });
    if (fma == 2 && mulf == 0 && addf == 0)
      ++fused;
    else if (fma == 0 && mulf == 2 && addf == 2)
      ++split;
    else
      fail("central generator published a mixed or unresolved shape");
  }
  if (fused != 1 || split != 1)
    fail("central generator duplicated or omitted one execution shape");
}

void invalidInMemoryDecisionFailsClosed() {
  auto parent = parseProgram();
  const loom::frontend::StructuredExecutionShapeDecision decision{
      static_cast<loom::raising::FMulAddExecutionShape>(99)};
  auto encoded =
      loom::frontend::encodeStructuredExecutionShapeDecision(decision);
  if (encoded)
    fail("execution-shape encoder accepted an unknown in-memory shape");
  llvm::consumeError(encoded.takeError());
  auto materialized =
      loom::frontend::materializeStructuredExecutionShapeDecision(parent,
                                                                  decision);
  if (materialized)
    fail("execution-shape materializer accepted an unknown in-memory shape");
  llvm::consumeError(materialized.takeError());
}

void ownerPayloadMustBelongToExactParentDecisionDomain() {
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-execution-shape-parent-domain", directory);
  if (error)
    fail("cannot create ArtifactStore directory: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));

  auto unresolved = parseProgram();
  auto fused = take(loom::frontend::materializeStructuredExecutionShapeDecision(
      unresolved, {loom::raising::FMulAddExecutionShape::Fused}));
  auto split = take(loom::frontend::materializeStructuredExecutionShapeDecision(
      unresolved, {loom::raising::FMulAddExecutionShape::Split}));
  auto fusedReference = take(
      loom::frontend::publishStructuredProgram(fused.structuredProgram, store));
  auto splitReference = take(
      loom::frontend::publishStructuredProgram(split.structuredProgram, store));

  auto config = take(
      loom::dse::projectResolvedStructuredExecutionShapeGeneratorConfigView());
  auto inputs =
      take(loom::dse::bindStructuredExecutionShapeCandidateGeneratorInputs(
          {fusedReference}, design.roots().front().reference()));
  auto binding =
      take(loom::dse::resolveStructuredExecutionShapeCandidateGeneratorBinding(
          config));
  auto payload = take(loom::frontend::encodeStructuredExecutionShapeDecision(
      {loom::raising::FMulAddExecutionShape::Split}));
  std::vector<loom::dse::CandidateGeneratorOutputBinding> outputs = {
      {loom::dse::CandidateGeneratorOutputSlotRef(0), {splitReference}}};
  std::vector<loom::dse::CandidateGeneratorLineageEdge> lineage = {{
      loom::dse::CandidateGeneratorLineageEdgeKind::CandidateDecision,
      loom::dse::CandidateGeneratorOutputSlotRef(0),
      splitReference,
      {fusedReference},
      std::move(payload),
  }};
  llvm::Error validation =
      loom::dse::validateCanonicalCandidateGeneratorInvocation(
          inputs, binding, outputs, lineage, true, store);
  if (!validation)
    fail("execution-shape owner accepted a decision outside the exact parent "
         "domain");
  llvm::consumeError(std::move(validation));

  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail("cannot remove ArtifactStore directory: " + cleanup.message());
}

} // namespace

int main() {
  finiteUniformDomainAndScopedMaterialization();
  selectedShapesRemainNativeObservable();
  ownershipLineageIsMechanicallyReprojected();
  centralGeneratorPublishesOnlyAdmittedUniformShapes();
  invalidInMemoryDecisionFailsClosed();
  ownerPayloadMustBelongToExactParentDecisionDomain();
  return EXIT_SUCCESS;
}
