#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/CandidateGenerator.h"
#include "DSE/StructuredScheduleCandidateGenerator.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Frontend/Compilation/FabricCapabilityIndex.h"
#include "Frontend/Compilation/StructuredSchedule.h"
#include "Frontend/Compilation/StructuredScop.h"
#include "Frontend/IR/LoomDialect.h"
#include "Frontend/IR/LoomOps.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Frontend/Lowering/CanonicalDataflowLowering.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <variant>

namespace {

[[noreturn]] void fail(const std::string &message) {
  llvm::errs() << "structuredScheduleGenerator: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireErrorContains(
    llvm::Expected<loom::dse::ResolvedStructuredScheduleGeneratorConfigView>
        value,
    llvm::StringRef fragment) {
  if (value)
    fail("expected invalid schedule config");
  std::string message = llvm::toString(value.takeError());
  if (!llvm::StringRef(message).contains(fragment))
    fail("unexpected schedule config error: " + message);
}

mlir::MLIRContext &context() {
  static mlir::MLIRContext *result = [] {
    mlir::DialectRegistry registry;
    registry.insert<dataflow::DataflowDialect, loom::LoomDialect,
                    mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                    mlir::DLTIDialect, mlir::func::FuncDialect,
                    mlir::math::MathDialect, mlir::LLVM::LLVMDialect,
                    mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                    mlir::vector::VectorDialect>();
    auto *created =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    created->loadAllAvailableDialects();
    return created;
  }();
  return *result;
}

loom::frontend::StructuredProgramCandidate parseProgram(llvm::StringRef text) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context());
  if (!module)
    fail("cannot parse Structured Program fixture");
  return take(loom::frontend::finalizeStructuredProgram(module.get()));
}

loom::frontend::StructuredEntityRef structuredLoopReference(
    const loom::frontend::StructuredProgramCandidate &candidate,
    llvm::StringRef functionName) {
  auto view = take(candidate.view());
  for (const loom::frontend::StructuredEntity &entity :
       view.entities(loom::frontend::StructuredEntityKind::Operation)) {
    mlir::Operation *loop = entity.operation;
    if (!llvm::isa_and_nonnull<mlir::scf::ForOp, mlir::affine::AffineForOp>(
            loop))
      continue;
    auto function = loop->getParentOfType<mlir::func::FuncOp>();
    if (function && function.getSymName() == functionName)
      return entity.reference;
  }
  fail("candidate has no selected structured loop");
}

std::optional<loom::frontend::StructuredScheduleDecision> firstVectorDecision(
    const loom::frontend::StructuredScheduleDecisionDomain &domain) {
  auto decision = llvm::find_if(domain.proposals, [](const auto &candidate) {
    return candidate.decision().kind ==
           loom::frontend::StructuredScheduleDecisionKind::Vectorize;
  });
  return decision == domain.proposals.end()
             ? std::nullopt
             : std::optional<loom::frontend::StructuredScheduleDecision>(
                   decision->decision());
}

std::uint64_t tripCount(mlir::scf::ForOp loop) {
  std::optional<llvm::APInt> count = loop.getStaticTripCount();
  if (!count || count->getActiveBits() > 64)
    fail("expected a host-representable static trip count");
  return count->getZExtValue();
}

std::optional<std::uint64_t> optionalTripCount(mlir::scf::ForOp loop) {
  std::optional<llvm::APInt> count = loop.getStaticTripCount();
  if (!count || count->getActiveBits() > 64)
    return std::nullopt;
  return count->getZExtValue();
}

std::optional<std::pair<std::uint64_t, std::uint64_t>>
outerInnerTrips(const loom::frontend::StructuredProgramCandidate &candidate,
                llvm::StringRef functionName) {
  mlir::func::FuncOp function =
      candidate.module().lookupSymbol<mlir::func::FuncOp>(functionName);
  if (!function)
    fail("candidate lost function " + functionName.str());
  mlir::scf::ForOp outer;
  for (mlir::Operation &operation : function.getBody().front()) {
    outer = llvm::dyn_cast<mlir::scf::ForOp>(&operation);
    if (outer)
      break;
  }
  if (!outer)
    return std::nullopt;
  auto inner = llvm::dyn_cast<mlir::scf::ForOp>(&outer.getBody()->front());
  if (!inner)
    return std::nullopt;
  return std::pair<std::uint64_t, std::uint64_t>{tripCount(outer),
                                                 tripCount(inner)};
}

std::size_t
storeCount(const loom::frontend::StructuredProgramCandidate &candidate,
           llvm::StringRef functionName) {
  mlir::func::FuncOp function =
      candidate.module().lookupSymbol<mlir::func::FuncOp>(functionName);
  if (!function)
    fail("candidate lost function " + functionName.str());
  std::size_t count = 0;
  function.walk([&](mlir::memref::StoreOp) { ++count; });
  return count;
}

std::size_t
forallCount(const loom::frontend::StructuredProgramCandidate &candidate,
            llvm::StringRef functionName) {
  mlir::func::FuncOp function =
      candidate.module().lookupSymbol<mlir::func::FuncOp>(functionName);
  if (!function)
    fail("candidate lost function " + functionName.str());
  std::size_t count = 0;
  function.walk([&](mlir::scf::ForallOp) { ++count; });
  return count;
}

std::size_t
maximumForallRank(const loom::frontend::StructuredProgramCandidate &candidate,
                  llvm::StringRef functionName) {
  mlir::func::FuncOp function =
      candidate.module().lookupSymbol<mlir::func::FuncOp>(functionName);
  if (!function)
    fail("candidate lost function " + functionName.str());
  std::size_t rank = 0;
  function.walk([&](mlir::scf::ForallOp forall) {
    rank = std::max(rank, static_cast<std::size_t>(forall.getRank()));
  });
  return rank;
}

bool hasUnrollAndJamShape(
    const loom::frontend::StructuredProgramCandidate &candidate,
    llvm::StringRef functionName) {
  mlir::func::FuncOp function =
      candidate.module().lookupSymbol<mlir::func::FuncOp>(functionName);
  if (!function)
    fail("candidate lost function " + functionName.str());
  mlir::scf::ForOp outer;
  for (mlir::Operation &operation : function.getBody().front()) {
    outer = llvm::dyn_cast<mlir::scf::ForOp>(&operation);
    if (outer)
      break;
  }
  if (!outer || optionalTripCount(outer) != 4)
    return false;

  llvm::SmallVector<mlir::scf::ForOp> directInnerLoops;
  for (mlir::Operation &operation : outer.getBody()->without_terminator())
    if (auto inner = llvm::dyn_cast<mlir::scf::ForOp>(&operation))
      directInnerLoops.push_back(inner);
  if (directInnerLoops.size() != 1 ||
      optionalTripCount(directInnerLoops.front()) != 4)
    return false;

  std::size_t stores = 0;
  directInnerLoops.front().getRegion().walk(
      [&](mlir::memref::StoreOp) { ++stores; });
  return stores == 2;
}

std::uint64_t
operationCapacity(const loom::frontend::StructuredProgramCandidate &candidate,
                  const loom::fabric::FinalizedFabricRoot &fabric,
                  llvm::StringRef functionName) {
  mlir::func::FuncOp function =
      candidate.module().lookupSymbol<mlir::func::FuncOp>(functionName);
  if (!function)
    fail("candidate lost function " + functionName.str());
  mlir::scf::ForOp loop;
  function.walk([&](mlir::scf::ForOp candidateLoop) {
    if (!loop)
      loop = candidateLoop;
  });
  if (!loop)
    fail("candidate lost its source loop");
  loom::frontend::FabricCapabilityIndex index(fabric.view());
  std::uint64_t capacity = std::numeric_limits<std::uint64_t>::max();
  loop.getRegion().walk([&](mlir::Operation *operation) {
    if (!dataflow::operationSchemaOf(operation))
      return;
    auto kind = dataflow::classifyCanonicalDataflowActor(operation);
    if (!kind)
      fail("registered actor lost its canonical kind");
    auto count = *kind == dataflow::CanonicalDataflowActorKind::Memory
                     ? take(index.admittingMemoryResourceCount(operation))
                     : take(index.admittingOperationResourceCount(operation));
    capacity = std::min(capacity, count);
  });
  if (capacity == std::numeric_limits<std::uint64_t>::max())
    fail("source loop contains no registered actor");
  return capacity;
}

std::vector<loom::ArtifactRootReference>
generated(const loom::frontend::StructuredProgramCandidate &program,
          const loom::fabric::FinalizedFabricRoot &fabric,
          const loom::ArtifactStore &store, const loom::BlobStore &blobs) {
  loom::ArtifactRootReference programReference =
      take(loom::frontend::publishStructuredProgram(program, store));
  auto config =
      take(loom::dse::projectResolvedStructuredScheduleGeneratorConfigView(
          loom::defaultResolvedConfig()));
  auto inputs = take(loom::dse::bindStructuredScheduleCandidateGeneratorInputs(
      {programReference}, fabric.reference()));
  auto binding = take(
      loom::dse::resolveStructuredScheduleCandidateGeneratorBinding(config));
  auto outcome =
      take(loom::dse::invokeCandidateGenerator(inputs, binding, store, blobs));
  const std::vector<loom::dse::CandidateGeneratorOutputBinding> *bindings =
      nullptr;
  const std::vector<loom::dse::CandidateGeneratorLineageEdge> *lineage =
      nullptr;
  if (auto *completed =
          std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
              &outcome.outcome)) {
    bindings = &completed->outputBindings;
    lineage = &completed->lineageEdges;
  } else if (auto *incomplete =
                 std::get_if<loom::dse::IncompleteCandidateGeneratorResult>(
                     &outcome.outcome);
             incomplete && incomplete->reason ==
                               loom::dse::CandidateGeneratorIncompleteReason::
                                   ProofNotEstablished) {
    bindings = &incomplete->retainedOutputBindings;
    lineage = &incomplete->lineageEdges;
  }
  if (!bindings || bindings->size() != 1 || !lineage)
    fail("schedule generator did not complete one output set");
  for (const loom::dse::CandidateGeneratorLineageEdge &edge : *lineage) {
    if (edge.kind !=
            loom::dse::CandidateGeneratorLineageEdgeKind::CandidateDecision ||
        edge.parents.size() != 2 ||
        !llvm::is_contained(edge.parents, programReference) ||
        !llvm::is_contained(edge.parents, fabric.reference()))
      fail("schedule generator changed its parent lineage");
    take(loom::frontend::adoptStructuredScheduleDecision(edge.ownerPayload));
  }
  return bindings->front().artifacts;
}

void configRoundTripsAndRejectsMalformedBytes() {
  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.schedule.scopeExpansionLimit = 7;
  auto projected =
      take(loom::dse::projectResolvedStructuredScheduleGeneratorConfigView(
          resolved, loom::dse::StructuredScheduleGenerationIntent::Balanced,
          11));
  auto adopted =
      take(loom::dse::adoptResolvedStructuredScheduleGeneratorConfigView(
          loom::dse::resolvedStructuredScheduleGeneratorConfigSchemaBytes(),
          projected.canonicalViewBytes(), projected.digest()));
  if (adopted.scopeExpansionLimit() != 7 ||
      adopted.maximumMaterializationAttempts() != 11 ||
      adopted.canonicalViewBytes() != projected.canonicalViewBytes())
    fail("schedule config did not round-trip exactly");
  requireErrorContains(
      loom::dse::projectResolvedStructuredScheduleGeneratorConfigView(
          resolved, loom::dse::StructuredScheduleGenerationIntent::Balanced,
          std::uint64_t{0}),
      "positive");

  std::vector<std::uint8_t> malformed(projected.canonicalViewBytes().begin(),
                                      projected.canonicalViewBytes().end());
  malformed.push_back(0);
  auto digest = take(loom::computeComponentViewDigest(
      loom::dse::resolvedStructuredScheduleGeneratorConfigSchemaBytes(),
      malformed));
  requireErrorContains(
      loom::dse::adoptResolvedStructuredScheduleGeneratorConfigView(
          loom::dse::resolvedStructuredScheduleGeneratorConfigSchemaBytes(),
          malformed, digest),
      "trailing");
}

void decisionCodecRejectsFactorOne() {
  auto program = parseProgram(R"mlir(
module {
  func.func @loop(%out: memref<4xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %value = arith.constant 1 : i32
      memref.store %value, %out[%i] : memref<4xi32>
    }
    return
  }
}
)mlir");
  auto view = take(program.view());
  std::optional<loom::frontend::StructuredEntityRef> loop;
  for (const loom::frontend::StructuredEntity &entity :
       view.entities(loom::frontend::StructuredEntityKind::Operation))
    if (llvm::isa<mlir::scf::ForOp>(entity.operation)) {
      loop = entity.reference;
      break;
    }
  if (!loop)
    fail("factor-one codec fixture has no loop");
  const loom::frontend::StructuredScheduleDecision decision{
      *loop, loom::frontend::StructuredScheduleDecisionKind::Tile, 1,
      std::nullopt};
  auto encoded = loom::frontend::encodeStructuredScheduleDecision(decision);
  if (encoded)
    fail("schedule decision encoder accepted factor one");
  llvm::consumeError(encoded.takeError());

  const loom::frontend::StructuredScheduleDecision oversizedVector{
      *loop, loom::frontend::StructuredScheduleDecisionKind::Vectorize, 0,
      loom::frontend::StructuredVectorScheduleCoordinate{
          {65},
          loom::frontend::StructuredVectorTailPolicy::Exact,
          260,
          loom::frontend::StructuredVectorAliasPolicy::ProviderProvenNoAlias,
          loom::frontend::StructuredReductionSchedule::None}};
  auto oversized =
      loom::frontend::encodeStructuredScheduleDecision(oversizedVector);
  if (oversized)
    fail("schedule decision encoder accepted an oversized vector factor");
  llvm::consumeError(oversized.takeError());
  const loom::frontend::StructuredScheduleDecision oversizedTile{
      *loop, loom::frontend::StructuredScheduleDecisionKind::Tile,
      loom::frontend::maximumCanonicalStructuredScheduleFactor + 1,
      std::nullopt};
  auto oversizedScalar =
      loom::frontend::encodeStructuredScheduleDecision(oversizedTile);
  if (oversizedScalar)
    fail("schedule decision encoder accepted an oversized scalar factor");
  llvm::consumeError(oversizedScalar.takeError());
}

void invalidInMemoryDecisionFailsClosed() {
  auto program = parseProgram(R"mlir(
module {
  func.func @loop(%out: memref<4xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %value = arith.constant 1 : i32
      memref.store %value, %out[%i] : memref<4xi32>
    }
    return
  }
}
)mlir");
  auto view = take(program.view());
  std::optional<loom::frontend::StructuredEntityRef> loop;
  for (const loom::frontend::StructuredEntity &entity :
       view.entities(loom::frontend::StructuredEntityKind::Operation))
    if (llvm::isa<mlir::scf::ForOp>(entity.operation)) {
      loop = entity.reference;
      break;
    }
  if (!loop)
    fail("invalid-decision fixture has no loop");
  const loom::frontend::StructuredScheduleDecision decision{
      *loop, static_cast<loom::frontend::StructuredScheduleDecisionKind>(99), 0,
      std::nullopt};
  auto encoded = loom::frontend::encodeStructuredScheduleDecision(decision);
  if (encoded)
    fail("schedule encoder accepted an unknown in-memory decision kind");
  llvm::consumeError(encoded.takeError());
  auto materialized =
      loom::frontend::materializeStructuredScheduleDecision(program, decision);
  if (materialized)
    fail("schedule materializer accepted an unknown in-memory decision kind");
  llvm::consumeError(materialized.takeError());
}

void lineageCodecRejectsAnOutOfRangeLoop() {
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-schedule-lineage-context", directory);
  if (error)
    fail("cannot create ArtifactStore directory: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));
  const loom::fabric::FinalizedFabricRoot &fabric = design.roots().front();
  auto program = parseProgram(R"mlir(
module {
  func.func @loop(%out: memref<4xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %value = arith.constant 1 : i32
      memref.store %value, %out[%i] : memref<4xi32>
    }
    return
  }
}
)mlir");
  auto parentReference =
      take(loom::frontend::publishStructuredProgram(program, store));
  const loom::frontend::StructuredScheduleDecision decision{
      {program.identity(), loom::frontend::StructuredEntityKind::Operation,
       999999},
      loom::frontend::StructuredScheduleDecisionKind::Tile,
      2,
      std::nullopt};
  auto encoded =
      take(loom::frontend::encodeStructuredScheduleDecision(decision));
  const auto *contract =
      loom::dse::structuredScheduleCandidateGeneratorDescriptor()
          .ownerLineagePayload;
  if (!contract)
    fail("schedule generator has no owner lineage contract");
  llvm::Error validation = contract->validateCanonical(
      encoded, parentReference, {parentReference, fabric.reference()}, store);
  if (!validation)
    fail("schedule lineage accepted an out-of-range parent-local loop");
  llvm::consumeError(std::move(validation));
  error = llvm::sys::fs::remove_directories(directory);
  if (error)
    fail("cannot remove ArtifactStore directory: " + error.message());
}

void lineageRejectsAValidForeignChild() {
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-schedule-lineage-child", directory);
  if (error)
    fail("cannot create ArtifactStore directory: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));
  const loom::fabric::FinalizedFabricRoot &fabric = design.roots().front();
  auto parent = parseProgram(R"mlir(
module {
  func.func @loop(%out: memref<8xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    scf.for %i = %c0 to %c8 step %c1 {
      %value = arith.index_cast %i : index to i32
      memref.store %value, %out[%i] : memref<8xi32>
    }
    return
  }
}
)mlir");
  auto parentReference =
      take(loom::frontend::publishStructuredProgram(parent, store));
  const auto loop = structuredLoopReference(parent, "loop");
  const loom::frontend::StructuredScheduleDecision declared{
      loop, loom::frontend::StructuredScheduleDecisionKind::Tile, 2,
      std::nullopt};
  const loom::frontend::StructuredScheduleDecision foreignDecision{
      loop, loom::frontend::StructuredScheduleDecisionKind::Tile, 4,
      std::nullopt};
  auto foreignChild =
      take(loom::frontend::materializeStructuredScheduleDecision(
          parent, foreignDecision));
  auto foreignReference = take(loom::frontend::publishStructuredProgram(
      foreignChild.structuredProgram, store));
  auto encoded =
      take(loom::frontend::encodeStructuredScheduleDecision(declared));
  const auto *contract =
      loom::dse::structuredScheduleCandidateGeneratorDescriptor()
          .ownerLineagePayload;
  if (!contract)
    fail("schedule generator has no owner lineage contract");
  llvm::Error validation = contract->validateCanonical(
      encoded, foreignReference, {parentReference, fabric.reference()}, store);
  if (!validation)
    fail("schedule lineage accepted another decision's valid child");
  llvm::consumeError(std::move(validation));

  const loom::frontend::StructuredScheduleDecision forgedDecision{
      loop, loom::frontend::StructuredScheduleDecisionKind::Tile, 3,
      std::nullopt};
  auto forgedChild = take(loom::frontend::materializeStructuredScheduleDecision(
      parent, forgedDecision));
  auto forgedReference = take(loom::frontend::publishStructuredProgram(
      forgedChild.structuredProgram, store));
  auto forgedPayload =
      take(loom::frontend::encodeStructuredScheduleDecision(forgedDecision));
  llvm::Error forgedValidation =
      contract->validateCanonical(forgedPayload, forgedReference,
                                  {parentReference, fabric.reference()}, store);
  if (!forgedValidation)
    fail("schedule lineage accepted a replayable out-of-domain decision");
  llvm::consumeError(std::move(forgedValidation));

  error = llvm::sys::fs::remove_directories(directory);
  if (error)
    fail("cannot remove ArtifactStore directory: " + error.message());
}

void transformationsAreTypedCapacityBoundAndDependenceChecked() {
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-schedule-generator", directory);
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
  const loom::fabric::FinalizedFabricRoot &fabric = design.roots().front();
  loom::adg::BuiltinTargetScale constrainedScale =
      loom::adg::builtinSmallTarget.scale;
  constrainedScale.accCoreCount = 1;
  auto constrainedDesign =
      take(loom::adg::buildBuiltinTarget(store, constrainedScale));
  const loom::fabric::FinalizedFabricRoot &constrainedFabric =
      constrainedDesign.roots().front();

  auto scoped = parseProgram(R"mlir(
#layout = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
module attributes {dlti.dl_spec = #layout} {
  func.func @earlier(%out: memref<4xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %value = arith.index_cast %i : index to i32
      memref.store %value, %out[%i] : memref<4xi32>
    }
    return
  }
  func.func @selected(%out: memref<8xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    scf.for %i = %c0 to %c8 step %c1 {
      %value = arith.index_cast %i : index to i32
      memref.store %value, %out[%i] : memref<8xi32>
    }
    return
  }
}
)mlir");
  auto scopedView = take(scoped.view());
  std::optional<loom::frontend::StructuredEntityRef> selectedFunction;
  for (const loom::frontend::StructuredEntity &entity :
       scopedView.entities(loom::frontend::StructuredEntityKind::Operation)) {
    auto function =
        llvm::dyn_cast_or_null<mlir::func::FuncOp>(entity.operation);
    if (function && function.getSymName() == "selected") {
      selectedFunction = entity.reference;
      break;
    }
  }
  if (!selectedFunction)
    fail("scoped schedule fixture lost its selected function");
  auto scopedDomain = take(loom::frontend::enumerateStructuredScheduleDecisions(
      scoped, fabric, 1, *selectedFunction));
  if (scopedDomain.inspectedLoopScopes != 1 ||
      !llvm::any_of(scopedDomain.proposals, [](const auto &proposal) {
        return proposal.decision().kind ==
               loom::frontend::StructuredScheduleDecisionKind::Parallelize;
      }))
    fail("loops outside an exact schedule scope consumed its bound");

  auto identityNoOp = parseProgram(R"mlir(
module {
  func.func @kernel() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    scf.for %i = %c0 to %c3 step %c1 {
      scf.for %j = %c0 to %c3 step %c1 {
      }
    }
    return
  }
}
)mlir");
  auto identityDomain =
      take(loom::frontend::enumerateStructuredScheduleDecisions(identityNoOp,
                                                                fabric, 2));
  auto identityProposal =
      llvm::find_if(identityDomain.proposals, [](const auto &proposal) {
        return proposal.decision().kind ==
               loom::frontend::StructuredScheduleDecisionKind::Interchange;
      });
  if (identityProposal == identityDomain.proposals.end())
    fail("symmetric perfect nest lost its interchange proposal");
  auto identityChild =
      take(loom::frontend::materializeStructuredScheduleProposal(
          identityNoOp, *identityProposal, fabric));
  if (identityChild.structuredProgram.identity() != identityNoOp.identity())
    fail("symmetric interchange did not produce its exact identity child");
  auto identityReference =
      take(loom::frontend::publishStructuredProgram(identityNoOp, store));
  auto identityInputs =
      take(loom::dse::bindStructuredScheduleCandidateGeneratorInputs(
          {identityReference}, fabric.reference()));
  auto identityConfig =
      take(loom::dse::projectResolvedStructuredScheduleGeneratorConfigView(
          loom::defaultResolvedConfig(),
          loom::dse::StructuredScheduleGenerationIntent::
              ForbidLogicalThreadDomain,
          1));
  auto identityBinding =
      take(loom::dse::resolveStructuredScheduleCandidateGeneratorBinding(
          identityConfig));
  auto identityResult = take(loom::dse::invokeCandidateGenerator(
      identityInputs, identityBinding, store, blobs));
  const auto *identityCompleted =
      std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
          &identityResult.outcome);
  if (!identityCompleted || identityCompleted->outputBindings.size() != 1 ||
      identityCompleted->outputBindings.front().artifacts !=
          std::vector<loom::ArtifactRootReference>{identityReference} ||
      !identityCompleted->lineageEdges.empty() ||
      identityResult.workSummary.size() != 5 ||
      identityResult.workSummary[1].planned != 1 ||
      identityResult.workSummary[1].consumed != 1 ||
      identityResult.workSummary[3].planned != 1 ||
      identityResult.workSummary[3].consumed != 1)
    fail("identity schedule materialization produced a self edge or output");

  auto capacityBound = parseProgram(R"mlir(
#layout = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
module attributes {dlti.dl_spec = #layout} {
  func.func @kernel(%out: memref<8x4xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c4 step %c1 {
        %row = arith.muli %i, %c4 : index
        %row1 = arith.muli %row, %c4 : index
        %row2 = arith.muli %row1, %c4 : index
        %row3 = arith.muli %row2, %c4 : index
        %row4 = arith.muli %row3, %c4 : index
        %index = arith.addi %row4, %j : index
        %value = arith.index_cast %index : index to i32
        memref.store %value, %out[%i, %j] : memref<8x4xi32>
      }
    }
    return
  }
}
)mlir");
  auto exactFabricDomain =
      take(loom::frontend::enumerateStructuredScheduleDecisions(capacityBound,
                                                                fabric, 8));
  auto constrainedFabricDomain =
      take(loom::frontend::enumerateStructuredScheduleDecisions(
          capacityBound, constrainedFabric, 8));
  for (loom::frontend::StructuredScheduleDecisionKind kind :
       {loom::frontend::StructuredScheduleDecisionKind::Unroll,
        loom::frontend::StructuredScheduleDecisionKind::UnrollAndJam}) {
    auto proposal =
        llvm::find_if(exactFabricDomain.proposals, [&](const auto &candidate) {
          return candidate.decision().kind == kind &&
                 candidate.decision().factor == 2;
        });
    if (proposal == exactFabricDomain.proposals.end())
      fail("larger Fabric lost a factor-two scalar proposal");
    if (llvm::any_of(constrainedFabricDomain.proposals,
                     [&](const auto &candidate) {
                       return candidate.decision().kind == kind &&
                              candidate.decision().factor == 2;
                     }))
      fail("constrained Fabric admitted an over-capacity scalar proposal");
    auto crossFabric = loom::frontend::materializeStructuredScheduleProposal(
        capacityBound, *proposal, constrainedFabric);
    if (crossFabric)
      fail("scalar proposal escaped its exact enumerating Fabric");
    llvm::consumeError(crossFabric.takeError());
  }

  auto safe = parseProgram(R"mlir(
#layout = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
module attributes {dlti.dl_spec = #layout} {
  func.func @kernel(%out: memref<8x4xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c4 step %c1 {
        %row = arith.muli %i, %c4 : index
        %index = arith.addi %row, %j : index
        %value = arith.index_cast %index : index to i32
        memref.store %value, %out[%i, %j] : memref<8x4xi32>
      }
    }
    return
  }
}
)mlir");
  std::vector<loom::ArtifactRootReference> safeOutputs =
      generated(safe, fabric, store, blobs);
  if (safeOutputs.size() <= 1)
    fail("schedule generator produced no transformed candidate");
  bool sawInterchange = false;
  for (const loom::ArtifactRootReference &reference : safeOutputs) {
    auto candidate =
        take(loom::frontend::importStructuredProgram(reference, store));
    sawInterchange |= outerInnerTrips(candidate, "kernel") ==
                      std::optional<std::pair<std::uint64_t, std::uint64_t>>(
                          std::in_place, 4, 8);
  }
  if (!sawInterchange)
    fail("proven-independent nested loops produced no interchange child");

  bool sawUnrollAndJam = false;
  for (const loom::ArtifactRootReference &reference : safeOutputs) {
    auto candidate =
        take(loom::frontend::importStructuredProgram(reference, store));
    sawUnrollAndJam |= hasUnrollAndJamShape(candidate, "kernel");
  }
  if (!sawUnrollAndJam)
    fail("proven-independent nested loops produced no unroll-and-jam child");

  bool sawParallel = false;
  bool sawRankTwoParallel = false;
  for (const loom::ArtifactRootReference &reference : safeOutputs) {
    auto candidate =
        take(loom::frontend::importStructuredProgram(reference, store));
    sawParallel |= forallCount(candidate, "kernel") != 0;
    sawRankTwoParallel |= maximumForallRank(candidate, "kernel") >= 2;
  }
  if (!sawParallel)
    fail("proven-independent loop produced no parallel child");
  if (!sawRankTwoParallel)
    fail("rectangular independent nest produced no rank-two parallel child");

  auto dependent = parseProgram(R"mlir(
#layout = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
module attributes {dlti.dl_spec = #layout} {
  func.func @kernel(%state: memref<9xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c4 step %c1 {
        %next = arith.addi %i, %c1 : index
        %value = memref.load %state[%i] : memref<9xi32>
        memref.store %value, %state[%next] : memref<9xi32>
      }
    }
    return
  }
}
)mlir");
  for (const loom::ArtifactRootReference &reference :
       generated(dependent, fabric, store, blobs)) {
    auto candidate =
        take(loom::frontend::importStructuredProgram(reference, store));
    if (outerInnerTrips(candidate, "kernel") ==
        std::optional<std::pair<std::uint64_t, std::uint64_t>>(std::in_place, 4,
                                                               8))
      fail("loop-carried dependence produced an interchange child");
    if (hasUnrollAndJamShape(candidate, "kernel"))
      fail("loop-carried dependence produced an unroll-and-jam child");
    if (forallCount(candidate, "kernel") != 0)
      fail("loop-carried dependence produced a parallel child");
  }

  auto unroll = parseProgram(R"mlir(
#layout = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
module attributes {dlti.dl_spec = #layout} {
  func.func @kernel(%out: memref<1024xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    scf.for %i = %c0 to %c1024 step %c1 {
      %value = arith.addi %i, %i : index
      %stored = arith.index_cast %value : index to i32
      memref.store %stored, %out[%i] : memref<1024xi32>
    }
    return
  }
}
)mlir");
  auto unrollDomain = take(
      loom::frontend::enumerateStructuredScheduleDecisions(unroll, fabric, 1));
  if (unrollDomain.inspectedDecisionCoordinates >
      4 * loom::frontend::maximumCanonicalStructuredScheduleFactor)
    fail("one loop exceeded the canonical bounded schedule domain");
  for (const auto &proposal : unrollDomain.proposals)
    if (proposal.decision().factor >
        loom::frontend::maximumCanonicalStructuredScheduleFactor)
      fail("static trip count escaped the canonical schedule-factor bound");
  const std::uint64_t admittedCapacity =
      operationCapacity(unroll, fabric, "kernel");
  std::size_t maximumReplication = 0;
  for (const loom::ArtifactRootReference &reference :
       generated(unroll, fabric, store, blobs)) {
    auto candidate =
        take(loom::frontend::importStructuredProgram(reference, store));
    maximumReplication =
        std::max(maximumReplication, storeCount(candidate, "kernel"));
  }
  if (maximumReplication < 2)
    fail("Fabric-admitted loop produced no unroll child");
  if (maximumReplication > admittedCapacity)
    fail("unroll replication exceeded exact aggregate Fabric capacity");

  auto unrollReference =
      take(loom::frontend::publishStructuredProgram(unroll, store));
  auto boundedConfig =
      take(loom::dse::projectResolvedStructuredScheduleGeneratorConfigView(
          loom::defaultResolvedConfig()));
  auto boundedInputs =
      take(loom::dse::bindStructuredScheduleCandidateGeneratorInputs(
          {unrollReference}, fabric.reference()));
  auto boundedBinding =
      take(loom::dse::resolveStructuredScheduleCandidateGeneratorBinding(
          boundedConfig));
  const std::array<loom::dse::CandidateGeneratorOutputDemand, 1> demands = {{
      {loom::dse::CandidateGeneratorOutputSlotRef(0), 2},
  }};
  const loom::dse::CandidateGeneratorInvocationView boundedInvocation(
      loom::ExecutionControlView{}, demands);
  auto bounded = take(loom::dse::invokeCandidateGenerator(
      boundedInputs, boundedBinding, store, blobs, boundedInvocation));
  const auto *incomplete =
      std::get_if<loom::dse::IncompleteCandidateGeneratorResult>(
          &bounded.outcome);
  if (!incomplete ||
      incomplete->reason !=
          loom::dse::CandidateGeneratorIncompleteReason::SemanticLimitReached ||
      incomplete->retainedOutputBindings.size() != 1 ||
      incomplete->retainedOutputBindings.front().artifacts.size() != 2 ||
      incomplete->lineageEdges.size() != 1 || bounded.workSummary.size() != 5 ||
      bounded.workSummary[0].planned != 1 ||
      bounded.workSummary[0].consumed != 1 ||
      bounded.workSummary[1].planned != 1 ||
      bounded.workSummary[1].consumed != 1 ||
      bounded.workSummary[2].planned == 0 ||
      bounded.workSummary[2].consumed != bounded.workSummary[2].planned ||
      bounded.workSummary[3].planned <= bounded.workSummary[3].consumed ||
      bounded.workSummary[3].consumed != 1)
    fail("bounded schedule generation lost exact work accounting");

  auto unresolvedSpecialMath = parseProgram(R"mlir(
#layout = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
module attributes {dlti.dl_spec = #layout} {
  func.func @kernel(%out: memref<4xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %integer = arith.index_cast %i : index to i32
      %value = arith.sitofp %integer : i32 to f32
      %exponential = math.exp %value fastmath<afn> : f32
      memref.store %exponential, %out[%i] : memref<4xf32>
    }
    return
  }
}
)mlir");
  if (generated(unresolvedSpecialMath, fabric, store, blobs).empty())
    fail("unresolved special math disappeared before its terminal owner");

  llvm::sys::fs::remove_directories(directory);
}

void exactScopVectorCoordinateIsCanonicalAndVerified() {
  llvm::SmallString<128> directory;
  std::error_code error =
      llvm::sys::fs::createUniqueDirectory("loom-schedule-vector", directory);
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
  const loom::fabric::FinalizedFabricRoot &fabric = design.roots().front();

  auto parent = parseProgram(R"mlir(
#layout = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
module attributes {dlti.dl_spec = #layout} {
  func.func @kernel(%lhs: memref<16xi32>, %rhs: memref<16xi32>) {
    %lhs_distinct, %rhs_distinct = memref.distinct_objects %lhs, %rhs
        : memref<16xi32>, memref<16xi32>
    %lhs_aligned = memref.assume_alignment %lhs_distinct, 64
        : memref<16xi32>
    %rhs_aligned = memref.assume_alignment %rhs_distinct, 64
        : memref<16xi32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    scf.for %i = %c0 to %c16 step %c1 {
      %left = memref.load %lhs_aligned[%i] : memref<16xi32>
      %right = memref.load %rhs_aligned[%i] : memref<16xi32>
      %sum = arith.addi %left, %right : i32
      memref.store %sum, %rhs_aligned[%i] : memref<16xi32>
    }
    return
  }
}
)mlir");
  const auto loop = structuredLoopReference(parent, "kernel");
  auto analysis =
      take(loom::frontend::analyzeExactStructuredScop(parent, loop));
  const auto *scop =
      std::get_if<loom::frontend::ExactStructuredScopView>(&analysis);
  if (!scop || scop->statementCount != 4 || scop->accesses.size() != 3 ||
      scop->minimumAlignmentBytes != 64 || scop->maximumElementBytes != 4 ||
      scop->constantTripCount != 16 ||
      scop->polyhedralSchedule.provider !=
          loom::frontend::StructuredPolyhedralProviderKind::PinnedPollyIsl ||
      scop->polyhedralSchedule.dependenceCount != 4 ||
      scop->polyhedralSchedule.scheduleMapCount() != 4 ||
      scop->polyhedralSchedule.scheduleBandCount == 0 ||
      scop->polyhedralSchedule.scheduleDimensionCount == 0 ||
      scop->reductionSchedule !=
          loom::frontend::StructuredReductionSchedule::None)
    fail("exact vector SCoP analysis lost a provider-proven fact");
  for (auto [ordinal, schedule] :
       llvm::enumerate(scop->polyhedralSchedule.statementSchedules)) {
    if (schedule.statementOrdinal != ordinal || schedule.pieces.empty())
      fail("typed provider schedule lost a statement relation");
    for (const auto &piece : schedule.pieces) {
      if (piece.sourceDimensionCount != 1 || piece.parameterCount != 0 ||
          piece.scheduleDimensionCount == 0 || piece.constraints.empty())
        fail("typed provider schedule has an invalid static piece");
      const std::uint64_t rowWidth =
          piece.sourceDimensionCount + piece.scheduleDimensionCount +
          piece.parameterCount + piece.divisions.size() + 1;
      for (const auto &division : piece.divisions)
        if (division.denominator == 0 || division.numerator.size() != rowWidth)
          fail("typed provider schedule has an invalid division");
      for (const auto &constraint : piece.constraints)
        if (constraint.coefficients.size() != rowWidth)
          fail("typed provider schedule has an invalid constraint row");
    }
  }

  auto domain = take(
      loom::frontend::enumerateStructuredScheduleDecisions(parent, fabric, 8));
  auto decision = firstVectorDecision(domain);
  if (!decision || !decision->vector || decision->factor != 0 ||
      decision->vector->shape != std::vector<std::uint64_t>{2} ||
      decision->vector->tailPolicy !=
          loom::frontend::StructuredVectorTailPolicy::Exact ||
      decision->vector->requiredAlignmentBytes != 8 ||
      decision->vector->reductionSchedule !=
          loom::frontend::StructuredReductionSchedule::None)
    fail("exact vector SCoP produced no canonical hardware-aware coordinate");

  auto encoded =
      take(loom::frontend::encodeStructuredScheduleDecision(*decision));
  auto adopted = take(loom::frontend::adoptStructuredScheduleDecision(encoded));
  if (!(adopted == *decision))
    fail("vector schedule coordinate did not round-trip canonically");

  auto child = take(
      loom::frontend::materializeStructuredScheduleDecision(parent, *decision));
  if (llvm::Error verification =
          loom::frontend::verifyStructuredVectorScheduleMaterialization(
              *scop, *decision->vector, child.structuredProgram.module()))
    fail("independent vector verifier rejected the materialized child: " +
         llvm::toString(std::move(verification)));
  std::size_t vectorAdds = 0;
  std::size_t vectorReads = 0;
  child.structuredProgram.module().walk([&](mlir::Operation *operation) {
    if (auto add = llvm::dyn_cast<mlir::arith::AddIOp>(operation))
      vectorAdds += llvm::isa<mlir::VectorType>(add.getType()) ? 1 : 0;
    vectorReads += llvm::isa<mlir::vector::TransferReadOp>(operation) ? 1 : 0;
  });
  if (vectorAdds != 1 || vectorReads != 2)
    fail("Affine provider did not materialize the selected vector shape");

  mlir::OwningOpRef<mlir::ModuleOp> mutated(
      llvm::cast<mlir::ModuleOp>(child.structuredProgram.module()->clone()));
  bool changed = false;
  mutated->walk([&](mlir::scf::ForOp candidate) {
    if (!changed && mlir::getConstantIntValue(candidate.getStep()) == 2) {
      mlir::OpBuilder builder(candidate);
      mlir::Value replacement =
          mlir::arith::ConstantIndexOp::create(builder, candidate.getLoc(), 4);
      candidate.getStepMutable().assign(replacement);
      changed = true;
    }
  });
  llvm::Error mutation =
      loom::frontend::verifyStructuredVectorScheduleMaterialization(
          *scop, *decision->vector, mutated.get());
  if (!changed || !mutation)
    fail("independent vector verifier accepted a changed schedule coordinate");
  llvm::consumeError(std::move(mutation));

  mlir::OwningOpRef<mlir::ModuleOp> changedBounds(
      llvm::cast<mlir::ModuleOp>(child.structuredProgram.module()->clone()));
  bool changedUpperBound = false;
  changedBounds->walk([&](mlir::scf::ForOp candidate) {
    if (!changedUpperBound &&
        mlir::getConstantIntValue(candidate.getStep()) == 2) {
      mlir::OpBuilder builder(candidate);
      mlir::Value replacement =
          mlir::arith::ConstantIndexOp::create(builder, candidate.getLoc(), 14);
      candidate.getUpperBoundMutable().assign(replacement);
      changedUpperBound = true;
    }
  });
  llvm::Error boundMutation =
      loom::frontend::verifyStructuredVectorScheduleMaterialization(
          *scop, *decision->vector, changedBounds.get());
  if (!changedUpperBound || !boundMutation)
    fail("independent vector verifier accepted a changed iteration domain");
  llvm::consumeError(std::move(boundMutation));

  bool productionVectorChild = false;
  for (const loom::ArtifactRootReference &reference :
       generated(parent, fabric, store, blobs)) {
    auto candidate =
        take(loom::frontend::importStructuredProgram(reference, store));
    candidate.module().walk(
        [&](mlir::vector::TransferReadOp) { productionVectorChild = true; });
  }
  if (!productionVectorChild)
    fail("production schedule provider published no vectorized child");

  auto mixedParent = parseProgram(R"mlir(
module {
  func.func @kernel(%input: memref<16xi32>, %output: memref<16xi32>) {
    %input_distinct, %output_distinct = memref.distinct_objects %input, %output
        : memref<16xi32>, memref<16xi32>
    %input_aligned = memref.assume_alignment %input_distinct, 64
        : memref<16xi32>
    %output_aligned = memref.assume_alignment %output_distinct, 64
        : memref<16xi32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    scf.for %i = %c0 to %c16 step %c1 {
      %unused = memref.load %input_aligned[%i] : memref<16xi32>
    }
    scf.for %i = %c0 to %c16 step %c1 {
      %value = memref.load %input_aligned[%i] : memref<16xi32>
      memref.store %value, %output_aligned[%i] : memref<16xi32>
    }
    return
  }
}
)mlir");
  auto mixedDomain = take(loom::frontend::enumerateStructuredScheduleDecisions(
      mixedParent, fabric, 2));
  if (!firstVectorDecision(mixedDomain) ||
      llvm::none_of(mixedDomain.refusals, [](const auto &refusal) {
        return refusal.kind == loom::frontend::StructuredScopRefusalKind::
                                   ProviderMaterializationRejected;
      }))
    fail("one refused SCoP suppressed an unrelated exact SCoP");

  llvm::sys::fs::remove_directories(directory);
}

void symbolicScopUsesExactParameterIdentity() {
  auto parent = parseProgram(R"mlir(
#layout = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
module attributes {dlti.dl_spec = #layout} {
  func.func @kernel(%input: memref<?xi32>, %output: memref<?xi32>, %n: index) {
    %input_distinct, %output_distinct =
        memref.distinct_objects %input, %output
        : memref<?xi32>, memref<?xi32>
    %input_aligned = memref.assume_alignment %input_distinct, 64
        : memref<?xi32>
    %output_aligned = memref.assume_alignment %output_distinct, 64
        : memref<?xi32>
    affine.for %i = 0 to %n {
      %value = affine.load %input_aligned[%i] : memref<?xi32>
      %doubled = arith.addi %value, %value : i32
      affine.store %doubled, %output_aligned[%i] : memref<?xi32>
    }
    return
  }
}
)mlir");
  const auto loop = structuredLoopReference(parent, "kernel");
  auto analysis =
      take(loom::frontend::analyzeExactStructuredScop(parent, loop));
  const auto *scop =
      std::get_if<loom::frontend::ExactStructuredScopView>(&analysis);
  if (!scop || scop->statementCount != 3 || scop->parameterCount != 1 ||
      scop->polyhedralSchedule.parameterCount != 1 ||
      scop->polyhedralSchedule.scheduleMapCount() != 3 ||
      scop->constantTripCount)
    fail("symbolic SCoP lost its exact provider parameter domain");
  for (const auto &schedule : scop->polyhedralSchedule.statementSchedules)
    for (const auto &piece : schedule.pieces)
      if (piece.parameterCount != 1)
        fail("typed symbolic schedule lost its exact parameter space");
}

void exactScopRefusalsAreLocalAndTyped() {
  auto unregisteredSupport = parseProgram(R"mlir(
#identity = affine_map<(d0) -> (d0)>
module {
  func.func @kernel(%input: memref<8xi32>) {
    %aligned = memref.assume_alignment %input, 32 : memref<8xi32>
    affine.for %i = 0 to 8 {
      %unused = affine.apply #identity(%i)
      %value = affine.load %aligned[%i] : memref<8xi32>
    }
    return
  }
}
)mlir");
  auto supportOutcome = take(loom::frontend::analyzeExactStructuredScop(
      unregisteredSupport,
      structuredLoopReference(unregisteredSupport, "kernel")));
  const auto *supportRefusal =
      std::get_if<loom::frontend::StructuredScopRefusal>(&supportOutcome);
  if (!supportRefusal ||
      supportRefusal->kind !=
          loom::frontend::StructuredScopRefusalKind::UnsupportedOperation)
    fail("unregistered affine support lost its typed local refusal");

  auto localPresburgerDomain = parseProgram(R"mlir(
#half = affine_map<(d0) -> (d0 floordiv 2)>
module {
  func.func @kernel(%input: memref<?xi32>, %n: index) {
    %aligned = memref.assume_alignment %input, 32 : memref<?xi32>
    affine.for %i = 0 to #half(%n) {
      %value = affine.load %aligned[%i] : memref<?xi32>
    }
    return
  }
}
)mlir");
  auto localDomainOutcome = take(loom::frontend::analyzeExactStructuredScop(
      localPresburgerDomain,
      structuredLoopReference(localPresburgerDomain, "kernel")));
  const auto *localDomainRefusal =
      std::get_if<loom::frontend::StructuredScopRefusal>(&localDomainOutcome);
  if (!localDomainRefusal ||
      localDomainRefusal->kind !=
          loom::frontend::StructuredScopRefusalKind::ProviderDomainNotAdmitted)
    fail("local Presburger domain lost its typed provider refusal");

  auto unresolvedAlias = parseProgram(R"mlir(
module {
  func.func @kernel(%lhs: memref<8xi32>, %rhs: memref<8xi32>) {
    %lhs_aligned = memref.assume_alignment %lhs, 32 : memref<8xi32>
    %rhs_aligned = memref.assume_alignment %rhs, 32 : memref<8xi32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    scf.for %i = %c0 to %c8 step %c1 {
      %value = memref.load %lhs_aligned[%i] : memref<8xi32>
      memref.store %value, %rhs_aligned[%i] : memref<8xi32>
    }
    return
  }
}
)mlir");
  auto aliasOutcome = take(loom::frontend::analyzeExactStructuredScop(
      unresolvedAlias, structuredLoopReference(unresolvedAlias, "kernel")));
  const auto *aliasRefusal =
      std::get_if<loom::frontend::StructuredScopRefusal>(&aliasOutcome);
  if (!aliasRefusal ||
      aliasRefusal->kind !=
          loom::frontend::StructuredScopRefusalKind::AliasProofNotEstablished)
    fail("unresolved memref alias did not retain its typed local refusal");

  auto strictReduction = parseProgram(R"mlir(
module {
  func.func @kernel(%input: memref<15xf32>) -> f32 {
    %aligned = memref.assume_alignment %input, 64 : memref<15xf32>
    %zero = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c15 = arith.constant 15 : index
    %sum = scf.for %i = %c0 to %c15 step %c1
        iter_args(%acc = %zero) -> f32 {
      %value = memref.load %aligned[%i] : memref<15xf32>
      %next = arith.addf %acc, %value : f32
      scf.yield %next : f32
    }
    return %sum : f32
  }
}
)mlir");
  auto strictOutcome = take(loom::frontend::analyzeExactStructuredScop(
      strictReduction, structuredLoopReference(strictReduction, "kernel")));
  const auto *strictRefusal =
      std::get_if<loom::frontend::StructuredScopRefusal>(&strictOutcome);
  if (!strictRefusal ||
      strictRefusal->kind !=
          loom::frontend::StructuredScopRefusalKind::StrictFloatingReduction)
    fail("strict floating reduction did not retain its typed local refusal");

  auto nonNeutralReduction = parseProgram(R"mlir(
module {
  func.func @kernel(%input: memref<15xf32>) -> f32 {
    %aligned = memref.assume_alignment %input, 64 : memref<15xf32>
    %one = arith.constant 1.0 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c15 = arith.constant 15 : index
    %sum = scf.for %i = %c0 to %c15 step %c1
        iter_args(%acc = %one) -> f32 {
      %value = memref.load %aligned[%i] : memref<15xf32>
      %next = arith.addf %acc, %value fastmath<reassoc> : f32
      scf.yield %next : f32
    }
    return %sum : f32
  }
}
)mlir");
  auto nonNeutralOutcome = take(loom::frontend::analyzeExactStructuredScop(
      nonNeutralReduction,
      structuredLoopReference(nonNeutralReduction, "kernel")));
  const auto *nonNeutralRefusal =
      std::get_if<loom::frontend::StructuredScopRefusal>(&nonNeutralOutcome);
  if (!nonNeutralRefusal ||
      nonNeutralRefusal->kind !=
          loom::frontend::StructuredScopRefusalKind::UnsupportedReduction)
    fail("non-neutral reduction did not retain its typed local refusal");

  auto offsetLayout = parseProgram(R"mlir(
module {
  func.func @kernel(
      %input: memref<8xi32, strided<[1], offset: 1>>) {
    %aligned = memref.assume_alignment %input, 32
        : memref<8xi32, strided<[1], offset: 1>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    scf.for %i = %c0 to %c8 step %c1 {
      %value = memref.load %aligned[%i]
          : memref<8xi32, strided<[1], offset: 1>>
    }
    return
  }
}
)mlir");
  auto offsetOutcome = take(loom::frontend::analyzeExactStructuredScop(
      offsetLayout, structuredLoopReference(offsetLayout, "kernel")));
  const auto *offsetRefusal =
      std::get_if<loom::frontend::StructuredScopRefusal>(&offsetOutcome);
  if (!offsetRefusal ||
      offsetRefusal->kind !=
          loom::frontend::StructuredScopRefusalKind::UnsupportedPhysicalOffset)
    fail("nonzero layout offset did not retain its typed local refusal");
}

void reassociatedReductionOwnsItsMaskedTail() {
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-schedule-reduction", directory);
  if (error)
    fail("cannot create ArtifactStore directory: " + error.message());
  loom::ArtifactStore store(directory);
  llvm::SmallString<128> blobPath(directory);
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  auto design = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Coverage));
  const loom::fabric::FinalizedFabricRoot &fabric = design.roots().front();
  auto parent = parseProgram(R"mlir(
module {
  func.func @kernel(%input: memref<15xf32>) -> f32 {
    %aligned = memref.assume_alignment %input, 64 : memref<15xf32>
    %zero = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c15 = arith.constant 15 : index
    %sum = scf.for %i = %c0 to %c15 step %c1
        iter_args(%acc = %zero) -> f32 {
      %value = memref.load %aligned[%i] : memref<15xf32>
      %next = arith.addf %acc, %value fastmath<reassoc> : f32
      scf.yield %next : f32
    }
    return %sum : f32
  }
}
)mlir");
  auto domain = take(
      loom::frontend::enumerateStructuredScheduleDecisions(parent, fabric, 4));
  auto maskedProposal =
      llvm::find_if(domain.proposals, [](const auto &proposal) {
        const auto &decision = proposal.decision();
        return decision.kind ==
                   loom::frontend::StructuredScheduleDecisionKind::Vectorize &&
               decision.vector &&
               decision.vector->shape == std::vector<std::uint64_t>{2} &&
               decision.vector->tailPolicy ==
                   loom::frontend::StructuredVectorTailPolicy::ReductionMask;
      });
  if (maskedProposal == domain.proposals.end())
    fail("masked-tail coordinate was not represented as a bounded proposal");
  auto refused = loom::frontend::materializeStructuredScheduleProposal(
      parent, *maskedProposal, fabric);
  if (refused)
    fail("Fabric admitted an unavailable masked-tail implementation");
  bool sawTypedRefusal = false;
  llvm::Error unhandled = llvm::handleErrors(
      refused.takeError(),
      [&](const loom::frontend::StructuredScheduleProposalRefusal &error) {
        sawTypedRefusal = error.kind() ==
                          loom::frontend::StructuredScopRefusalKind::
                              VectorLoweringUnavailable;
      });
  if (unhandled)
    fail("masked-tail proposal returned an untyped error: " +
         llvm::toString(std::move(unhandled)));
  if (!sawTypedRefusal)
    fail("masked-tail lowering rejection lost its typed local refusal");

  loom::adg::BuiltinTargetScale scalarOnlyScale =
      loom::adg::builtinSmallTarget.scale;
  scalarOnlyScale.spatialFuOccurrences.vectorCompute = 0;
  scalarOnlyScale.spatialFuOccurrences.vectorAdapter = 0;
  scalarOnlyScale.spatialFuOccurrences.vectorStructural = 0;
  scalarOnlyScale.temporalFuOccurrences.vectorCompute = 0;
  scalarOnlyScale.temporalFuOccurrences.vectorAdapter = 0;
  scalarOnlyScale.temporalFuOccurrences.vectorStructural = 0;
  auto attemptDesign =
      take(loom::adg::buildBuiltinTarget(store, scalarOnlyScale));
  const loom::fabric::FinalizedFabricRoot &attemptFabric =
      attemptDesign.roots().front();
  auto attemptParent = parseProgram(R"mlir(
#layout = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
module attributes {dlti.dl_spec = #layout} {
  func.func @kernel(%lhs: memref<16xi32>, %rhs: memref<16xi32>) {
    %lhs_distinct, %rhs_distinct = memref.distinct_objects %lhs, %rhs
        : memref<16xi32>, memref<16xi32>
    %lhs_aligned = memref.assume_alignment %lhs_distinct, 64
        : memref<16xi32>
    %rhs_aligned = memref.assume_alignment %rhs_distinct, 64
        : memref<16xi32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    scf.for %i = %c0 to %c16 step %c1 {
      %left = memref.load %lhs_aligned[%i] : memref<16xi32>
      %right = memref.load %rhs_aligned[%i] : memref<16xi32>
      %sum = arith.addi %left, %right : i32
      memref.store %sum, %rhs_aligned[%i] : memref<16xi32>
    }
    return
  }
}
)mlir");
  auto attemptDomain =
      take(loom::frontend::enumerateStructuredScheduleDecisions(
          attemptParent, attemptFabric, 4));
  if (attemptDomain.proposals.empty() ||
      attemptDomain.proposals.front().decision().kind !=
          loom::frontend::StructuredScheduleDecisionKind::Vectorize)
    fail("attempt-ledger fixture lost its leading typed refusal");
  auto firstAttempt = loom::frontend::materializeStructuredScheduleProposal(
      attemptParent, attemptDomain.proposals.front(), attemptFabric);
  if (firstAttempt)
    fail("scalar-only Fabric admitted the leading vector proposal");
  bool sawFabricRefusal = false;
  llvm::Error firstAttemptUnhandled = llvm::handleErrors(
      firstAttempt.takeError(),
      [&](const loom::frontend::StructuredScheduleProposalRefusal &error) {
        sawFabricRefusal = error.kind() ==
                           loom::frontend::StructuredScopRefusalKind::
                               FabricCapabilityUnavailable;
      });
  if (firstAttemptUnhandled)
    fail("scalar-only Fabric returned an untyped vector error: " +
         llvm::toString(std::move(firstAttemptUnhandled)));
  if (!sawFabricRefusal)
    fail("scalar-only Fabric lost its typed vector capability refusal");
  auto parentReference =
      take(loom::frontend::publishStructuredProgram(attemptParent, store));
  auto inputs = take(loom::dse::bindStructuredScheduleCandidateGeneratorInputs(
      {parentReference}, attemptFabric.reference()));
  const std::array<loom::dse::CandidateGeneratorOutputDemand, 1> demands = {{
      {loom::dse::CandidateGeneratorOutputSlotRef(0), 2},
  }};
  const loom::dse::CandidateGeneratorInvocationView invocation(
      loom::ExecutionControlView{}, demands);

  auto limitedConfig =
      take(loom::dse::projectResolvedStructuredScheduleGeneratorConfigView(
          loom::defaultResolvedConfig(),
          loom::dse::StructuredScheduleGenerationIntent::Balanced, 1));
  auto limitedBinding =
      take(loom::dse::resolveStructuredScheduleCandidateGeneratorBinding(
          limitedConfig));
  auto limited = take(loom::dse::invokeCandidateGenerator(
      inputs, limitedBinding, store, blobs, invocation));
  const auto *limitedIncomplete =
      std::get_if<loom::dse::IncompleteCandidateGeneratorResult>(
          &limited.outcome);
  if (!limitedIncomplete ||
      limitedIncomplete->reason !=
          loom::dse::CandidateGeneratorIncompleteReason::SemanticLimitReached ||
      limitedIncomplete->retainedOutputBindings.size() != 1 ||
      limitedIncomplete->retainedOutputBindings.front().artifacts.size() != 1 ||
      !limitedIncomplete->lineageEdges.empty() ||
      limited.workSummary.size() != 5 || limited.workSummary[1].planned != 1 ||
      limited.workSummary[1].consumed != 1 ||
      limited.workSummary[3].planned != 2 ||
      limited.workSummary[3].consumed != 1)
    fail("schedule attempt grant did not retain its exact refusal ledger");

  auto refillConfig =
      take(loom::dse::projectResolvedStructuredScheduleGeneratorConfigView(
          loom::defaultResolvedConfig(),
          loom::dse::StructuredScheduleGenerationIntent::Balanced, 32));
  auto refillBinding =
      take(loom::dse::resolveStructuredScheduleCandidateGeneratorBinding(
          refillConfig));
  auto refill = take(loom::dse::invokeCandidateGenerator(
      inputs, refillBinding, store, blobs, invocation));
  const auto *refillIncomplete =
      std::get_if<loom::dse::IncompleteCandidateGeneratorResult>(
          &refill.outcome);
  if (!refillIncomplete ||
      refillIncomplete->reason !=
          loom::dse::CandidateGeneratorIncompleteReason::SemanticLimitReached ||
      refillIncomplete->retainedOutputBindings.size() != 1 ||
      refillIncomplete->retainedOutputBindings.front().artifacts.size() != 2 ||
      refillIncomplete->lineageEdges.size() != 1 ||
      refill.workSummary.size() != 5 || refill.workSummary[1].planned <= 1 ||
      refill.workSummary[1].planned != refill.workSummary[1].consumed ||
      refill.workSummary[3].planned <= refill.workSummary[3].consumed ||
      refill.workSummary[3].consumed != refill.workSummary[1].consumed)
    fail("typed schedule refusal consumed the distinct publication slot");

  const auto loop = structuredLoopReference(parent, "kernel");
  auto analysis =
      take(loom::frontend::analyzeExactStructuredScop(parent, loop));
  const auto *scop =
      std::get_if<loom::frontend::ExactStructuredScopView>(&analysis);
  if (!scop)
    fail("reassociated reduction lost its exact SCoP analysis");
  const loom::frontend::StructuredScheduleDecision decision{
      loop, loom::frontend::StructuredScheduleDecisionKind::Vectorize, 0,
      loom::frontend::StructuredVectorScheduleCoordinate{
          {2},
          loom::frontend::StructuredVectorTailPolicy::ReductionMask,
          8,
          loom::frontend::StructuredVectorAliasPolicy::ProviderProvenNoAlias,
          loom::frontend::StructuredReductionSchedule::FloatingReassociated}};
  auto child = take(
      loom::frontend::materializeStructuredScheduleDecision(parent, decision));
  if (llvm::Error verification =
          loom::frontend::verifyStructuredVectorScheduleMaterialization(
              *scop, *decision.vector, child.structuredProgram.module()))
    fail("independent verifier rejected the masked-tail image: " +
         llvm::toString(std::move(verification)));
  std::size_t maskedReads = 0;
  std::size_t reductions = 0;
  std::size_t extracts = 0;
  std::size_t inserts = 0;
  child.structuredProgram.module().walk([&](mlir::Operation *operation) {
    if (auto read = llvm::dyn_cast<mlir::vector::TransferReadOp>(operation))
      maskedReads += read.getMask() ? 1 : 0;
    reductions += llvm::isa<mlir::vector::ReductionOp>(operation) ? 1 : 0;
    extracts += llvm::isa<mlir::vector::ExtractOp>(operation) ? 1 : 0;
    inserts += llvm::isa<mlir::vector::InsertOp>(operation) ? 1 : 0;
  });
  const std::size_t factor = decision.vector->shape.front();
  if (maskedReads != 1 || reductions != 0 || extracts != factor ||
      inserts != factor)
    fail("reassociated reduction lost its exact masked-tail materialization");

  mlir::OwningOpRef<mlir::ModuleOp> missingMask(
      llvm::cast<mlir::ModuleOp>(child.structuredProgram.module()->clone()));
  mlir::vector::TransferReadOp maskedRead;
  missingMask->walk([&](mlir::vector::TransferReadOp read) {
    if (!maskedRead && read.getMask())
      maskedRead = read;
  });
  if (!maskedRead)
    fail("masked-tail mutation found no transfer mask");
  {
    mlir::OpBuilder builder(maskedRead);
    auto replacement = mlir::vector::TransferReadOp::create(
        builder, maskedRead.getLoc(), maskedRead.getVectorType(),
        maskedRead.getBase(), maskedRead.getIndices(),
        maskedRead.getPermutationMapAttr(), maskedRead.getPadding(),
        mlir::Value{}, maskedRead.getInBoundsAttr());
    maskedRead.getResult().replaceAllUsesWith(replacement.getResult());
    maskedRead.erase();
  }
  llvm::Error missingMaskError =
      loom::frontend::verifyStructuredVectorScheduleMaterialization(
          *scop, *decision.vector, missingMask.get());
  if (!missingMaskError)
    fail("independent verifier accepted a missing tail mask");
  llvm::consumeError(std::move(missingMaskError));

  mlir::OwningOpRef<mlir::ModuleOp> swappedNeutral(
      llvm::cast<mlir::ModuleOp>(child.structuredProgram.module()->clone()));
  mlir::arith::SelectOp tailSelect;
  swappedNeutral->walk([&](mlir::arith::SelectOp select) {
    if (!tailSelect && llvm::isa<mlir::VectorType>(select.getType()))
      tailSelect = select;
  });
  if (!tailSelect)
    fail("masked-tail mutation found no neutral select");
  {
    mlir::OpBuilder builder(tailSelect);
    auto replacement = mlir::arith::SelectOp::create(
        builder, tailSelect.getLoc(), tailSelect.getCondition(),
        tailSelect.getFalseValue(), tailSelect.getTrueValue());
    tailSelect.getResult().replaceAllUsesWith(replacement.getResult());
    tailSelect.erase();
  }
  llvm::Error swappedNeutralError =
      loom::frontend::verifyStructuredVectorScheduleMaterialization(
          *scop, *decision.vector, swappedNeutral.get());
  if (!swappedNeutralError)
    fail("independent verifier accepted a changed neutral arm");
  llvm::consumeError(std::move(swappedNeutralError));

  mlir::OwningOpRef<mlir::ModuleOp> changedCombiner(
      llvm::cast<mlir::ModuleOp>(child.structuredProgram.module()->clone()));
  mlir::arith::AddFOp horizontalAdd;
  changedCombiner->walk([&](mlir::arith::AddFOp add) {
    if (!horizontalAdd && !llvm::isa<mlir::VectorType>(add.getType()))
      horizontalAdd = add;
  });
  if (!horizontalAdd)
    fail("reduction mutation found no horizontal combiner");
  {
    mlir::OpBuilder builder(horizontalAdd);
    auto replacement = mlir::arith::MulFOp::create(
        builder, horizontalAdd.getLoc(), horizontalAdd.getLhs(),
        horizontalAdd.getRhs(), horizontalAdd.getFastMathFlagsAttr());
    horizontalAdd.getResult().replaceAllUsesWith(replacement.getResult());
    horizontalAdd.erase();
  }
  llvm::Error changedCombinerError =
      loom::frontend::verifyStructuredVectorScheduleMaterialization(
          *scop, *decision.vector, changedCombiner.get());
  if (!changedCombinerError)
    fail("independent verifier accepted a changed horizontal combiner");
  llvm::consumeError(std::move(changedCombinerError));

  mlir::OwningOpRef<mlir::ModuleOp> changedResult(
      llvm::cast<mlir::ModuleOp>(child.structuredProgram.module()->clone()));
  mlir::arith::AddFOp reductionRoot;
  mlir::vector::ExtractOp firstLane;
  mlir::func::ReturnOp returned;
  changedResult->walk([&](mlir::Operation *operation) {
    if (auto add = llvm::dyn_cast<mlir::arith::AddFOp>(operation))
      if (!reductionRoot && !llvm::isa<mlir::VectorType>(add.getType()))
        reductionRoot = add;
    if (auto extract = llvm::dyn_cast<mlir::vector::ExtractOp>(operation))
      if (!firstLane &&
          extract.getStaticPosition() == llvm::ArrayRef<std::int64_t>{0})
        firstLane = extract;
    if (auto candidate = llvm::dyn_cast<mlir::func::ReturnOp>(operation))
      returned = candidate;
  });
  if (!reductionRoot || !firstLane || !returned)
    fail("reduction result mutation found no exact result chain");
  {
    mlir::OpBuilder builder(returned);
    mlir::arith::NegFOp::create(builder, returned.getLoc(),
                                reductionRoot.getResult());
    returned->setOperand(0, firstLane.getResult());
  }
  llvm::Error changedResultError =
      loom::frontend::verifyStructuredVectorScheduleMaterialization(
          *scop, *decision.vector, changedResult.get());
  if (!changedResultError)
    fail("independent verifier accepted a foreign reduction result");
  llvm::consumeError(std::move(changedResultError));

  auto primeParent = parseProgram(R"mlir(
module {
  func.func @kernel(%input: memref<67xi32>) {
    %aligned = memref.assume_alignment %input, 256 : memref<67xi32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c67 = arith.constant 67 : index
    scf.for %i = %c0 to %c67 step %c1 {
      %value = memref.load %aligned[%i] : memref<67xi32>
      memref.store %value, %aligned[%i] : memref<67xi32>
    }
    return
  }
}
)mlir");
  auto primeDomain = take(loom::frontend::enumerateStructuredScheduleDecisions(
      primeParent, fabric, 2));
  auto tailRefusal =
      llvm::find_if(primeDomain.refusals, [](const auto &candidate) {
        return candidate.kind ==
               loom::frontend::StructuredScopRefusalKind::UnsupportedTail;
      });
  if (tailRefusal == primeDomain.refusals.end())
    fail("non-reduction tail rejection lost its typed local refusal");
  llvm::sys::fs::remove_directories(directory);
}

void nestedImperfectScopFreezesExactRelations() {
  auto parent = parseProgram(R"mlir(
#layout = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
module attributes {dlti.dl_spec = #layout} {
  func.func @kernel(%a: memref<?xi32>, %b: memref<?xi32>,
                    %c: memref<?x?xi32>, %d: memref<?xi32>,
                    %m: index, %n: index) {
    %a0, %b0, %c0, %d0 = memref.distinct_objects %a, %b, %c, %d
        : memref<?xi32>, memref<?xi32>, memref<?x?xi32>, memref<?xi32>
    %a1 = memref.assume_alignment %a0, 64 : memref<?xi32>
    %d1 = memref.assume_alignment %d0, 64 : memref<?xi32>
    affine.for %i = 0 to %m {
      %outer = affine.load %a0[%i] : memref<?xi32>
      affine.for %j = 0 to %n {
        %lhs = affine.load %a0[%i] : memref<?xi32>
        %rhs = affine.load %b0[%j] : memref<?xi32>
        %sum = arith.addi %lhs, %rhs : i32
        affine.store %sum, %c0[%i, %j] : memref<?x?xi32>
        %roundtrip = affine.load %c0[%i, %j] : memref<?x?xi32>
      }
      %after = affine.load %d0[%i] : memref<?xi32>
    }
    affine.for %k = 0 to %m {
      %flat = affine.load %a1[%k] : memref<?xi32>
      affine.store %flat, %d1[%k] : memref<?xi32>
    }
    return
  }
}
)mlir");
  auto view = take(parent.view());
  std::optional<loom::frontend::StructuredEntityRef> loop;
  for (const loom::frontend::StructuredEntity &entity :
       view.entities(loom::frontend::StructuredEntityKind::Operation)) {
    auto affine =
        llvm::dyn_cast_or_null<mlir::affine::AffineForOp>(entity.operation);
    if (!affine || affine->getParentOfType<mlir::affine::AffineForOp>())
      continue;
    std::uint64_t nestedLoopCount = 0;
    affine->walk([&](mlir::affine::AffineForOp) { ++nestedLoopCount; });
    if (nestedLoopCount != 2)
      continue;
    auto function = affine->getParentOfType<mlir::func::FuncOp>();
    if (function && function.getSymName() == "kernel") {
      loop = entity.reference;
      break;
    }
  }
  if (!loop)
    fail("nested symbolic fixture lost its top-level loop");
  auto outcome =
      take(loom::frontend::analyzeStructuredPolyhedralScop(parent, *loop));
  const auto *scop =
      std::get_if<loom::frontend::StructuredPolyhedralScopView>(&outcome);
  if (!scop) {
    const auto &refusal =
        std::get<loom::frontend::StructuredScopRefusal>(outcome);
    fail("nested symbolic SCoP refusal=" +
         std::to_string(static_cast<std::uint32_t>(refusal.kind)));
  }
  if (scop->loopCount != 2 || scop->maximumLoopDepth != 2 ||
      !scop->imperfectNest || scop->statements.size() != 7 ||
      scop->accesses.size() != 6 || scop->dependenceQueryCount != 9 ||
      scop->schedule.provider !=
          loom::frontend::StructuredPolyhedralProviderKind::PinnedPollyIsl ||
      scop->parameters.size() != 2 || scop->schedule.parameterCount != 2 ||
      scop->schedule.scheduleMapCount() != 7)
    fail("nested symbolic SCoP summary mismatch: loops=" +
         std::to_string(scop->loopCount) +
         " depth=" + std::to_string(scop->maximumLoopDepth) +
         " imperfect=" + std::to_string(scop->imperfectNest) +
         " statements=" + std::to_string(scop->statements.size()) +
         " accesses=" + std::to_string(scop->accesses.size()) +
         " queries=" + std::to_string(scop->dependenceQueryCount) +
         " global_parameters=" + std::to_string(scop->parameters.size()) +
         " parameters=" + std::to_string(scop->schedule.parameterCount) +
         " provider=" +
         std::to_string(static_cast<std::uint32_t>(scop->schedule.provider)) +
         " maps=" + std::to_string(scop->schedule.scheduleMapCount()));
  const std::size_t scalarDependences =
      llvm::count_if(scop->dependences, [](const auto &dependence) {
        return dependence.kind ==
               loom::frontend::StructuredPolyhedralDependenceKind::ScalarSsa;
      });
  auto memoryDependence = llvm::find_if(scop->dependences, [](const auto
                                                                  &dependence) {
    return dependence.kind ==
           loom::frontend::StructuredPolyhedralDependenceKind::ReadAfterWrite;
  });
  if (scalarDependences != 3 || memoryDependence == scop->dependences.end() ||
      !memoryDependence->relation ||
      memoryDependence->relation->sourceDimensionCount != 2 ||
      memoryDependence->relation->destinationDimensionCount != 2 ||
      memoryDependence->relation->parameters.size() != 2)
    fail("nested symbolic SCoP changed exact dependence ownership");
  auto matrixWrite = llvm::find_if(scop->accesses, [](const auto &access) {
    return access.kind ==
               loom::frontend::StructuredPolyhedralAccessKind::Write &&
           access.relation.destinationDimensionCount == 2;
  });
  if (matrixWrite == scop->accesses.end() ||
      matrixWrite->relation.sourceDimensionCount != 2 ||
      matrixWrite->relation.parameters.size() != 2 ||
      matrixWrite->constantFootprintElementUpperBound)
    fail("multidimensional symbolic footprint was not frozen exactly");
  for (const auto &statement : scop->statements)
    if (statement.domain.constraints.empty() ||
        statement.domain.dimensions.empty())
      fail("nested statement lost its exact Presburger domain");

  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-schedule-polyhedral", directory);
  if (error)
    fail("cannot create ArtifactStore directory: " + error.message());
  loom::ArtifactStore store(directory);
  llvm::SmallString<128> blobPath(directory);
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code blobError = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + blobError.message());
  const loom::BlobStore blobs(blobPath);
  auto design = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));
  auto domain = take(loom::frontend::enumerateStructuredScheduleDecisions(
      parent, design.roots().front(), 3));
  if (domain.polyhedralScops.size() != 2 ||
      llvm::none_of(
          domain.polyhedralScops,
          [&](const auto &candidate) { return candidate.root == *loop; }) ||
      domain.inspectedPolyhedralDependenceQueries != 17 ||
      llvm::none_of(domain.refusals,
                    [](const auto &refusal) {
                      return refusal.kind ==
                             loom::frontend::StructuredScopRefusalKind::
                                 PolyhedralMaterializationUnavailable;
                    }) ||
      llvm::none_of(domain.refusals, [](const auto &refusal) {
        return refusal.kind ==
               loom::frontend::StructuredScopRefusalKind::NestedAffineRoot;
      }))
    fail("production enumeration dropped the admitted polyhedral SCoP: " +
         std::to_string(domain.polyhedralScops.size()) + " scops, " +
         std::to_string(domain.inspectedPolyhedralDependenceQueries) +
         " queries, " + std::to_string(domain.refusals.size()) + " refusals");
  auto parentReference =
      take(loom::frontend::publishStructuredProgram(parent, store));
  auto inputs = take(loom::dse::bindStructuredScheduleCandidateGeneratorInputs(
      {parentReference}, design.roots().front().reference()));
  auto config =
      take(loom::dse::projectResolvedStructuredScheduleGeneratorConfigView(
          loom::defaultResolvedConfig(),
          loom::dse::StructuredScheduleGenerationIntent::Balanced, 3));
  auto binding = take(
      loom::dse::resolveStructuredScheduleCandidateGeneratorBinding(config));
  auto generated =
      take(loom::dse::invokeCandidateGenerator(inputs, binding, store, blobs));
  const auto *incomplete =
      std::get_if<loom::dse::IncompleteCandidateGeneratorResult>(
          &generated.outcome);
  // Nested general 9 + flat exact 6 + flat general 2.
  if (!incomplete ||
      incomplete->reason !=
          loom::dse::CandidateGeneratorIncompleteReason::ProofNotEstablished ||
      generated.workSummary.size() != 5 ||
      generated.workSummary[4].planned != 17 ||
      generated.workSummary[4].consumed != 17)
    fail("production work ledger dropped exact dependence queries: " +
         std::to_string(generated.workSummary.size() == 5
                            ? generated.workSummary[4].consumed
                            : 0));
  llvm::sys::fs::remove_directories(directory);
}

void productionVectorScheduleLowersToCanonicalDataflow() {
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-schedule-vector-lowering", directory);
  if (error)
    fail("cannot create ArtifactStore directory: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));
  const loom::fabric::FinalizedFabricRoot &fabric = design.roots().front();

  auto parent = parseProgram(R"mlir(
#layout = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
module attributes {dlti.dl_spec = #layout} {
  memref.global constant @vector_lhs : memref<16xi32> = dense<1>
  memref.global @vector_rhs : memref<16xi32> = dense<0>

  dataflow.thread private @vector_thread
      domain(#dataflow.thread_domain<dense>)(
      %lhs: memref<16xi32>, %rhs: memref<16xi32>) ctrl (%start: none) {
    "loom.spatial_region"(%lhs, %rhs)
        <{operandSegmentSizes = array<i32: 0, 0, 2, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%left_input: memref<16xi32>, %right_input: memref<16xi32>):
        %left_distinct, %right_distinct = memref.distinct_objects
            %left_input, %right_input : memref<16xi32>, memref<16xi32>
        %left_aligned = memref.assume_alignment %left_distinct, 64
            : memref<16xi32>
        %right_aligned = memref.assume_alignment %right_distinct, 64
            : memref<16xi32>
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c16 = arith.constant 16 : index
        scf.for %i = %c0 to %c16 step %c1 {
          %left = memref.load %left_aligned[%i] : memref<16xi32>
          %right = memref.load %right_aligned[%i] : memref<16xi32>
          %sum = arith.addi %left, %right : i32
          memref.store %sum, %right_aligned[%i] : memref<16xi32>
        }
        scf.for %j = %c0 to %c16 step %c1 {
          %doubled = arith.addi %j, %j : index
        }
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "vector_graph", source_maps = []} :
        (memref<16xi32>, memref<16xi32>) -> ()
    dataflow.thread.yield
  }

  llvm.func @entry() {
    %lhs = memref.get_global @vector_lhs : memref<16xi32>
    %rhs = memref.get_global @vector_rhs : memref<16xi32>
    %token = dataflow.thread.launch @vector_thread(%lhs, %rhs) :
        (memref<16xi32>, memref<16xi32>) -> !dataflow.thread_token
    dataflow.thread.wait %token : !dataflow.thread_token
    llvm.return
  }
}
)mlir");
  auto domain = take(
      loom::frontend::enumerateStructuredScheduleDecisions(parent, fabric, 8));
  auto vectorProposal =
      llvm::find_if(domain.proposals, [](const auto &proposal) {
        return proposal.decision().kind ==
               loom::frontend::StructuredScheduleDecisionKind::Vectorize;
      });
  if (vectorProposal == domain.proposals.end())
    fail("production vector fixture produced no admitted coordinate");
  const loom::frontend::StructuredScheduleDecision &decision =
      vectorProposal->decision();
  auto child = take(
      loom::frontend::materializeStructuredScheduleDecision(parent, decision));
  auto frozenChild = take(loom::frontend::materializeStructuredScheduleProposal(
      parent, *vectorProposal, fabric));
  if (frozenChild.structuredProgram.identity() !=
      child.structuredProgram.identity())
    fail("frozen SCoP proposal changed the exact materialized child");
  if (llvm::Error verification =
          loom::frontend::verifyStructuredScheduleDerivation(
              parent, fabric, decision, frozenChild.structuredProgram))
    fail("schedule derivation replay rejected its exact child: " +
         llvm::toString(std::move(verification)));

  auto parallelProposal =
      llvm::find_if(domain.proposals, [](const auto &proposal) {
        return proposal.decision().kind ==
               loom::frontend::StructuredScheduleDecisionKind::Parallelize;
      });
  if (parallelProposal == domain.proposals.end())
    fail("production Spatial fixture produced no parallel proposal");
  auto view = take(parent.view());
  std::optional<loom::frontend::StructuredEntityRef> spatial;
  for (const loom::frontend::StructuredEntity &entity :
       view.entities(loom::frontend::StructuredEntityKind::Operation))
    if (llvm::isa_and_nonnull<loom::SpatialRegionOp>(entity.operation)) {
      spatial = entity.reference;
      break;
    }
  if (!spatial)
    fail("production Spatial fixture lost its exact region reference");
  auto untrackedParallel =
      take(loom::frontend::materializeStructuredScheduleProposal(
          parent, *parallelProposal, fabric));
  auto trackedParallel =
      take(loom::frontend::materializeStructuredScheduleProposal(
          parent, *parallelProposal, fabric, *spatial));
  if (untrackedParallel.structuredProgram.identity() !=
          trackedParallel.structuredProgram.identity() ||
      untrackedParallel.trackedSpatialRegion ||
      !trackedParallel.trackedSpatialRegion)
    fail("tracked Spatial projection changed schedule child semantics");
  auto lowered = take(loom::lowering::lowerStructuredProgramToCanonicalDataflow(
      child.structuredProgram));

  std::size_t vectorLoads = 0;
  std::size_t vectorStores = 0;
  std::size_t vectorAdds = 0;
  std::size_t transfers = 0;
  lowered.module().walk([&](mlir::Operation *operation) {
    if (auto load = llvm::dyn_cast<dataflow::LoadOp>(operation))
      vectorLoads += llvm::isa<mlir::VectorType>(load.getData().getType());
    if (auto store = llvm::dyn_cast<dataflow::StoreOp>(operation))
      vectorStores += llvm::isa<mlir::VectorType>(store.getData().getType());
    if (auto add = llvm::dyn_cast<mlir::arith::AddIOp>(operation))
      vectorAdds += llvm::isa<mlir::VectorType>(add.getType());
    transfers +=
        llvm::isa<mlir::vector::TransferReadOp, mlir::vector::TransferWriteOp>(
            operation);
  });
  if (vectorLoads != 2 || vectorStores != 1 || vectorAdds != 1 ||
      transfers != 0)
    fail("production lowering did not preserve the admitted vector actors");
  llvm::sys::fs::remove_directories(directory);
}

} // namespace

int main() {
  configRoundTripsAndRejectsMalformedBytes();
  decisionCodecRejectsFactorOne();
  invalidInMemoryDecisionFailsClosed();
  lineageCodecRejectsAnOutOfRangeLoop();
  lineageRejectsAValidForeignChild();
  transformationsAreTypedCapacityBoundAndDependenceChecked();
  exactScopVectorCoordinateIsCanonicalAndVerified();
  symbolicScopUsesExactParameterIdentity();
  exactScopRefusalsAreLocalAndTyped();
  reassociatedReductionOwnsItsMaskedTail();
  nestedImperfectScopFreezesExactRelations();
  productionVectorScheduleLowersToCanonicalDataflow();
  return EXIT_SUCCESS;
}
