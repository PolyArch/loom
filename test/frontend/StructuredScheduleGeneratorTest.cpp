#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/CandidateGenerator.h"
#include "DSE/StructuredScheduleCandidateGenerator.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Frontend/Compilation/FabricCapabilityIndex.h"
#include "Frontend/Compilation/StructuredSchedule.h"
#include "Frontend/IR/StructuredProgramArtifact.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
    registry.insert<mlir::arith::ArithDialect, mlir::DLTIDialect,
                    mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                    mlir::scf::SCFDialect>();
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
    auto count = take(index.admittingOperationResourceCount(operation));
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
  auto *completed = std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
      &outcome.outcome);
  if (!completed || completed->outputBindings.size() != 1)
    fail("schedule generator did not complete one output set");
  for (const loom::dse::CandidateGeneratorLineageEdge &edge :
       completed->lineageEdges) {
    if (edge.kind !=
            loom::dse::CandidateGeneratorLineageEdgeKind::CandidateDecision ||
        edge.parents !=
            std::vector<loom::ArtifactRootReference>{programReference})
      fail("schedule generator changed its parent lineage");
    take(loom::frontend::adoptStructuredScheduleDecision(edge.ownerPayload));
  }
  return completed->outputBindings.front().artifacts;
}

void configRoundTripsAndRejectsMalformedBytes() {
  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.schedule.scopeExpansionLimit = 7;
  auto projected =
      take(loom::dse::projectResolvedStructuredScheduleGeneratorConfigView(
          resolved));
  auto adopted =
      take(loom::dse::adoptResolvedStructuredScheduleGeneratorConfigView(
          loom::dse::resolvedStructuredScheduleGeneratorConfigSchemaBytes(),
          projected.canonicalViewBytes(), projected.digest()));
  if (adopted.scopeExpansionLimit() != 7 ||
      adopted.canonicalViewBytes() != projected.canonicalViewBytes())
    fail("schedule config did not round-trip exactly");

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
      *loop, loom::frontend::StructuredScheduleDecisionKind::Tile, 1};
  auto encoded = loom::frontend::encodeStructuredScheduleDecision(decision);
  if (encoded)
    fail("schedule decision encoder accepted factor one");
  llvm::consumeError(encoded.takeError());
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
      *loop, static_cast<loom::frontend::StructuredScheduleDecisionKind>(99),
      0};
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
      2};
  auto encoded =
      take(loom::frontend::encodeStructuredScheduleDecision(decision));
  const auto *contract =
      loom::dse::structuredScheduleCandidateGeneratorDescriptor()
          .ownerLineagePayload;
  if (!contract)
    fail("schedule generator has no owner lineage contract");
  llvm::Error validation =
      contract->validateCanonical(encoded, {parentReference}, store);
  if (!validation)
    fail("schedule lineage accepted an out-of-range parent-local loop");
  llvm::consumeError(std::move(validation));
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

  auto safe = parseProgram(R"mlir(
#layout = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
module attributes {dlti.dl_spec = #layout} {
  func.func @kernel(%out: memref<32xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c4 step %c1 {
        %row = arith.muli %i, %c4 : index
        %index = arith.addi %row, %j : index
        %value = arith.index_cast %index : index to i32
        memref.store %value, %out[%index] : memref<32xi32>
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
  for (const loom::ArtifactRootReference &reference : safeOutputs) {
    auto candidate =
        take(loom::frontend::importStructuredProgram(reference, store));
    sawParallel |= forallCount(candidate, "kernel") != 0;
  }
  if (!sawParallel)
    fail("proven-independent loop produced no parallel child");

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
  func.func @kernel(%out: memref<128xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    scf.for %i = %c0 to %c128 step %c1 {
      %value = arith.addi %i, %i : index
      %stored = arith.index_cast %value : index to i32
      memref.store %stored, %out[%i] : memref<128xi32>
    }
    return
  }
}
)mlir");
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

  llvm::sys::fs::remove_directories(directory);
}

} // namespace

int main() {
  configRoundTripsAndRejectsMalformedBytes();
  decisionCodecRejectsFactorOne();
  invalidInMemoryDecisionFailsClosed();
  lineageCodecRejectsAnOutOfRangeLoop();
  transformationsAreTypedCapacityBoundAndDependenceChecked();
  return EXIT_SUCCESS;
}
