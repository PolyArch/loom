#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/CandidateGenerator.h"
#include "DSE/StructuredScheduleCandidateGenerator.h"
#include "Frontend/Compilation/StructuredSchedule.h"
#include "Frontend/Compilation/StructuredScop.h"
#include "Frontend/IR/LoomDialect.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Simulator/NativeSimulationOracle.h"
#include "Simulator/SimulationArtifacts.h"
#include "StructuredPolyhedralMaterializer.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace {

[[noreturn]] void fail(const std::string &message) {
  llvm::errs() << "structuredPolyhedralSchedule: " << message << '\n';
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
                    mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                    mlir::DLTIDialect, mlir::func::FuncDialect,
                    mlir::LLVM::LLVMDialect, mlir::memref::MemRefDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    auto *created =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    created->loadAllAvailableDialects();
    return created;
  }();
  return *result;
}

llvm::StringRef nativeDataLayout() {
  static std::string layout = [] {
    if (llvm::InitializeNativeTarget() ||
        llvm::InitializeNativeTargetAsmPrinter())
      fail("cannot initialize the native target");
    auto target = take(llvm::orc::JITTargetMachineBuilder::detectHost());
    return take(target.getDefaultDataLayoutForTarget())
        .getStringRepresentation();
  }();
  return layout;
}

llvm::StringRef nativeTargetTriple() {
  static std::string triple = [] {
    auto target = take(llvm::orc::JITTargetMachineBuilder::detectHost());
    return target.getTargetTriple().str();
  }();
  return triple;
}

loom::frontend::StructuredProgramCandidate parseProgram(llvm::StringRef text) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context());
  if (!module)
    fail("cannot parse Structured Program fixture");
  module->getOperation()->setAttr(
      "llvm.data_layout",
      mlir::StringAttr::get(&context(), nativeDataLayout()));
  module->getOperation()->setAttr(
      "llvm.target_triple",
      mlir::StringAttr::get(&context(), nativeTargetTriple()));
  return take(loom::frontend::finalizeStructuredProgram(module.get()));
}

loom::frontend::StructuredEntityRef affineRootReference(
    const loom::frontend::StructuredProgramCandidate &candidate) {
  auto view = take(candidate.view());
  std::optional<loom::frontend::StructuredEntityRef> root;
  for (const loom::frontend::StructuredEntity &entity :
       view.entities(loom::frontend::StructuredEntityKind::Operation)) {
    auto loop =
        llvm::dyn_cast_or_null<mlir::affine::AffineForOp>(entity.operation);
    if (!loop || loop->getParentOfType<mlir::affine::AffineForOp>())
      continue;
    if (root)
      fail("fixture has more than one affine root");
    root = entity.reference;
  }
  if (!root)
    fail("fixture lost its affine root");
  return *root;
}

std::pair<loom::frontend::StructuredEntityRef,
          loom::frontend::StructuredEntityRef>
loopReferences(const loom::frontend::StructuredProgramCandidate &candidate) {
  auto view = take(candidate.view());
  std::optional<loom::frontend::StructuredEntityRef> outer;
  std::optional<loom::frontend::StructuredEntityRef> inner;
  for (const loom::frontend::StructuredEntity &entity :
       view.entities(loom::frontend::StructuredEntityKind::Operation)) {
    auto loop =
        llvm::dyn_cast_or_null<mlir::affine::AffineForOp>(entity.operation);
    if (!loop)
      continue;
    if (loop->getParentOfType<mlir::affine::AffineForOp>())
      inner = entity.reference;
    else
      outer = entity.reference;
  }
  if (!outer || !inner)
    fail("fixture lost its exact loop references");
  return {*outer, *inner};
}

std::pair<loom::frontend::StructuredEntityRef,
          loom::frontend::StructuredEntityRef>
scfLoopReferences(const loom::frontend::StructuredProgramCandidate &candidate) {
  auto view = take(candidate.view());
  std::optional<loom::frontend::StructuredEntityRef> outer;
  std::optional<loom::frontend::StructuredEntityRef> inner;
  for (const loom::frontend::StructuredEntity &entity :
       view.entities(loom::frontend::StructuredEntityKind::Operation)) {
    auto loop = llvm::dyn_cast_or_null<mlir::scf::ForOp>(entity.operation);
    if (!loop)
      continue;
    if (loop->getParentOfType<mlir::scf::ForOp>())
      inner = entity.reference;
    else
      outer = entity.reference;
  }
  if (!outer || !inner)
    fail("fixture lost its exact SCF loop references");
  return {*outer, *inner};
}

void physicalLayoutInjectivityIsRequired() {
  const auto requireRefusal = [](llvm::StringRef text) {
    auto candidate = parseProgram(text);
    auto analysis = take(loom::frontend::analyzeStructuredPolyhedralScop(
        candidate, loopReferences(candidate).first));
    const auto *refusal =
        std::get_if<loom::frontend::StructuredScopRefusal>(&analysis);
    if (!refusal || refusal->kind != loom::frontend::StructuredScopRefusalKind::
                                         PhysicalLayoutProofNotEstablished)
      fail("a physically aliasing memref entered the general schedule domain");
  };
  requireRefusal(R"mlir(
module {
  func.func @stride_zero(%state: memref<2xi32, strided<[0]>>) {
    %value = arith.constant 7 : i32
    affine.for %i = 0 to 2 {
      affine.for %j = 0 to 2 {
        affine.store %value, %state[%i] : memref<2xi32, strided<[0]>>
      }
    }
    return
  }
}
)mlir");
  requireRefusal(R"mlir(
module {
  func.func @overlap(%state: memref<2x2xi32, strided<[1, 1]>>) {
    %value = arith.constant 11 : i32
    affine.for %i = 0 to 2 {
      affine.for %j = 0 to 2 {
        affine.store %value, %state[%i, %j]
            : memref<2x2xi32, strided<[1, 1]>>
      }
    }
    return
  }
}
)mlir");

  auto padded = parseProgram(R"mlir(
module {
  func.func @padded(
      %state: memref<2x3xi32, strided<[4, 1], offset: 5>>) {
    %value = arith.constant 13 : i32
    affine.for %i = 0 to 2 {
      affine.for %j = 0 to 3 {
        affine.store %value, %state[%i, %j]
            : memref<2x3xi32, strided<[4, 1], offset: 5>>
      }
    }
    return
  }
}
)mlir");
  auto paddedAnalysis = take(loom::frontend::analyzeStructuredPolyhedralScop(
      padded, loopReferences(padded).first));
  if (!std::holds_alternative<loom::frontend::StructuredPolyhedralScopView>(
          paddedAnalysis))
    fail("a proven injective padded layout left the general schedule domain");
}

loom::frontend::StructuredPolyhedralSchedulePieceView
divisionSchedulePiece(std::uint64_t statementOrdinal, std::int64_t parity,
                      bool fission) {
  using Constraint = loom::frontend::StructuredPolyhedralConstraintView;
  using ConstraintKind = loom::frontend::StructuredPolyhedralConstraintKind;
  constexpr std::size_t rowWidth = 8;
  loom::frontend::StructuredPolyhedralSchedulePieceView piece;
  piece.sourceDimensionCount = 2;
  piece.scheduleDimensionCount = 4;
  piece.parameterCount = 0;
  piece.divisions.push_back({2, {0, 1, 0, 0, 0, 0, 0, 0}});
  const auto equality = [&](std::initializer_list<std::int64_t> values) {
    std::vector<std::int64_t> row(values);
    if (row.size() != rowWidth)
      fail("a local-div schedule row has the wrong width");
    piece.constraints.push_back(
        Constraint{ConstraintKind::Equality, std::move(row)});
  };
  const std::int64_t statement = static_cast<std::int64_t>(statementOrdinal);
  if (fission) {
    equality({0, 0, 1, 0, 0, 0, 0, -statement});
    equality({-1, 0, 0, 1, 0, 0, 0, 0});
    equality({-1, 0, 0, 0, 1, 0, -1, 0});
    equality({0, -1, 0, 0, 0, 1, 2, 0});
    equality({0, 0, 0, 0, 0, 1, 0, -parity});
  } else {
    equality({-1, 0, 1, 0, 0, 0, 0, 0});
    equality({-1, 0, 0, 1, 0, 0, -1, 0});
    equality({0, -1, 0, 0, 1, 0, 2, 0});
    equality({0, 0, 0, 0, 0, 1, 0, -statement});
    equality({0, 0, 0, 0, 1, 0, 0, -parity});
  }
  return piece;
}

loom::frontend::StructuredPolyhedralScopView
withDivisionSchedule(loom::frontend::StructuredPolyhedralScopView scop,
                     bool fission) {
  if (scop.statements.size() != 2 || !scop.parameters.empty())
    fail("the local-div fixture has an unexpected frozen SCoP");
  loom::frontend::StructuredPolyhedralScheduleView schedule;
  schedule.provider =
      loom::frontend::StructuredPolyhedralProviderKind::PinnedPollyIsl;
  schedule.form = loom::frontend::StructuredPolyhedralScheduleForm::General;
  schedule.parameterCount = 0;
  schedule.dependenceCount = 0;
  schedule.scheduleBandCount = 2;
  schedule.scheduleDimensionCount = 4;
  schedule.coincidentDimensionCount = 0;
  for (std::uint64_t statement = 0; statement != scop.statements.size();
       ++statement) {
    loom::frontend::StructuredPolyhedralStatementScheduleView statementView;
    statementView.statementOrdinal = statement;
    statementView.pieces.push_back(
        divisionSchedulePiece(statement, 0, fission));
    statementView.pieces.push_back(
        divisionSchedulePiece(statement, 1, fission));
    schedule.statementSchedules.push_back(std::move(statementView));
  }
  scop.schedule = std::move(schedule);
  return scop;
}

loom::frontend::StructuredProgramCandidate materializeFrozenScop(
    const loom::frontend::StructuredProgramCandidate &parent,
    const loom::frontend::StructuredEntityRef &root,
    const loom::frontend::StructuredPolyhedralScopView &scop) {
  auto parentView = take(parent.view());
  auto source = take(parentView.resolve(root));
  mlir::IRMapping mapping;
  auto clone = take(loom::frontend::cloneStructuredProgramWithSourceLocations(
      parent, {}, mapping));
  mlir::Operation *clonedRoot = mapping.lookupOrNull(source.operation);
  if (!clonedRoot)
    fail("the local-div root was not mapped into its private clone");
  llvm::SmallVector<mlir::Operation *> materializedOperations;
  auto materialized = take(loom::frontend::detail::materializePinnedIslSchedule(
      clonedRoot, scop, parentView, mapping, materializedOperations));
  if (materialized)
    fail("the local-div schedule was refused with kind " +
         std::to_string(static_cast<std::uint32_t>(*materialized)));
  if (materializedOperations.empty() || mlir::failed(mlir::verify(*clone)))
    fail("the local-div materializer did not produce a valid exact clone");
  return take(loom::frontend::finalizeStructuredProgram(clone.get()));
}

std::pair<loom::frontend::StructuredEntityRef,
          loom::frontend::StructuredEntityRef>
nativeOracleReferences(
    const loom::frontend::StructuredProgramCandidate &candidate) {
  auto view = take(candidate.view());
  std::optional<loom::frontend::StructuredEntityRef> entry;
  std::optional<loom::frontend::StructuredEntityRef> observation;
  for (const loom::frontend::StructuredEntity &entity :
       view.entities(loom::frontend::StructuredEntityKind::Operation)) {
    if (auto function =
            llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(entity.operation);
        function && function.getSymName() == "entry")
      entry = entity.reference;
    if (auto global =
            llvm::dyn_cast_or_null<mlir::LLVM::GlobalOp>(entity.operation);
        global && global.getSymName() == "observation")
      observation = entity.reference;
  }
  if (!entry || !observation)
    fail("the local-div oracle lost its entry or observation global");
  return {*entry, *observation};
}

loom::frontend::StructuredProgramCandidate withoutDistinctObjectAssumptions(
    const loom::frontend::StructuredProgramCandidate &candidate) {
  mlir::OwningOpRef<mlir::ModuleOp> clone(
      llvm::cast<mlir::ModuleOp>(candidate.module()->clone()));
  llvm::SmallVector<mlir::memref::DistinctObjectsOp> assumptions;
  clone->walk([&](mlir::memref::DistinctObjectsOp operation) {
    assumptions.push_back(operation);
  });
  for (mlir::memref::DistinctObjectsOp operation : assumptions) {
    for (auto [result, operand] :
         llvm::zip(operation.getResults(), operation.getOperands()))
      result.replaceAllUsesWith(operand);
    operation.erase();
  }
  mlir::PassManager manager(clone->getContext());
  manager.addPass(mlir::createLowerAffinePass());
  if (mlir::failed(manager.run(*clone)))
    fail("the independent oracle could not lower affine control");
  return take(loom::frontend::finalizeStructuredProgram(clone.get()));
}

void requireEquivalentNativeObservations(
    const loom::frontend::StructuredProgramCandidate &parent,
    const loom::frontend::StructuredProgramCandidate &child) {
  auto executableParent = withoutDistinctObjectAssumptions(parent);
  auto executableChild = withoutDistinctObjectAssumptions(child);
  auto references = nativeOracleReferences(executableParent);
  auto view = take(executableParent.view());
  loom::sim::StructuredProgramSimulationWorkload workloadDraft{
      references.first};
  workloadDraft.observableContract.returnValue = true;
  workloadDraft.observableContract.memories.push_back(
      {loom::sim::GlobalObjectTarget{references.second},
       loom::sim::MemoryObservationForm::FullState});
  auto workload =
      take(loom::sim::finalizeSimulationWorkload(workloadDraft, view));
  loom::sim::StructuredProgramSimulationRuntimeInputDraft inputDraft{
      workload.identity()};
  auto input = take(
      loom::sim::finalizeSimulationRuntimeInput(inputDraft, workload, view));
  auto expected = take(loom::sim::executeNativeStructuredProgram(
      executableParent, workload, input));
  auto actual = take(loom::sim::executeSelectedStructuredProgram(
      executableChild, executableParent, workload, input));
  if (expected.memories.size() != 1 || actual.memories.size() != 1 ||
      !expected.returnValue || expected.returnValue->lanes.size() != 1 ||
      expected.returnValue->lanes.front().bits != llvm::APInt(32, 18) ||
      !loom::sim::haveEquivalentFunctionalObservations(expected, actual))
    fail("the local-div parent and child changed native memory observations");
}

void localDivisionSchedulesHaveIndependentSemantics() {
  auto parent = parseProgram(R"mlir(
#layout = #dlti.dl_spec<#dlti.dl_entry<index, 64>>
module attributes {dlti.dl_spec = #layout} {
  memref.global @left : memref<2x4xi32> = dense<0>
  memref.global @right : memref<2x4xi32> = dense<0>
  llvm.mlir.global internal @observation(0 : i32) : i32

  llvm.func @entry() -> i32 {
    %left = memref.get_global @left : memref<2x4xi32>
    %right = memref.get_global @right : memref<2x4xi32>
    %left0, %right0 = memref.distinct_objects %left, %right
        : memref<2x4xi32>, memref<2x4xi32>
    %seven = arith.constant 7 : i32
    %eleven = arith.constant 11 : i32
    affine.for %i = 0 to 2 {
      affine.for %j = 0 to 4 {
        affine.store %seven, %left0[%i, %j] : memref<2x4xi32>
        affine.store %eleven, %right0[%i, %j] : memref<2x4xi32>
      }
    }
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %lhs = memref.load %left0[%c1, %c3] : memref<2x4xi32>
    %rhs = memref.load %right0[%c1, %c3] : memref<2x4xi32>
    %sum = arith.addi %lhs, %rhs : i32
    %address = llvm.mlir.addressof @observation : !llvm.ptr
    llvm.store %sum, %address : i32, !llvm.ptr
    llvm.return %sum : i32
  }
}
)mlir");
  const auto root = loopReferences(parent).first;
  auto analysis =
      take(loom::frontend::analyzeStructuredPolyhedralScop(parent, root));
  const auto *original =
      std::get_if<loom::frontend::StructuredPolyhedralScopView>(&analysis);
  if (!original)
    fail("the local-div oracle was refused with kind " +
         std::to_string(static_cast<std::uint32_t>(
             std::get<loom::frontend::StructuredScopRefusal>(analysis).kind)));
  if (original->statements.size() != 2 || !original->dependences.empty())
    fail("the local-div oracle did not form two independent statements");

  auto fusedScop = withDivisionSchedule(*original, false);
  auto fused = materializeFrozenScop(parent, root, fusedScop);
  requireEquivalentNativeObservations(parent, fused);
  auto fissionScop = withDivisionSchedule(*original, true);
  auto fission = materializeFrozenScop(parent, root, fissionScop);
  requireEquivalentNativeObservations(parent, fission);
}

void scfStatementMajorScheduleMaterializes(
    const loom::fabric::FinalizedFabricRoot &fabric) {
  auto parent = parseProgram(R"mlir(
module {
  func.func @scf_kernel(%state: memref<?x?xi32>, %m: index, %n: index,
                        %lhs: i32, %rhs: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %m step %c1 {
      scf.for %j = %c0 to %n step %c1 {
        memref.store %lhs, %state[%i, %j] : memref<?x?xi32>
        memref.store %rhs, %state[%i, %j] : memref<?x?xi32>
      }
    }
    return
  }
}
)mlir");
  const auto references = scfLoopReferences(parent);
  auto analysis = take(loom::frontend::analyzeStructuredPolyhedralScop(
      parent, references.first));
  const auto *scop =
      std::get_if<loom::frontend::StructuredPolyhedralScopView>(&analysis);
  if (!scop ||
      scop->schedule.form !=
          loom::frontend::StructuredPolyhedralScheduleForm::StatementMajor)
    fail("symbolic SCF fixture left the statement-major provider form: " +
         std::to_string(scop ? static_cast<std::uint32_t>(scop->schedule.form)
                             : 99));
  auto domain = take(
      loom::frontend::enumerateStructuredScheduleDecisions(parent, fabric, 2));
  auto proposal = llvm::find_if(domain.proposals, [](const auto &candidate) {
    return candidate.decision().kind ==
           loom::frontend::StructuredScheduleDecisionKind::PolyhedralSchedule;
  });
  if (proposal == domain.proposals.end())
    fail("symbolic SCF schedule did not produce a materialization proposal");
  auto child = take(loom::frontend::materializeStructuredScheduleProposal(
      parent, *proposal, fabric));
  mlir::func::FuncOp function =
      child.structuredProgram.module().lookupSymbol<mlir::func::FuncOp>(
          "scf_kernel");
  if (!function)
    fail("SCF schedule child lost its function");
  std::size_t distributedRoots = 0;
  for (mlir::Operation &operation : function.getBody().front()) {
    auto root = llvm::dyn_cast<mlir::scf::ForOp>(&operation);
    if (!root)
      continue;
    ++distributedRoots;
    auto nested = llvm::dyn_cast<mlir::scf::ForOp>(&root.getBody()->front());
    if (!nested ||
        !llvm::hasSingleElement(nested.getBody()->without_terminator()) ||
        !llvm::isa<mlir::memref::StoreOp>(nested.getBody()->front()))
      fail("SCF statement-major schedule changed a distributed nest");
  }
  if (distributedRoots != 2)
    fail("SCF statement-major schedule did not materialize exact fission");
}

void imperfectGeneralScheduleMaterializes(
    const loom::fabric::FinalizedFabricRoot &fabric) {
  auto parent = parseProgram(R"mlir(
#layout = #dlti.dl_spec<#dlti.dl_entry<index, 64>>
module attributes {dlti.dl_spec = #layout} {
  func.func @imperfect(%a: memref<?xi32>, %b: memref<?xi32>,
                       %c: memref<?x?xi32>, %d: memref<?xi32>,
                       %m: index, %n: index) {
    %a0, %b0, %c0, %d0 = memref.distinct_objects %a, %b, %c, %d
        : memref<?xi32>, memref<?xi32>, memref<?x?xi32>, memref<?xi32>
    affine.for %i = 0 to %m {
      %outer = affine.load %a0[%i] : memref<?xi32>
      affine.for %j = 0 to %n {
        %lhs = affine.load %a0[%i] : memref<?xi32>
        %rhs = affine.load %b0[%j] : memref<?xi32>
        %sum = arith.addi %lhs, %rhs : i32
        affine.store %sum, %c0[%i, %j] : memref<?x?xi32>
      }
      %after = affine.load %d0[%i] : memref<?xi32>
    }
    return
  }
}
)mlir");
  const loom::frontend::StructuredEntityRef root = loopReferences(parent).first;
  auto analysis =
      take(loom::frontend::analyzeStructuredPolyhedralScop(parent, root));
  const auto *scop =
      std::get_if<loom::frontend::StructuredPolyhedralScopView>(&analysis);
  if (!scop || !scop->imperfectNest || scop->loopCount != 2 ||
      scop->maximumLoopDepth != 2 || scop->statements.size() != 6 ||
      scop->schedule.scheduleBandCount < 2 ||
      scop->schedule.form !=
          loom::frontend::StructuredPolyhedralScheduleForm::General ||
      llvm::none_of(
          scop->statements,
          [](const auto &statement) {
            return statement.domain.dimensions.size() == 1;
          }) ||
      llvm::none_of(
          scop->statements,
          [](const auto &statement) {
            return statement.domain.dimensions.size() == 2;
          }) ||
      llvm::none_of(
          scop->dependences,
          [](const auto &dependence) {
            return dependence.kind ==
                   loom::frontend::StructuredPolyhedralDependenceKind::
                       ScalarSsa;
          }))
    fail("imperfect fixture left its admitted general schedule");

  auto domain = take(
      loom::frontend::enumerateStructuredScheduleDecisions(parent, fabric, 2));
  auto proposal = llvm::find_if(domain.proposals, [&](const auto &candidate) {
    return candidate.decision().loop == root &&
           candidate.decision().kind ==
               loom::frontend::StructuredScheduleDecisionKind::
                   PolyhedralSchedule;
  });
  if (proposal == domain.proposals.end() ||
      llvm::any_of(domain.refusals, [&](const auto &refusal) {
        return refusal.loop == root &&
               refusal.kind == loom::frontend::StructuredScopRefusalKind::
                                   PolyhedralMaterializationUnavailable;
      }))
    fail("imperfect general schedule did not produce an exact proposal");

  auto child = take(loom::frontend::materializeStructuredScheduleProposal(
      parent, *proposal, fabric));
  auto direct = take(loom::frontend::materializeStructuredScheduleDecision(
      parent, proposal->decision()));
  if (child.structuredProgram.identity() != direct.structuredProgram.identity())
    fail("imperfect general schedule replay changed candidate identity");
  if (llvm::Error error = loom::frontend::verifyStructuredScheduleDerivation(
          parent, fabric, proposal->decision(), child.structuredProgram))
    fail("imperfect general derivation verification failed: " +
         llvm::toString(std::move(error)));

  mlir::func::FuncOp function =
      child.structuredProgram.module().lookupSymbol<mlir::func::FuncOp>(
          "imperfect");
  if (!function)
    fail("imperfect general child lost its function");
  std::size_t affineLoops = 0;
  std::size_t scheduledLoops = 0;
  std::size_t loads = 0;
  std::size_t stores = 0;
  std::size_t additions = 0;
  function.walk([&](mlir::Operation *operation) {
    affineLoops += llvm::isa<mlir::affine::AffineForOp>(operation);
    scheduledLoops += llvm::isa<mlir::scf::ForOp>(operation);
    loads += llvm::isa<mlir::memref::LoadOp>(operation);
    stores += llvm::isa<mlir::memref::StoreOp>(operation);
    additions += llvm::isa<mlir::arith::AddIOp>(operation) &&
                 operation->getResult(0).getType().isInteger(32);
  });
  if (affineLoops != 0 || scheduledLoops < 2 || loads != 4 || stores != 1 ||
      additions != 1)
    fail("imperfect general child changed its exact statement realization");
}

void generalAnalysisOwnsVectorDomainFallback(
    const loom::fabric::FinalizedFabricRoot &fabric) {
  auto affineSupport = parseProgram(R"mlir(
#identity = affine_map<(d0) -> (d0)>
module {
  func.func @affine_support(%input: memref<8xi32>) {
    %aligned = memref.assume_alignment %input, 32 : memref<8xi32>
    affine.for %i = 0 to 8 {
      %index = affine.apply #identity(%i)
      %value = affine.load %aligned[%index] : memref<8xi32>
    }
    return
  }
}
)mlir");
  const loom::frontend::StructuredEntityRef affineSupportRoot =
      affineRootReference(affineSupport);
  auto affineSupportDomain =
      take(loom::frontend::enumerateStructuredScheduleDecisions(affineSupport,
                                                                fabric, 1));
  if (affineSupportDomain.polyhedralScops.size() != 1 ||
      affineSupportDomain.polyhedralScops.front().root != affineSupportRoot ||
      llvm::none_of(affineSupportDomain.refusals, [&](const auto &refusal) {
        return refusal.loop == affineSupportRoot &&
               refusal.kind == loom::frontend::StructuredScopRefusalKind::
                                   UnsupportedOperation;
      }))
    fail("general analysis did not independently admit affine support");

  auto noVectorCoordinate = parseProgram(R"mlir(
module {
  func.func @no_vector_coordinate(%input: memref<5xi32>) {
    %aligned = memref.assume_alignment %input, 4 : memref<5xi32>
    affine.for %i = 0 to 5 {
      %value = affine.load %aligned[%i] : memref<5xi32>
    }
    return
  }
}
)mlir");
  const loom::frontend::StructuredEntityRef noVectorRoot =
      affineRootReference(noVectorCoordinate);
  auto noVectorDomain =
      take(loom::frontend::enumerateStructuredScheduleDecisions(
          noVectorCoordinate, fabric, 1));
  if (noVectorDomain.polyhedralScops.size() != 1 ||
      noVectorDomain.polyhedralScops.front().root != noVectorRoot ||
      llvm::any_of(noVectorDomain.proposals,
                   [&](const auto &proposal) {
                     return proposal.decision().loop == noVectorRoot &&
                            proposal.decision().kind ==
                                loom::frontend::StructuredScheduleDecisionKind::
                                    Vectorize;
                   }) ||
      llvm::none_of(noVectorDomain.refusals, [&](const auto &refusal) {
        return refusal.loop == noVectorRoot &&
               refusal.kind == loom::frontend::StructuredScopRefusalKind::
                                   AlignmentProofNotEstablished;
      }))
    fail("general analysis did not own the empty vector-coordinate fallback");

  auto partialVectorDomain = parseProgram(R"mlir(
module {
  func.func @partial_vector_domain(%input: memref<8xi32>) {
    %aligned = memref.assume_alignment %input, 8 : memref<8xi32>
    affine.for %i = 0 to 8 {
      %value = affine.load %aligned[%i] : memref<8xi32>
    }
    return
  }
}
)mlir");
  const loom::frontend::StructuredEntityRef partialVectorRoot =
      affineRootReference(partialVectorDomain);
  auto partialDomain =
      take(loom::frontend::enumerateStructuredScheduleDecisions(
          partialVectorDomain, fabric, 1));
  if (llvm::none_of(partialDomain.proposals,
                    [&](const auto &proposal) {
                      const auto &decision = proposal.decision();
                      return decision.loop == partialVectorRoot &&
                             decision.vector &&
                             decision.vector->shape ==
                                 std::vector<std::uint64_t>{2};
                    }) ||
      llvm::none_of(partialDomain.refusals, [&](const auto &refusal) {
        return refusal.loop == partialVectorRoot &&
               refusal.kind == loom::frontend::StructuredScopRefusalKind::
                                   AlignmentProofNotEstablished;
      }))
    fail("an admitted vector factor hid another factor's incomplete proof");
}

/// Polyhedral tiling is the provider schedule tiled by one canonical factor of
/// the root's static trip count. Each proven factor is one coordinate whose
/// child is the exact sequential realization of the tiled relation, replays
/// through lineage, and preserves native observations; a factor outside the
/// enumerated domain is not a lineage.
void tiledPolyhedralSchedulesMaterializeAndReplay(
    const loom::fabric::FinalizedFabricRoot &fabric) {
  auto parent = parseProgram(R"mlir(
#layout = #dlti.dl_spec<#dlti.dl_entry<index, 64>>
module attributes {dlti.dl_spec = #layout} {
  memref.global @left : memref<8xi32> = dense<0>
  memref.global @right : memref<8xi32> = dense<0>
  llvm.mlir.global internal @observation(0 : i32) : i32

  llvm.func @entry() -> i32 {
    %left = memref.get_global @left : memref<8xi32>
    %right = memref.get_global @right : memref<8xi32>
    %left0, %right0 = memref.distinct_objects %left, %right
        : memref<8xi32>, memref<8xi32>
    %seven = arith.constant 7 : i32
    %eleven = arith.constant 11 : i32
    affine.for %i = 0 to 8 {
      affine.store %seven, %left0[%i] : memref<8xi32>
      affine.store %eleven, %right0[%i] : memref<8xi32>
    }
    %c5 = arith.constant 5 : index
    %lhs = memref.load %left0[%c5] : memref<8xi32>
    %rhs = memref.load %right0[%c5] : memref<8xi32>
    %sum = arith.addi %lhs, %rhs : i32
    %address = llvm.mlir.addressof @observation : !llvm.ptr
    llvm.store %sum, %address : i32, !llvm.ptr
    llvm.return %sum : i32
  }
}
)mlir");
  const loom::frontend::StructuredEntityRef root = affineRootReference(parent);
  const std::vector<std::uint64_t> factors = {2, 4};
  auto analysis = take(
      loom::frontend::analyzeStructuredPolyhedralScop(parent, root, factors));
  const auto *scop =
      std::get_if<loom::frontend::StructuredPolyhedralScopView>(&analysis);
  if (!scop || scop->tiledSchedules.size() != factors.size())
    fail("the provider did not freeze one tiled schedule per factor");
  for (auto [factor, tiled] : llvm::zip(factors, scop->tiledSchedules))
    if (tiled.factor != factor ||
        tiled.schedule.form !=
            loom::frontend::StructuredPolyhedralScheduleForm::General ||
        tiled.schedule.scheduleDimensionCount <=
            scop->schedule.scheduleDimensionCount ||
        tiled.schedule.statementSchedules.size() != scop->statements.size())
      fail("a tiled schedule lost its exact tile coordinates");

  auto domain = take(
      loom::frontend::enumerateStructuredScheduleDecisions(parent, fabric, 1));
  for (const std::uint64_t factor : factors) {
    auto proposal = llvm::find_if(domain.proposals, [&](const auto &candidate) {
      return candidate.decision().loop == root &&
             candidate.decision().kind ==
                 loom::frontend::StructuredScheduleDecisionKind::
                     PolyhedralSchedule &&
             candidate.decision().factor == factor;
    });
    if (proposal == domain.proposals.end())
      fail("enumeration did not propose polyhedral tile factor " +
           std::to_string(factor));
    auto encoded = take(
        loom::frontend::encodeStructuredScheduleDecision(proposal->decision()));
    if (!(take(loom::frontend::adoptStructuredScheduleDecision(encoded)) ==
          proposal->decision()))
      fail("a tiled polyhedral decision did not round-trip canonically");
    auto child = take(loom::frontend::materializeStructuredScheduleProposal(
        parent, *proposal, fabric));
    auto direct = take(loom::frontend::materializeStructuredScheduleDecision(
        parent, proposal->decision()));
    if (child.structuredProgram.identity() !=
        direct.structuredProgram.identity())
      fail("frozen and replayed tiled materialization differ");
    if (llvm::Error error = loom::frontend::verifyStructuredScheduleDerivation(
            parent, fabric, proposal->decision(), child.structuredProgram))
      fail("tiled polyhedral lineage replay failed: " +
           llvm::toString(std::move(error)));
    mlir::LLVM::LLVMFuncOp function =
        child.structuredProgram.module().lookupSymbol<mlir::LLVM::LLVMFuncOp>(
            "entry");
    if (!function)
      fail("tiled polyhedral child lost its entry function");
    std::size_t affineLoops = 0;
    std::size_t scheduledLoops = 0;
    std::size_t maximumDepth = 0;
    std::size_t stores = 0;
    function.walk([&](mlir::Operation *operation) {
      affineLoops += llvm::isa<mlir::affine::AffineForOp>(operation);
      if (llvm::isa<mlir::scf::ForOp>(operation)) {
        ++scheduledLoops;
        std::size_t depth = 1;
        for (mlir::Operation *ancestor = operation->getParentOp();
             llvm::isa_and_nonnull<mlir::scf::ForOp>(ancestor);
             ancestor = ancestor->getParentOp())
          ++depth;
        maximumDepth = std::max(maximumDepth, depth);
      }
      stores += llvm::isa<mlir::memref::StoreOp>(operation);
    });
    if (affineLoops != 0 || scheduledLoops < 2 || maximumDepth < 2 ||
        stores != 2)
      fail("tiled polyhedral child changed its exact statement realization");
    requireEquivalentNativeObservations(parent, child.structuredProgram);
  }

  loom::frontend::StructuredScheduleDecision foreignFactor{
      root, loom::frontend::StructuredScheduleDecisionKind::PolyhedralSchedule,
      3, std::nullopt};
  auto foreignChild = take(loom::frontend::materializeStructuredScheduleDecision(
      parent, foreignFactor));
  llvm::Error foreignError = loom::frontend::verifyStructuredScheduleDerivation(
      parent, fabric, foreignFactor, foreignChild.structuredProgram);
  if (!foreignError)
    fail("lineage accepted a tile factor outside the enumerated domain");
  llvm::consumeError(std::move(foreignError));
}

void refusalDispositionPreservesIncompleteProofs() {
  using loom::frontend::StructuredScopRefusalDisposition;
  using loom::frontend::StructuredScopRefusalKind;
  if (loom::frontend::classifyStructuredScopRefusal(
          StructuredScopRefusalKind::ProviderScheduleNotEstablished) !=
          StructuredScopRefusalDisposition::IncompleteProof ||
      loom::frontend::classifyStructuredScopRefusal(
          StructuredScopRefusalKind::NestedControl) !=
          StructuredScopRefusalDisposition::OutsideAdmittedDomain ||
      loom::frontend::structuredScopRefusalKindSpelling(
          StructuredScopRefusalKind::ProviderScheduleNotEstablished) !=
          "provider_schedule_not_established")
    fail("SCoP refusal disposition lost its typed completeness boundary");
}

void statementMajorScheduleMaterializesAndReplays() {
  auto parent = parseProgram(R"mlir(
module {
  func.func @kernel(%state: memref<?x?xi32>, %m: index, %n: index,
                    %value: i32) {
    affine.for %i = 0 to %m {
      affine.for %j = 0 to %n {
        affine.store %value, %state[%i, %j] : memref<?x?xi32>
        %observed = affine.load %state[%i, %j] : memref<?x?xi32>
      }
    }
    return
  }
}
)mlir");
  const auto references = loopReferences(parent);
  const loom::frontend::StructuredEntityRef outer = references.first;
  const loom::frontend::StructuredEntityRef inner = references.second;
  auto analysis =
      take(loom::frontend::analyzeStructuredPolyhedralScop(parent, outer));
  const auto *scop =
      std::get_if<loom::frontend::StructuredPolyhedralScopView>(&analysis);
  if (!scop)
    fail("perfect symbolic SCoP was not admitted");
  if (scop->imperfectNest || scop->loopCount != 2 ||
      scop->maximumLoopDepth != 2 || scop->statements.size() != 2 ||
      llvm::any_of(scop->dependences,
                   [](const auto &dependence) {
                     return dependence.kind ==
                            loom::frontend::StructuredPolyhedralDependenceKind::
                                ScalarSsa;
                   }) ||
      scop->schedule.form !=
          loom::frontend::StructuredPolyhedralScheduleForm::StatementMajor)
    fail("provider schedule left the closed distribution form: form=" +
         std::to_string(static_cast<std::uint32_t>(scop->schedule.form)) +
         " dependences=" + std::to_string(scop->dependences.size()));

  llvm::SmallString<128> directory;
  if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
          "loom-polyhedral-schedule", directory))
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
  auto domain = take(
      loom::frontend::enumerateStructuredScheduleDecisions(parent, fabric, 2));
  auto proposal = llvm::find_if(domain.proposals, [](const auto &candidate) {
    return candidate.decision().kind ==
           loom::frontend::StructuredScheduleDecisionKind::PolyhedralSchedule;
  });
  if (proposal == domain.proposals.end() ||
      domain.polyhedralScops.size() != 1 ||
      domain.inspectedDecisionCoordinates != 1 ||
      domain.inspectedPolyhedralDependenceQueries !=
          scop->dependenceQueryCount ||
      llvm::any_of(domain.refusals, [&](const auto &refusal) {
        return refusal.loop == outer &&
               refusal.kind == loom::frontend::StructuredScopRefusalKind::
                                   PolyhedralMaterializationUnavailable;
      }))
    fail("production enumeration did not admit the exact provider schedule: " +
         std::to_string(proposal != domain.proposals.end()) + "," +
         std::to_string(domain.polyhedralScops.size()) + "," +
         std::to_string(domain.inspectedDecisionCoordinates) + "," +
         std::to_string(domain.inspectedPolyhedralDependenceQueries) + "," +
         std::to_string(scop->dependenceQueryCount) + "," +
         std::to_string(domain.refusals.size()));
  auto encoded = take(
      loom::frontend::encodeStructuredScheduleDecision(proposal->decision()));
  auto adopted = take(loom::frontend::adoptStructuredScheduleDecision(encoded));
  if (!(adopted == proposal->decision()))
    fail("polyhedral schedule decision did not round-trip canonically");

  auto child = take(loom::frontend::materializeStructuredScheduleProposal(
      parent, *proposal, fabric));
  auto direct = take(loom::frontend::materializeStructuredScheduleDecision(
      parent, proposal->decision()));
  if (child.structuredProgram.identity() != direct.structuredProgram.identity())
    fail("frozen and replayed polyhedral materialization differ");
  if (llvm::Error error = loom::frontend::verifyStructuredScheduleDerivation(
          parent, fabric, proposal->decision(), child.structuredProgram))
    fail("exact polyhedral lineage replay failed: " +
         llvm::toString(std::move(error)));

  auto parentReference =
      take(loom::frontend::publishStructuredProgram(parent, store));
  auto inputs = take(loom::dse::bindStructuredScheduleCandidateGeneratorInputs(
      {parentReference}, fabric.reference()));
  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.schedule.scopeExpansionLimit = 2;
  auto config =
      take(loom::dse::projectResolvedStructuredScheduleGeneratorConfigView(
          resolved,
          loom::dse::StructuredScheduleGenerationIntent::
              ForbidLogicalThreadDomain,
          4));
  auto binding = take(
      loom::dse::resolveStructuredScheduleCandidateGeneratorBinding(config));
  auto generated =
      take(loom::dse::invokeCandidateGenerator(inputs, binding, store, blobs));
  const auto *completed =
      std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
          &generated.outcome);
  if (!completed || completed->outputBindings.size() != 1 ||
      completed->outputBindings.front().artifacts.size() != 2 ||
      completed->lineageEdges.size() != 1 || generated.workSummary.size() != 5)
    fail("production generator did not publish one polyhedral child");
  const loom::dse::CandidateGeneratorLineageEdge &edge =
      completed->lineageEdges.front();
  auto lineageDecision =
      take(loom::frontend::adoptStructuredScheduleDecision(edge.ownerPayload));
  if (!(lineageDecision == proposal->decision()) ||
      edge.kind !=
          loom::dse::CandidateGeneratorLineageEdgeKind::CandidateDecision ||
      edge.parents.size() != 2 ||
      !llvm::is_contained(edge.parents, parentReference) ||
      !llvm::is_contained(edge.parents, fabric.reference()) ||
      edge.output.artifact != child.structuredProgram.identity() ||
      !llvm::is_contained(completed->outputBindings.front().artifacts,
                          edge.output))
    fail("production generator changed the exact polyhedral lineage");
  auto generatedChild =
      take(loom::frontend::importStructuredProgram(edge.output, store));
  if (generatedChild.identity() != child.structuredProgram.identity() ||
      generated.workSummary[0].planned != domain.inspectedLoopScopes ||
      generated.workSummary[0].consumed != domain.inspectedLoopScopes ||
      generated.workSummary[1].planned != 1 ||
      generated.workSummary[1].consumed != 1 ||
      generated.workSummary[2].planned != domain.inspectedDecisionCoordinates ||
      generated.workSummary[2].consumed !=
          domain.inspectedDecisionCoordinates ||
      generated.workSummary[3].planned != 1 ||
      generated.workSummary[3].consumed != 1 ||
      generated.workSummary[4].planned !=
          domain.inspectedPolyhedralDependenceQueries ||
      generated.workSummary[4].consumed !=
          domain.inspectedPolyhedralDependenceQueries)
    fail("production generator lost its exact polyhedral work ledger");

  mlir::func::FuncOp function =
      child.structuredProgram.module().lookupSymbol<mlir::func::FuncOp>(
          "kernel");
  if (!function)
    fail("polyhedral schedule child lost its function");
  std::size_t outerLoops = 0;
  std::size_t singleStatementNests = 0;
  std::vector<std::uint32_t> statementOrder;
  for (mlir::Operation &operation : function.getBody().front()) {
    auto root = llvm::dyn_cast<mlir::affine::AffineForOp>(&operation);
    if (!root)
      continue;
    ++outerLoops;
    mlir::affine::AffineForOp nested;
    for (mlir::Operation &bodyOperation : root.getBody()->without_terminator())
      nested = llvm::dyn_cast<mlir::affine::AffineForOp>(&bodyOperation);
    if (nested &&
        llvm::hasSingleElement(nested.getBody()->without_terminator())) {
      ++singleStatementNests;
      mlir::Operation &statement = nested.getBody()->front();
      if (llvm::isa<mlir::affine::AffineStoreOp>(statement))
        statementOrder.push_back(0);
      else if (llvm::isa<mlir::affine::AffineLoadOp>(statement))
        statementOrder.push_back(1);
    }
  }
  if (outerLoops != 2 || singleStatementNests != 2 ||
      statementOrder != std::vector<std::uint32_t>{0, 1})
    fail("statement-major schedule changed exact fission order");

  loom::frontend::StructuredScheduleDecision forged = proposal->decision();
  forged.loop = inner;
  llvm::Error forgedError = loom::frontend::verifyStructuredScheduleDerivation(
      parent, fabric, forged, child.structuredProgram);
  if (!forgedError)
    fail("lineage accepted a nested-root polyhedral forgery");
  llvm::consumeError(std::move(forgedError));

  auto scalarExpansion = parseProgram(R"mlir(
module {
  func.func @scalar_expansion(%input: memref<?x?xi32>,
                              %output: memref<?x?xi32>,
                              %m: index, %n: index) {
    %input0, %output0 = memref.distinct_objects %input, %output
        : memref<?x?xi32>, memref<?x?xi32>
    affine.for %i = 0 to %m {
      affine.for %j = 0 to %n {
        %value = affine.load %input0[%i, %j] : memref<?x?xi32>
        affine.store %value, %output0[%i, %j] : memref<?x?xi32>
      }
    }
    return
  }
}
)mlir");
  const loom::frontend::StructuredEntityRef scalarRoot =
      loopReferences(scalarExpansion).first;
  auto scalarAnalysis = take(loom::frontend::analyzeStructuredPolyhedralScop(
      scalarExpansion, scalarRoot));
  const auto *scalarScop =
      std::get_if<loom::frontend::StructuredPolyhedralScopView>(
          &scalarAnalysis);
  if (!scalarScop ||
      scalarScop->schedule.form !=
          loom::frontend::StructuredPolyhedralScheduleForm::SourceOrder ||
      llvm::none_of(scalarScop->dependences, [](const auto &dependence) {
        return dependence.kind ==
               loom::frontend::StructuredPolyhedralDependenceKind::ScalarSsa;
      }))
    fail("scalar precedence fixture left its source-order provider form: " +
         std::to_string(
             scalarScop ? static_cast<std::uint32_t>(scalarScop->schedule.form)
                        : 99));
  auto scalarDomain = take(loom::frontend::enumerateStructuredScheduleDecisions(
      scalarExpansion, fabric, 2));
  if (llvm::any_of(scalarDomain.proposals,
                   [&](const auto &candidate) {
                     return candidate.decision().loop == scalarRoot &&
                            candidate.decision().kind ==
                                loom::frontend::StructuredScheduleDecisionKind::
                                    PolyhedralSchedule;
                   }) ||
      llvm::any_of(scalarDomain.refusals, [&](const auto &refusal) {
        return refusal.loop == scalarRoot &&
               refusal.kind == loom::frontend::StructuredScopRefusalKind::
                                   PolyhedralMaterializationUnavailable;
      }))
    fail("source-order scalar precedence acquired a false transform refusal");
  scfStatementMajorScheduleMaterializes(fabric);
  tiledPolyhedralSchedulesMaterializeAndReplay(fabric);
  imperfectGeneralScheduleMaterializes(fabric);
  generalAnalysisOwnsVectorDomainFallback(fabric);
  llvm::sys::fs::remove_directories(directory);
}

} // namespace

int main() {
  refusalDispositionPreservesIncompleteProofs();
  physicalLayoutInjectivityIsRequired();
  localDivisionSchedulesHaveIndependentSemantics();
  statementMajorScheduleMaterializesAndReplays();
  return EXIT_SUCCESS;
}
