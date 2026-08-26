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

#include "Dataflow/IR/DataflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
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

#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
                    mlir::memref::MemRefDialect>();
    registry.insert<mlir::scf::SCFDialect>();
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
  imperfectGeneralScheduleMaterializes(fabric);
  llvm::sys::fs::remove_directories(directory);
}

} // namespace

int main() {
  statementMajorScheduleMaterializesAndReplays();
  return EXIT_SUCCESS;
}
