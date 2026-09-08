#include "StructuredScheduleTestSupport.h"

#include "Config/ResolvedConfig.h"
#include "DSE/CandidateGenerator.h"
#include "DSE/StructuredScheduleCandidateGenerator.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Frontend/IR/LoomDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <variant>

namespace loom::frontend::schedule_test {

[[noreturn]] void fail(const std::string &message) {
  llvm::errs() << "structuredScheduleGenerator: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

static mlir::MLIRContext &context() {
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

} // namespace loom::frontend::schedule_test
