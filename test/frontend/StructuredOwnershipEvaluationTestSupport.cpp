#include "StructuredOwnershipEvaluationTestSupport.h"

#include "DSE/PreMappingExploration.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <vector>

namespace loom::test::structured_ownership {

[[noreturn]] void fail(const std::string &message) {
  llvm::errs() << "structuredOwnershipEvaluation: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

bool hasGeneratorInvocation(
    const loom::dse::CompletedPreMappingSelection &selection,
    llvm::StringRef spelling) {
  for (const loom::dse::DsePlanGenerateInvocationRecords &planInvocation :
       selection.planGenerateInvocations) {
    const auto matches =
        [&](const loom::dse::GenerateInvocationRecord &record) {
          const loom::dse::CandidateGeneratorDescriptor *descriptor =
              record.generatorBinding.descriptorRef().descriptor();
          if (!descriptor)
            fail("pre-Mapping Generate provenance lost its exact descriptor");
          return descriptor->spelling == spelling;
        };
    if (llvm::any_of(planInvocation.completed(), matches) ||
        llvm::any_of(planInvocation.incomplete(), matches))
      return true;
  }
  return false;
}

loom::frontend::StructuredEntityRef
findCallable(const loom::frontend::StructuredProgramCandidate &candidate,
             llvm::StringRef name) {
  auto view = take(candidate.view());
  for (const loom::frontend::StructuredEntity &entity :
       view.entities(loom::frontend::StructuredEntityKind::Operation)) {
    auto function =
        llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(entity.operation);
    if (function && function.getSymName() == name)
      return entity.reference;
  }
  fail("callable is absent from the Structured Program: " + name.str());
}

loom::sim::RuntimeMemoryObject zeroedMemory(std::size_t byteCount) {
  return loom::sim::RuntimeMemoryObject{
      std::vector<loom::sim::SemanticMemoryByte>(
          byteCount, {loom::sim::SemanticState::Defined, std::uint8_t{0}})};
}

} // namespace loom::test::structured_ownership
