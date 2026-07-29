#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Common/ResolvedConfig.h"
#include "DSE/PreMappingExploration.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <system_error>
#include <variant>

namespace {

[[noreturn]] void fail(const std::string &message) {
  llvm::errs() << "structuredOwnershipAccounting: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

std::unique_ptr<llvm::Module> parseModule(llvm::LLVMContext &context) {
  constexpr llvm::StringLiteral source = R"llvm(
target datalayout = "e-m:e-p:64:64-i64:64-n32:64-S128"
target triple = "riscv64-unknown-unknown"

declare i32 @external(i32)

define void @kernel(ptr %base, i64 %index) {
entry:
  %address = getelementptr float, ptr %base, i64 %index
  %value = load float, ptr %address, align 4
  %result = fadd float %value, 1.000000e+00
  store float %result, ptr %address, align 4
  ret void
}

define i32 @main(ptr %base, i64 %index) {
entry:
  call void @kernel(ptr %base, i64 %index)
  ret i32 0
}
)llvm";
  llvm::SMDiagnostic diagnostic;
  auto buffer = llvm::MemoryBuffer::getMemBuffer(source, "<accounting>");
  auto module = llvm::parseIR(buffer->getMemBufferRef(), diagnostic, context);
  if (!module) {
    std::string message;
    llvm::raw_string_ostream stream(message);
    diagnostic.print("structuredOwnershipAccounting", stream);
    fail(stream.str());
  }
  return module;
}

std::unique_ptr<llvm::Module> parseEmptyScopeModule(llvm::LLVMContext &context) {
  constexpr llvm::StringLiteral source = R"llvm(
target datalayout = "e-m:e-p:64:64-i64:64-n32:64-S128"
target triple = "riscv64-unknown-unknown"

define i32 @main(i1 %condition) {
entry:
  br i1 %condition, label %empty, label %merge
empty:
  br label %merge
merge:
  ret i32 0
}
)llvm";
  llvm::SMDiagnostic diagnostic;
  auto buffer = llvm::MemoryBuffer::getMemBuffer(source, "<empty-scope>");
  auto module = llvm::parseIR(buffer->getMemBufferRef(), diagnostic, context);
  if (!module) {
    std::string message;
    llvm::raw_string_ostream stream(message);
    diagnostic.print("structuredOwnershipAccounting", stream);
    fail(stream.str());
  }
  return module;
}

loom::dse::CompletedPreMappingSelection
explore(const loom::fabric::FinalizedFabricRoot &fabric,
        const loom::ArtifactStore &store, std::uint32_t workers) {
  llvm::LLVMContext context;
  loom::dse::PreMappingExplorationOptions options{
      {},
      {{},
       {loom::evaluation::MetricRequestOrdinal(0),
        loom::dse::ObjectiveDirection::Minimize, 1},
       workers}};
  auto outcome = take(loom::dse::exploreLlvmModuleToPreMapping(
      parseModule(context), fabric, loom::defaultResolvedConfig(), options,
      store));
  auto *completed =
      std::get_if<loom::dse::CompletedPreMappingSelection>(&outcome);
  if (!completed)
    fail("finite candidate accounting did not complete selection");
  return std::move(*completed);
}

void requireCompleteAccounting(
    const loom::dse::CompletedPreMappingSelection &selection) {
  if (selection.dispositions.size() != 3)
    fail("candidate domain included a declaration or omitted an attempt");
  bool sawScopeRejection = false;
  bool sawDecisionRejection = false;
  bool sawSuccessfulDecision = false;
  for (const loom::dse::StructuredOwnershipCandidateDisposition &disposition :
       selection.dispositions) {
    const auto *rejection =
        std::get_if<loom::dse::StructuredOwnershipCandidateRejectionRecord>(
            &disposition.result);
    const auto *candidate =
        std::get_if<loom::ArtifactRootReference>(&disposition.result);
    if (!disposition.coordinate.decision) {
      if (!rejection ||
          rejection->kind !=
              loom::frontend::SpatialOwnershipCandidateRejectionKind::
                  NonFinalizable ||
          rejection->message.find("unresolved nested call") ==
              std::string::npos)
        fail("whole-callable rejection lost its typed scope disposition");
      sawScopeRejection = true;
      continue;
    }
    if (disposition.coordinate.decision->canonicalIndexWidth == 32) {
      if (!rejection ||
          rejection->kind !=
              loom::frontend::SpatialOwnershipCandidateRejectionKind::
                  NonFinalizable ||
          rejection->message.find("cannot prove") == std::string::npos)
        fail("unsafe 32-bit narrowing lost its typed decision disposition");
      sawDecisionRejection = true;
      continue;
    }
    if (disposition.coordinate.decision->canonicalIndexWidth == 64) {
      if (!candidate)
        fail("valid 64-bit ownership decision did not retain its child");
      sawSuccessfulDecision = true;
    }
  }
  if (!sawScopeRejection || !sawDecisionRejection || !sawSuccessfulDecision)
    fail("candidate domain accounting was incomplete");
}

void requireEmptyScopeIsCandidateRejection(
    const loom::fabric::FinalizedFabricRoot &fabric) {
  llvm::LLVMContext context;
  auto structured = take(loom::frontend::raiseLlvmModuleToStructured(
      parseEmptyScopeModule(context), fabric));
  auto domain = take(loom::frontend::enumerateSpatialOwnershipScopeDomain(
      structured.structuredProgram));

  bool sawEmptyScope = false;
  for (const loom::frontend::SpatialOwnershipScopeDomainEntry &entry : domain) {
    const auto *scope =
        std::get_if<loom::frontend::SpatialOwnershipScope>(&entry);
    if (!scope)
      continue;
    auto decisions = take(
        loom::frontend::enumerateSpatialOwnershipDecisionDomain(
            structured.structuredProgram, scope->selection));
    for (const loom::frontend::SpatialOwnershipDecisionPoint &decision :
         decisions) {
      auto candidate = loom::frontend::materializeSpatialOwnershipDecision(
          structured.structuredProgram, *scope, decision, fabric);
      if (candidate)
        continue;
      bool classified = false;
      llvm::Error unhandled = llvm::handleErrors(
          candidate.takeError(),
          [&](const loom::frontend::SpatialOwnershipCandidateRejection &error) {
            classified = true;
            if (error.kind() ==
                    loom::frontend::SpatialOwnershipCandidateRejectionKind::
                        NonFinalizable &&
                error.message().find("no SpatialCore workload") !=
                    std::string::npos)
              sawEmptyScope = true;
          });
      if (unhandled)
        fail("empty Spatial scope escaped candidate rejection: " +
             llvm::toString(std::move(unhandled)));
      if (!classified)
        fail("empty Spatial scope failed without a typed rejection");
    }
  }
  if (!sawEmptyScope)
    fail("empty Spatial scope did not produce a NonFinalizable disposition");
}

} // namespace

int main() {
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-ownership-accounting", directory);
  if (error)
    fail("cannot create artifact store directory: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));

  requireEmptyScopeIsCandidateRejection(design.roots().front());

  auto serial = explore(design.roots().front(), store, 1);
  requireCompleteAccounting(serial);
  auto parallel = explore(design.roots().front(), store, 2);
  requireCompleteAccounting(parallel);
  if (serial.dispositions != parallel.dispositions)
    fail("candidate worker count changed the disposition sequence");

  error = llvm::sys::fs::remove_directories(directory);
  if (error)
    fail("cannot remove artifact store directory: " + error.message());
  return EXIT_SUCCESS;
}
