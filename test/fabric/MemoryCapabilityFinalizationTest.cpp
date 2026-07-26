#include "Fabric/IR/MemoryCapabilityFinalization.h"

#include "Fabric/IR/FabricDialect.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <optional>
#include <string>

using namespace fabric;

namespace {

[[noreturn]] void fail(const char *test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void expectInvalid(const char *test, llvm::Error error,
                   MemoryCapabilityFinalizationReason expectedReason,
                   const char *expectedText) {
  if (!error)
    fail(test, "accepted an unavailable persistent contract");

  std::optional<MemoryCapabilityFinalizationReason> observedReason;
  std::string text;
  llvm::raw_string_ostream stream(text);
  llvm::handleAllErrors(std::move(error),
                        [&](const MemoryCapabilityFinalizationError &failure) {
                          observedReason = failure.reason();
                          failure.log(stream);
                        });
  stream.flush();

  if (observedReason != expectedReason)
    fail(test, "received a different typed finalization failure");
  if (text != expectedText)
    fail(test, "diagnostic differs: " + text);
}

MemoryContractAttr contract(mlir::MLIRContext &context, MemoryEngineAttr engine,
                            LocalMemoryServiceAttr localService) {
  mlir::DenseI32ArrayAttr noEndpoints =
      mlir::DenseI32ArrayAttr::get(&context, {});
  return MemoryContractAttr::get(&context, engine, localService, noEndpoints,
                                 noEndpoints);
}

} // namespace

int main() {
  mlir::DialectRegistry registry;
  registry.insert<FabricDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  if (llvm::Error error = validateMemoryCapabilityFinalization({}, {}))
    fail("empty occurrence", llvm::toString(std::move(error)));

  MemoryEngineAttr engine = MemoryEngineAttr::get(&context, Schedule::Spatial);
  expectInvalid(
      "incomplete engine",
      validateMemoryCapabilityFinalization(contract(context, engine, {}), {}),
      MemoryCapabilityFinalizationReason::MissingMemoryCapabilityContract,
      "Invalid(missing-memory-capability-contract)");

  MemoryServiceContractAttr serviceContract =
      MemoryServiceContractAttr::get(&context, MemoryServiceBehavior::Storage);
  LocalMemoryServiceAttr local =
      LocalMemoryServiceAttr::get(&context, 4096, serviceContract);
  expectInvalid(
      "local service",
      validateMemoryCapabilityFinalization(contract(context, {}, local), {}),
      MemoryCapabilityFinalizationReason::MissingMemoryServiceContract,
      "Invalid(missing-memory-service-contract)");
  expectInvalid(
      "engine with local service",
      validateMemoryCapabilityFinalization(contract(context, engine, local),
                                           {}),
      MemoryCapabilityFinalizationReason::MissingMemoryServiceContract,
      "Invalid(missing-memory-service-contract)");
  return EXIT_SUCCESS;
}
