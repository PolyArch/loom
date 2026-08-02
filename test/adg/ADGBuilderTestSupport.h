#ifndef LOOM_TEST_ADG_ADGBUILDERTESTSUPPORT_H
#define LOOM_TEST_ADG_ADGBUILDERTESTSUPPORT_H

#include "ADG/Builder.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <utility>

namespace mlir {
class MLIRContext;
}

namespace loom::adg::test {

[[noreturn]] void fail(llvm::StringRef test, const std::string &message);
void require(llvm::StringRef test, bool condition, llvm::StringRef message);

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

void expectError(llvm::StringRef test, llvm::Error error,
                 llvm::StringRef diagnostic);

template <typename T>
void expectError(llvm::StringRef test, llvm::Expected<T> value,
                 llvm::StringRef diagnostic) {
  if (value)
    fail(test, "accepted invalid ADG authoring");
  expectError(test, value.takeError(), diagnostic);
}

class TemporaryDirectory final {
public:
  explicit TemporaryDirectory(llvm::StringRef test);
  ~TemporaryDirectory();

  llvm::StringRef path() const { return path_; }

private:
  std::string test_;
  std::string path_;
};

std::uint64_t uniqueEntity(llvm::StringRef test,
                           const loom::fabric::FabricArtifactView &view,
                           loom::fabric::FabricEntityKind expectedKind);
std::uint64_t entityCount(const loom::fabric::FabricArtifactView &view,
                          loom::fabric::FabricEntityKind expectedKind);
loom::fabric::FabricFuTemplateRef
uniqueFuTemplate(llvm::StringRef test,
                 const loom::fabric::FabricArtifactView &view);

OperationCapabilitySpec
integerCapability(::fabric::ImplementationFamilyId family,
                  ::dataflow::OperationSchemaId operation,
                  const PortType &outputType);
::fabric::ResourceContract singleUseResourceContract(llvm::StringRef test);
loom::fabric::InstructionCoreArchitecturalContract
instructionArchitecture(llvm::StringRef test);
loom::fabric::InstructionCoreMicroarchitecturalRealization
inOrderMicroarchitecture(llvm::StringRef test);
loom::fabric::InstructionCoreMicroarchitecturalRealization
outOfOrderMicroarchitecture(llvm::StringRef test);
::fabric::MemoryServiceContractRecord
systemMemoryContract(llvm::StringRef test, mlir::MLIRContext &context);
loom::fabric::CanonicalServiceCapabilitySet
systemMemoryCapabilities(llvm::StringRef test,
                         loom::fabric::ServiceRateContractRecord serviceRate);

void runBuilderTests();
void runBuiltinTests();
void runTopologyTests();
void runConformanceAnchorTests();

void regularAndIrregularSpatialCoresFinalize();
void temporalResourceGrantFinalizes();
void publicMemoryLibraryBuildsHybridLocalMemories();
void builtinPresetsExpandThroughPublicBuilder();
void heterogeneousSystemFinalizes();

} // namespace loom::adg::test

#endif
