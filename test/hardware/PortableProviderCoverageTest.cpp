#include "Hardware/RTL/PortableProviders.h"
#include "Hardware/RTL/Providers/FixedVectorFloatFma.h"
#include "Hardware/RTL/Providers/FixedVectorIntegerAddSub.h"
#include "Hardware/RTL/Providers/FixedVectorIntegerCompareMinMax.h"
#include "Hardware/RTL/Providers/FixedVectorIntegerMultiply.h"
#include "Hardware/RTL/Providers/FixedVectorPackUnpack.h"
#include "Hardware/RTL/Providers/FixedVectorShuffle.h"
#include "Hardware/RTL/Providers/FixedVectorSliceAlignMerge.h"
#include "Hardware/RTL/Providers/FixedVectorValueSelect.h"
#include "Hardware/RTL/Providers/FloatAddSub.h"
#include "Hardware/RTL/Providers/FloatCompareMinMax.h"
#include "Hardware/RTL/Providers/FloatConversions.h"
#include "Hardware/RTL/Providers/FloatDivideRemainder.h"
#include "Hardware/RTL/Providers/FloatMultiply.h"
#include "Hardware/RTL/Providers/FloatSign.h"
#include "Hardware/RTL/Providers/IntegerCountZeros.h"
#include "Hardware/RTL/Providers/IntegerLogic.h"
#include "Hardware/RTL/Providers/IntegerSaturatingAddSub.h"
#include "Hardware/RTL/Providers/IntegerShift.h"
#include "Hardware/RTL/Providers/LoopCarry.h"
#include "Hardware/RTL/Providers/LoopGate.h"
#include "Hardware/RTL/Providers/LoopInvariant.h"
#include "Hardware/RTL/Providers/LoopStream.h"
#include "Hardware/RTL/Providers/MathRoot.h"
#include "Hardware/RTL/Providers/ScalarBitReinterpret.h"
#include "Hardware/RTL/Providers/ScalarFloatFma.h"
#include "Hardware/RTL/Providers/ScalarIntegerAddSub.h"
#include "Hardware/RTL/Providers/ScalarIntegerCast.h"
#include "Hardware/RTL/Providers/ScalarIntegerCompareMinMax.h"
#include "Hardware/RTL/Providers/ScalarIntegerMultiply.h"
#include "Hardware/RTL/Providers/ScalarSignedIntegerDivRem.h"
#include "Hardware/RTL/Providers/ScalarUnsignedIntegerDivRem.h"
#include "Hardware/RTL/Providers/ScalarValueSelect.h"
#include "Hardware/RTL/Providers/TokenConstantSync.h"
#include "Hardware/RTL/Providers/TokenMuxDemux.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdlib>
#include <utility>
#include <vector>

namespace {

using loom::hardware::rtl::BackendRecipeKey;
using loom::hardware::rtl::FabricOperationProviderCoverage;
using loom::hardware::rtl::FabricOperationProviderRegistry;

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "portable provider coverage test: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(bool condition, llvm::StringRef message) {
  if (!condition)
    fail(message);
}

bool sameCoverage(llvm::ArrayRef<FabricOperationProviderCoverage> lhs,
                  llvm::ArrayRef<FabricOperationProviderCoverage> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (std::size_t index = 0; index < lhs.size(); ++index)
    if (lhs[index].implementationFamily != rhs[index].implementationFamily ||
        lhs[index].recipes != rhs[index].recipes)
      return false;
  return true;
}

void registerIndependentProviders(FabricOperationProviderRegistry &registry) {
  auto requireRegistration = [](llvm::Error error) {
    if (error)
      fail(llvm::toString(std::move(error)));
  };
  requireRegistration(
      loom::hardware::rtl::registerPortableScalarIntegerAddSubProvider(
          registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableIntegerLogicProviders(registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableIntegerShiftProviders(registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableScalarIntegerCompareMinMaxProvider(
          registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableScalarValueSelectProvider(registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableScalarIntegerCastProvider(registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableScalarBitReinterpretProvider(
          registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableFloatSignProviders(registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableFloatAddSubProviders(registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableFloatCompareMinMaxProviders(
          registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableFloatConversionProviders(registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableFloatMultiplyProviders(registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableFloatDivideRemainderProviders(
          registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableScalarIntegerMultiplyProvider(
          registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableScalarFloatFmaProvider(registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableFixedVectorFloatFmaProvider(
          registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableLoopCarryProvider(registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableLoopInvariantProvider(registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableLoopGateProvider(registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableLoopStreamProvider(registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableMathRootProviders(registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableFixedVectorIntegerAddSubProvider(
          registry));
  requireRegistration(
      loom::hardware::rtl::
          registerPortableFixedVectorIntegerCompareMinMaxProvider(registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableFixedVectorValueSelectProvider(
          registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableFixedVectorIntegerMultiplyProvider(
          registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableFixedVectorPackProvider(registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableFixedVectorUnpackProvider(registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableScalarSignedIntegerDivRemProvider(
          registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableScalarUnsignedIntegerDivRemProvider(
          registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableIntegerSaturatingAddSubProviders(
          registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableIntegerCountZerosProviders(
          registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableFixedVectorSliceAlignMergeProvider(
          registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableFixedVectorShuffleProvider(
          registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableTokenConstantSyncProviders(
          registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableTokenMuxDemuxProviders(registry));
}

std::vector<::fabric::ImplementationFamilyId>
registeredFamilies(llvm::ArrayRef<FabricOperationProviderCoverage> coverage) {
  std::vector<::fabric::ImplementationFamilyId> families;
  for (const FabricOperationProviderCoverage &entry : coverage)
    if (!entry.recipes.empty())
      families.push_back(entry.implementationFamily);
  return families;
}

void aggregateRegistrationIsTheCoverageAuthority() {
  FabricOperationProviderRegistry registry;
  if (llvm::Error error =
          loom::hardware::rtl::registerPortableOperationProviders(registry))
    fail(llvm::toString(std::move(error)));

  const std::vector<FabricOperationProviderCoverage> coverage =
      registry.coverage();
  require(coverage.size() == ::fabric::implementationFamilyCount(),
          "coverage is not a total projection of the generated registry");
  for (const FabricOperationProviderCoverage &entry : coverage) {
    require(entry.recipes.size() <= 1,
            "portable assembly registered more than one recipe for a family");
    if (!entry.recipes.empty())
      require(entry.recipes.front() == BackendRecipeKey::PortableSystemVerilog,
              "portable assembly registered a native recipe");
  }

  const std::vector<::fabric::ImplementationFamilyId> expectedFamilies = {
      ::fabric::ImplementationFamilyId::ScalarIntegerAddSub,
      ::fabric::ImplementationFamilyId::ScalarIntegerLogic,
      ::fabric::ImplementationFamilyId::ScalarIntegerShift,
      ::fabric::ImplementationFamilyId::ScalarIntegerCompareMinMax,
      ::fabric::ImplementationFamilyId::ScalarValueSelect,
      ::fabric::ImplementationFamilyId::ScalarIntegerCast,
      ::fabric::ImplementationFamilyId::ScalarBitReinterpret,
      ::fabric::ImplementationFamilyId::ScalarFloatSign,
      ::fabric::ImplementationFamilyId::ScalarFloatAddSub,
      ::fabric::ImplementationFamilyId::ScalarFloatCompareMinMax,
      ::fabric::ImplementationFamilyId::ScalarFloatWidthCast,
      ::fabric::ImplementationFamilyId::ScalarIntegerToFloat,
      ::fabric::ImplementationFamilyId::ScalarFloatToInteger,
      ::fabric::ImplementationFamilyId::ScalarIntegerMultiply,
      ::fabric::ImplementationFamilyId::ScalarFloatMultiply,
      ::fabric::ImplementationFamilyId::ScalarFloatFma,
      ::fabric::ImplementationFamilyId::LoopStream,
      ::fabric::ImplementationFamilyId::LoopCarry,
      ::fabric::ImplementationFamilyId::LoopInvariant,
      ::fabric::ImplementationFamilyId::LoopGate,
      ::fabric::ImplementationFamilyId::FixedVectorIntegerAddSub,
      ::fabric::ImplementationFamilyId::FixedVectorIntegerLogic,
      ::fabric::ImplementationFamilyId::FixedVectorIntegerShift,
      ::fabric::ImplementationFamilyId::FixedVectorIntegerCompareMinMax,
      ::fabric::ImplementationFamilyId::FixedVectorValueSelect,
      ::fabric::ImplementationFamilyId::FixedVectorIntegerMultiply,
      ::fabric::ImplementationFamilyId::FixedVectorFloatSign,
      ::fabric::ImplementationFamilyId::FixedVectorFloatAddSub,
      ::fabric::ImplementationFamilyId::FixedVectorFloatCompareMinMax,
      ::fabric::ImplementationFamilyId::FixedVectorFloatMultiply,
      ::fabric::ImplementationFamilyId::FixedVectorFloatFma,
      ::fabric::ImplementationFamilyId::FixedVectorPack,
      ::fabric::ImplementationFamilyId::FixedVectorUnpack,
      ::fabric::ImplementationFamilyId::TokenConstant,
      ::fabric::ImplementationFamilyId::TokenSync,
      ::fabric::ImplementationFamilyId::TokenMux,
      ::fabric::ImplementationFamilyId::TokenDemux,
      ::fabric::ImplementationFamilyId::ScalarSignedIntegerDivRem,
      ::fabric::ImplementationFamilyId::ScalarUnsignedIntegerDivRem,
      ::fabric::ImplementationFamilyId::ScalarFloatDivide,
      ::fabric::ImplementationFamilyId::ScalarFloatRemainder,
      ::fabric::ImplementationFamilyId::ScalarMathSqrt,
      ::fabric::ImplementationFamilyId::ScalarMathRsqrt,
      ::fabric::ImplementationFamilyId::ScalarIntegerSaturatingAddSub,
      ::fabric::ImplementationFamilyId::FixedVectorIntegerSaturatingAddSub,
      ::fabric::ImplementationFamilyId::ScalarIntegerCountZeros,
      ::fabric::ImplementationFamilyId::FixedVectorIntegerCountZeros,
      ::fabric::ImplementationFamilyId::FixedVectorSliceAlignMerge,
      ::fabric::ImplementationFamilyId::FixedVectorShuffle,
  };
  require(registeredFamilies(coverage) == expectedFamilies,
          "portable entry points do not own the expected family set");

  FabricOperationProviderRegistry independentRegistry;
  registerIndependentProviders(independentRegistry);
  require(sameCoverage(coverage, independentRegistry.coverage()),
          "aggregate registration omitted an existing provider entry point");
}

void aggregateRegistrationIsTransactional() {
  FabricOperationProviderRegistry registry;
  if (llvm::Error error =
          loom::hardware::rtl::registerPortableLoopCarryProvider(registry))
    fail(llvm::toString(std::move(error)));
  const std::vector<FabricOperationProviderCoverage> before =
      registry.coverage();

  llvm::Error error =
      loom::hardware::rtl::registerPortableOperationProviders(registry);
  require(static_cast<bool>(error),
          "aggregate registration accepted a duplicate provider");
  llvm::consumeError(std::move(error));
  require(sameCoverage(before, registry.coverage()),
          "failed aggregate registration changed the registry");
}

} // namespace

int main() {
  aggregateRegistrationIsTheCoverageAuthority();
  aggregateRegistrationIsTransactional();
  return EXIT_SUCCESS;
}
