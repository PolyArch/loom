#include "Hardware/RTL/PortableProviders.h"
#include "Hardware/RTL/SpatialCoreImplementation.h"
#include "Hardware/RTL/Providers/FixedVectorFloatFma.h"
#include "Hardware/RTL/Providers/FixedVectorIntegerAddSub.h"
#include "Hardware/RTL/Providers/FixedVectorIntegerCompareMinMax.h"
#include "Hardware/RTL/Providers/FixedVectorIntegerMultiply.h"
#include "Hardware/RTL/Providers/FixedVectorPackUnpack.h"
#include "Hardware/RTL/Providers/FixedVectorParallelizeSerialize.h"
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
#include "Hardware/RTL/Providers/MathErf.h"
#include "Hardware/RTL/Providers/MathExponential.h"
#include "Hardware/RTL/Providers/MathLogarithm.h"
#include "Hardware/RTL/Providers/MathPower.h"
#include "Hardware/RTL/Providers/MathRoot.h"
#include "Hardware/RTL/Providers/MathRounding.h"
#include "Hardware/RTL/Providers/ScalarBitReinterpret.h"
#include "Hardware/RTL/Providers/ScalarFloatFma.h"
#include "Hardware/RTL/Providers/ScalarIntegerAddSub.h"
#include "Hardware/RTL/Providers/ScalarIntegerCast.h"
#include "Hardware/RTL/Providers/ScalarIntegerCompareMinMax.h"
#include "Hardware/RTL/Providers/ScalarIntegerMultiply.h"
#include "Hardware/RTL/Providers/ScalarMathHyperbolic.h"
#include "Hardware/RTL/Providers/ScalarMathTrigonometric.h"
#include "Hardware/RTL/Providers/ScalarSignedIntegerDivRem.h"
#include "Hardware/RTL/Providers/ScalarUnsignedIntegerDivRem.h"
#include "Hardware/RTL/Providers/ScalarValueSelect.h"
#include "Hardware/RTL/Providers/TokenConstantSync.h"
#include "Hardware/RTL/Providers/TokenMuxDemux.h"

#include "ConfigurationABITestSupport.h"

#include "ADG/Builder.h"
#include "ADG/FuLibrary.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Hardware/Configuration/ConfigurationABI.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
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

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireSuccess(llvm::Error error) {
  if (error)
    fail(llvm::toString(std::move(error)));
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    llvm::SmallString<128> path;
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-portable-provider-closure", path))
      fail(error.message());
    path_ = path.str().str();
  }

  ~TemporaryDirectory() {
    if (std::error_code error = llvm::sys::fs::remove_directories(path_))
      llvm::errs() << "portable provider coverage test: unable to remove "
                   << path_ << ": " << error.message() << '\n';
  }

  llvm::StringRef path() const { return path_; }

private:
  std::string path_;
};

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
      loom::hardware::rtl::registerPortableScalarMathTrigonometricProviders(
          registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableScalarMathHyperbolicProviders(
          registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableMathExponentialProviders(registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableMathLogarithmProviders(registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableMathRoundingProviders(registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableMathRootProviders(registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableMathErfProvider(registry));
  requireRegistration(
      loom::hardware::rtl::registerPortableMathPowerProvider(registry));
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
      loom::hardware::rtl::
          registerPortableFixedVectorParallelizeSerializeProviders(registry));
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
      ::fabric::ImplementationFamilyId::FixedVectorParallelize,
      ::fabric::ImplementationFamilyId::FixedVectorSerialize,
      ::fabric::ImplementationFamilyId::TokenConstant,
      ::fabric::ImplementationFamilyId::TokenSync,
      ::fabric::ImplementationFamilyId::TokenMux,
      ::fabric::ImplementationFamilyId::TokenDemux,
      ::fabric::ImplementationFamilyId::ScalarSignedIntegerDivRem,
      ::fabric::ImplementationFamilyId::ScalarUnsignedIntegerDivRem,
      ::fabric::ImplementationFamilyId::ScalarFloatDivide,
      ::fabric::ImplementationFamilyId::ScalarFloatRemainder,
      ::fabric::ImplementationFamilyId::ScalarMathSin,
      ::fabric::ImplementationFamilyId::ScalarMathCos,
      ::fabric::ImplementationFamilyId::ScalarMathTan,
      ::fabric::ImplementationFamilyId::ScalarMathSinh,
      ::fabric::ImplementationFamilyId::ScalarMathCosh,
      ::fabric::ImplementationFamilyId::ScalarMathTanh,
      ::fabric::ImplementationFamilyId::ScalarMathExp,
      ::fabric::ImplementationFamilyId::ScalarMathExp2,
      ::fabric::ImplementationFamilyId::ScalarMathExpM1,
      ::fabric::ImplementationFamilyId::ScalarMathLog,
      ::fabric::ImplementationFamilyId::ScalarMathLog2,
      ::fabric::ImplementationFamilyId::ScalarMathLog10,
      ::fabric::ImplementationFamilyId::ScalarMathLog1p,
      ::fabric::ImplementationFamilyId::ScalarMathFloor,
      ::fabric::ImplementationFamilyId::ScalarMathCeil,
      ::fabric::ImplementationFamilyId::ScalarMathRound,
      ::fabric::ImplementationFamilyId::ScalarMathTrunc,
      ::fabric::ImplementationFamilyId::ScalarMathRoundEven,
      ::fabric::ImplementationFamilyId::ScalarMathSqrt,
      ::fabric::ImplementationFamilyId::ScalarMathRsqrt,
      ::fabric::ImplementationFamilyId::ScalarMathErf,
      ::fabric::ImplementationFamilyId::ScalarIntegerSaturatingAddSub,
      ::fabric::ImplementationFamilyId::FixedVectorIntegerSaturatingAddSub,
      ::fabric::ImplementationFamilyId::ScalarIntegerCountZeros,
      ::fabric::ImplementationFamilyId::FixedVectorIntegerCountZeros,
      ::fabric::ImplementationFamilyId::ScalarMathPow,
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

loom::fabric::FinalizedFabricRoot
buildPortableSpecialMathModule(const loom::ArtifactStore &artifacts) {
  const loom::adg::PortType bits128 =
      take(loom::adg::PortType::bits(128));
  const std::vector<loom::adg::PortType> inputTypes(2, bits128);
  const std::vector<loom::adg::PortType> outputTypes(1, bits128);
  loom::adg::DesignBuilder design(artifacts);
  auto spatial = take(design.createSpatialCore(
      "portable-special-math-provider-closure", inputTypes, outputTypes));
  std::vector<loom::adg::SpatialValue> spatialInputs;
  for (std::size_t ordinal = 0; ordinal != inputTypes.size(); ++ordinal)
    spatialInputs.push_back(take(spatial.input(ordinal)));
  auto pe = take(spatial.addPe(
      spatialInputs, loom::adg::PeSpec::spatial(inputTypes, outputTypes)));
  std::vector<loom::adg::PeValue> peInputs;
  for (std::size_t ordinal = 0; ordinal != inputTypes.size(); ++ordinal)
    peInputs.push_back(take(pe.input(ordinal)));
  requireSuccess(loom::adg::addSpecialMathFu(
      pe, peInputs,
      loom::adg::BuiltinSpecialMathCapabilityProfile::PortableProviderClosed));
  requireSuccess(pe.close());
  requireSuccess(spatial.close({take(pe.output(0))}));
  auto finalized = take(std::move(design).finalize());
  require(finalized.roots().size() == 1,
          "special-math fixture did not finalize one Module");
  return std::move(finalized.roots().front());
}

struct PortableClosureResult final {
  loom::ArtifactRootReference module;
  loom::ArtifactRootReference system;
  loom::ArtifactRootReference configurationAbi;
  loom::ArtifactRootReference implementation;
  std::vector<std::uint8_t> implementationBytes;
};

PortableClosureResult materializePortableSpecialMathClosure(
    const std::filesystem::path &root) {
  const std::filesystem::path objectRoot = root / "objects";
  const std::filesystem::path blobRoot = root / "blobs";
  std::error_code error;
  std::filesystem::create_directories(objectRoot, error);
  if (error)
    fail("could not create the provider-closure ArtifactStore");
  std::filesystem::create_directories(blobRoot, error);
  if (error)
    fail("could not create the provider-closure BlobStore");
  loom::ArtifactStore artifacts(objectRoot.string());
  loom::BlobStore blobs(blobRoot.string());
  auto module = buildPortableSpecialMathModule(artifacts);
  auto system = take(loom::hardware::test::makeSingleSpatialCoreSystem(
      module, artifacts));
  auto abiDraft = take(
      loom::hardware::test::makeCompleteConfigurationABIDraft(system));
  auto abi = take(loom::hardware::finalizeConfigurationABI(
      std::move(abiDraft), artifacts));
  const loom::fabric::SpatialCoreOccurrenceRef subject =
      take(loom::hardware::test::requireSingleSpatialCoreOccurrence(system));
  auto implementation = take(
      loom::hardware::rtl::finalizePortableSpatialCoreHardwareImplementation(
          abi, subject, std::nullopt, artifacts, blobs));
  requireSuccess(
      loom::hardware::rtl::verifyPortableSpatialCoreHardwareImplementation(
          abi, implementation));
  return {module.reference(),
          system.reference(),
          abi.reference(),
          implementation.reference(),
          std::vector<std::uint8_t>(
              implementation.canonicalBytes().bytes().begin(),
              implementation.canonicalBytes().bytes().end())};
}

void portableSpecialMathProfileHasExecutableProviderClosure() {
  TemporaryDirectory directory;
  const std::filesystem::path root(directory.path().str());
  const PortableClosureResult first =
      materializePortableSpecialMathClosure(root / "first");
  const PortableClosureResult repeated =
      materializePortableSpecialMathClosure(root / "repeated");
  require(first.module == repeated.module && first.system == repeated.system &&
              first.configurationAbi == repeated.configurationAbi &&
              first.implementation == repeated.implementation &&
              first.implementationBytes == repeated.implementationBytes,
          "cold portable provider closure changed an artifact identity");
}

} // namespace

int main() {
  aggregateRegistrationIsTheCoverageAuthority();
  aggregateRegistrationIsTransactional();
  portableSpecialMathProfileHasExecutableProviderClosure();
  return EXIT_SUCCESS;
}
