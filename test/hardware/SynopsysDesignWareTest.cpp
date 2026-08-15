#include "ConfigurationABITestSupport.h"
#include "Hardware/RTL/CirctConformance.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/FloatDivideRemainder.h"
#include "Hardware/RTL/Providers/Native/DesignWare.h"
#include "Hardware/RTL/Providers/ScalarFloatFma.h"
#include "PortableProviderTestSupport.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

using loom::ArtifactStore;
using loom::BlobDigest;
using loom::BlobStore;
using loom::fabric::FabricFuOccurrenceNodeRef;
using loom::fabric::FinalizedFabricRoot;
using namespace loom::hardware;
using namespace loom::hardware::rtl;

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, llvm::StringRef message) {
  if (!condition)
    fail(test, message.str());
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

mlir::MLIRContext &fabricContext() {
  static mlir::MLIRContext *context = [] {
    mlir::DialectRegistry registry;
    registry.insert<::dataflow::DataflowDialect, ::fabric::FabricDialect,
                    mlir::arith::ArithDialect, mlir::func::FuncDialect>();
    auto *result =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    result->loadAllAvailableDialects();
    return result;
  }();
  return *context;
}

enum class Family { Fma, Divide };

struct CapabilitySpec final {
  Family family = Family::Fma;
  llvm::StringRef formats = R"("f32")";
  llvm::StringRef roundings = R"("to_nearest_even")";
  llvm::StringRef nanBehaviors = R"("ieee")";
  llvm::StringRef signedZeroBehaviors = R"("preserve")";
  llvm::StringRef fastmath = "none";
  unsigned portWidth = 64;
  bool wrongContract = false;
};

::fabric::ImplementationFamilyId familyId(Family family) {
  return family == Family::Fma
             ? ::fabric::ImplementationFamilyId::ScalarFloatFma
             : ::fabric::ImplementationFamilyId::ScalarFloatDivide;
}

std::string fabricSource(const CapabilitySpec &spec) {
  const bool fma = spec.family == Family::Fma;
  std::string text;
  llvm::raw_string_ostream source(text);
  source << "module { fabric.module @designware_fixture("
         << "%a: !fabric.bits<" << spec.portWidth << ">, "
         << "%b: !fabric.bits<" << spec.portWidth << ">";
  if (fma)
    source << ", %c: !fabric.bits<" << spec.portWidth << ">";
  source << ") -> !fabric.bits<" << spec.portWidth << "> { "
         << "%pe = fabric.pe [spatial]("
         << "%pa = %a : !fabric.bits<" << spec.portWidth << ">, "
         << "%pb = %b : !fabric.bits<" << spec.portWidth << ">";
  if (fma)
    source << ", %pc = %c : !fabric.bits<" << spec.portWidth << ">";
  source << ") -> !fabric.bits<" << spec.portWidth << "> { "
         << "%fu = fabric.fu("
         << "%fa = %pa : !fabric.bits<" << spec.portWidth << ">, "
         << "%fb = %pb : !fabric.bits<" << spec.portWidth << ">";
  if (fma)
    source << ", %fc = %pc : !fabric.bits<" << spec.portWidth << ">";
  source << ") -> !fabric.bits<" << spec.portWidth << "> { "
         << "%value = fabric.op [@" << (fma ? "math.fma" : "arith.divf")
         << "] (%fa, %fb";
  if (fma)
    source << ", %fc";
  source << ") {implementation_family = #fabric.implementation_family<"
         << (fma ? "ScalarFloatFma" : "ScalarFloatDivide")
         << ">, hw_params = {float_formats = [" << spec.formats
         << "], behavior = {rounding_modes = [" << spec.roundings
         << "], nan_behaviors = [" << spec.nanBehaviors
         << "], subnormal_behaviors = [\"preserve\"], "
            "signed_zero_behaviors = ["
         << spec.signedZeroBehaviors << "], fastmath = \"" << spec.fastmath
         << "\"}}} : (!fabric.bits<" << spec.portWidth << ">, !fabric.bits<"
         << spec.portWidth << ">";
  if (fma)
    source << ", !fabric.bits<" << spec.portWidth << ">";
  source << ") -> !fabric.bits<" << spec.portWidth
         << "> fabric.yield %value : !fabric.bits<" << spec.portWidth
         << "> } } fabric.yield %pe : !fabric.bits<" << spec.portWidth
         << "> } }";
  return source.str();
}

struct FabricFixture final {
  FinalizedFabricRoot fabric;
  FabricFuOccurrenceNodeRef occurrence;
  FinalizedFabricRoot system;
  loom::fabric::FabricPhysicalOccurrenceOwnerRef physicalOccurrence;
};

FabricFixture makeFabric(llvm::StringRef test, const ArtifactStore &store,
                         const CapabilitySpec &spec = {}) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(fabricSource(spec),
                                                        &fabricContext());
  require(test, static_cast<bool>(source), "could not parse Fabric fixture");
  const ::fabric::ResourceContract contract =
      spec.wrongContract ? ::fabric::loopCarryOperationResourceContract()
                         : ::fabric::oneCycleElasticOperationResourceContract();
  const std::vector<std::uint8_t> encoded =
      take(test, ::fabric::encodeResourceContractRecord(contract));
  const std::vector<std::int8_t> signedBytes(encoded.begin(), encoded.end());
  source->walk([&](::fabric::OpOp operation) {
    operation->setAttr(
        ::fabric::kResourceContractRecordAttrName,
        mlir::DenseI8ArrayAttr::get(&fabricContext(), signedBytes));
  });

  ::fabric::ModuleOp root;
  source->walk([&](::fabric::ModuleOp candidate) { root = candidate; });
  require(test, static_cast<bool>(root), "Fabric fixture has no root");
  FinalizedFabricRoot fabric =
      take(test, loom::fabric::finalizeFabricRoot(root, store));
  for (const auto fuOccurrence : fabric.view().fuOccurrences()) {
    const auto definition = fabric.view().fuTemplateOf(fuOccurrence);
    if (!definition)
      continue;
    for (const auto &capability :
         fabric.view().resolvedFabricOpCapabilities(*definition)) {
      if (capability.implementationFamily != familyId(spec.family))
        continue;
      FabricFuOccurrenceNodeRef occurrence =
          take(test, loom::fabric::deriveFabricFuOccurrenceNode(
                         fabric.view(), capability.occurrence, fuOccurrence));
      FinalizedFabricRoot system =
          take(test, loom::hardware::test::makeSingleSpatialCoreSystem(fabric,
                                                                       store));
      auto systemView =
          take(test, loom::fabric::requireSystemRoot(system.view()));
      auto operations =
          take(test, enumerateFabricPhysicalOperations(systemView));
      const auto physical = llvm::find_if(operations, [&](const auto &entry) {
        return entry.localOccurrence == occurrence;
      });
      require(test, physical != operations.end(),
              "System has no physical operation occurrence");
      return FabricFixture{std::move(fabric), occurrence, std::move(system),
                           physical->physicalOccurrence};
    }
  }
  fail(test, "Fabric fixture has no requested operation occurrence");
}

FinalizedConfigurationABI makeConfigurationAbi(llvm::StringRef test,
                                               const ArtifactStore &store,
                                               const FabricFixture &fixture) {
  return take(
      test,
      finalizeConfigurationABI(
          take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                         fixture.system)),
          store));
}

std::unique_ptr<mlir::MLIRContext> makeCirctContext() {
  mlir::DialectRegistry registry;
  registry.insert<circt::comb::CombDialect, circt::hw::HWDialect,
                  circt::seq::SeqDialect, circt::sv::SVDialect>();
  auto context = std::make_unique<mlir::MLIRContext>(
      registry, mlir::MLIRContext::Threading::DISABLED);
  context->loadAllAvailableDialects();
  return context;
}

struct SkeletonFixture final {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  circt::hw::HWModuleGeneratedOp leaf;
};

SkeletonFixture makeSkeleton(llvm::StringRef test, mlir::MLIRContext &context,
                             const FabricFixture &fabric,
                             const ConfigurationABI &abi,
                             llvm::StringRef moduleName) {
  const auto *capability =
      fabric.fabric.view().resolvedFabricOpCapability(fabric.occurrence);
  require(test, capability != nullptr, "Fabric capability did not resolve");
  mlir::OpBuilder builder(&context);
  const mlir::Location location = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(module->getBody());
  circt::hw::HWGeneratorSchemaOp::create(
      builder, location, fabricOperationGeneratorSchemaSymbol,
      fabricOperationGeneratorDescriptor, builder.getArrayAttr({}));
  std::vector<circt::hw::PortInfo> ports =
      take(test, deriveFabricOperationLeafPorts(
                     builder, fabric.physicalOccurrence, *capability, abi));
  circt::hw::HWModuleGeneratedOp leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr(moduleName), ports);
  return SkeletonFixture{std::move(module), leaf};
}

std::string moduleText(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  return text;
}

ExternalInputBinding exactDesignWareInput() {
  return {synopsysDesignWareComponentInputSlot.str(),
          ToolBundledResourceDependency{
              synopsysDesignWareBuildIdentity.str(),
              synopsysDesignWareDwFpMacResourceKey.str()}};
}

FabricOperationProviderRegistry makeRegistry(llvm::StringRef test) {
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableScalarFloatFmaProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error =
          registerPortableFloatDivideRemainderProviders(registry))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error =
          registerSynopsysDesignWareScalarFloatFmaProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  return registry;
}

ExternalImplementationContractCatalog
makeContractCatalog(llvm::StringRef test) {
  ExternalImplementationContractCatalog catalog;
  if (llvm::Error error = registerSynopsysDesignWareExternalContract(catalog))
    fail(test, llvm::toString(std::move(error)));
  return catalog;
}

struct SpecializationResult final {
  FabricOperationProviderOutput output;
  std::string rtl;
};

llvm::Expected<SpecializationResult>
specialize(SkeletonFixture &skeleton, const FabricFixture &fabric,
           const FinalizedConfigurationABI &abi,
           const FabricOperationProviderRegistry &registry,
           const ExternalImplementationContractCatalog &catalog,
           BackendRecipeKey recipe, std::vector<ExternalInputBinding> inputs) {
  const std::string before = moduleText(*skeleton.module);
  const std::vector<FabricOperationLeafAssociation> leaves{
      {skeleton.leaf, fabric.physicalOccurrence}};
  const std::vector<FabricOperationRecipeBinding> recipes{
      {fabric.physicalOccurrence, recipe, std::move(inputs)}};
  auto output = specializeFabricOperationLeaves(*skeleton.module, abi, leaves,
                                                recipes, registry, catalog);
  if (!output) {
    llvm::Error error = output.takeError();
    if (moduleText(*skeleton.module) != before) {
      llvm::consumeError(std::move(error));
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "failed specialization did not roll back");
    }
    return std::move(error);
  }
  auto rtl = lowerAndExportSpecializedSystemVerilog(*skeleton.module);
  if (!rtl)
    return rtl.takeError();
  return SpecializationResult{std::move(*output), std::move(*rtl)};
}

void expectUnsupported(llvm::StringRef test, const std::filesystem::path &root,
                       const CapabilitySpec &spec,
                       const FabricOperationProviderRegistry &registry,
                       const ExternalImplementationContractCatalog &catalog,
                       std::vector<ExternalInputBinding> inputs = {
                           exactDesignWareInput()}) {
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric = makeFabric(test, store, spec);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton =
      makeSkeleton(test, *context, fabric, abi.abi(), "unsupported_native");
  auto result =
      specialize(skeleton, fabric, abi, registry, catalog,
                 BackendRecipeKey::SynopsysDesignWare, std::move(inputs));
  require(test, !result, "accepted an unverified DesignWare capability");
  bool classified = false;
  llvm::handleAllErrors(
      result.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        classified = error.implementationFamily() == familyId(spec.family) &&
                     error.recipe() == BackendRecipeKey::SynopsysDesignWare;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test, "unverified capability returned the wrong error class: " +
                       error.message());
      });
  require(test, classified,
          "unverified capability lost typed Unsupported classification");
}

void exactOccurrenceBindingAndDeterminism(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  require(test, synopsysDesignWareDwFpMacResourceKey == "dwbb/DW_fp_mac",
          "DesignWare resource key does not name the selected component");
  std::filesystem::create_directories(root / "artifacts");
  std::filesystem::create_directories(root / "blobs");
  ArtifactStore store((root / "artifacts").string());
  BlobStore blobs((root / "blobs").string());
  FabricFixture fabric = makeFabric(test, store);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  FabricOperationProviderRegistry registry = makeRegistry(test);
  ExternalImplementationContractCatalog catalog = makeContractCatalog(test);

  const auto coverage = registry.coverage();
  const auto &fmaCoverage = coverage[static_cast<std::uint32_t>(
      ::fabric::ImplementationFamilyId::ScalarFloatFma)];
  require(
      test,
      fmaCoverage.recipes ==
          std::vector<BackendRecipeKey>{BackendRecipeKey::PortableSystemVerilog,
                                        BackendRecipeKey::SynopsysDesignWare},
      "portable and native FMA recipes are not independently registered");

  std::unique_ptr<mlir::MLIRContext> firstContext = makeCirctContext();
  SkeletonFixture first =
      makeSkeleton(test, *firstContext, fabric, abi.abi(), "designware_fma");
  SpecializationResult firstResult =
      take(test, specialize(first, fabric, abi, registry, catalog,
                            BackendRecipeKey::SynopsysDesignWare,
                            {exactDesignWareInput()}));
  std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
  SkeletonFixture second =
      makeSkeleton(test, *secondContext, fabric, abi.abi(), "designware_fma");
  SpecializationResult secondResult =
      take(test, specialize(second, fabric, abi, registry, catalog,
                            BackendRecipeKey::SynopsysDesignWare,
                            {exactDesignWareInput()}));
  require(test, firstResult.rtl == secondResult.rtl,
          "native wrapper generation is not deterministic");
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "generated", {{"designware_fma.sv", firstResult.rtl}}))
    fail(test, llvm::toString(std::move(error)));

  const llvm::StringRef rtl(firstResult.rtl);
  require(
      test,
      rtl.contains("module designware_fma") && rtl.contains("DW_fp_mac #(") &&
          rtl.contains(".sig_width(23)") && rtl.contains(".exp_width(8)") &&
          rtl.contains(".ieee_compliance(1)") && rtl.contains(".rnd(3'b000)") &&
          rtl.contains(".status(") && !rtl.contains("function automatic"),
      "native wrapper does not contain the exact DesignWare instance");

  require(test,
          firstResult.output.payloads.size() == 1 &&
              firstResult.output.activityPoints.empty() &&
              firstResult.output.externalImplementationBindings.size() == 1,
          "native provider output has the wrong ownership surface");
  const FabricOperationProviderPayload &payload =
      firstResult.output.payloads.front();
  require(test,
          payload.role == PayloadRole::BlackBoxContract &&
              payload.canonicalLogicalName ==
                  "black-box/synopsys-designware-dw-fp-mac-f32-rne-ieee-v1" &&
              std::string(payload.bytes.begin(), payload.bytes.end()) ==
                  "synopsys.designware.DW_fp_mac.f32.rne.ieee.v1\n",
          "native provider emitted the wrong opaque BlackBoxContract");
  const ExternalImplementationBindingDraft &binding =
      firstResult.output.externalImplementationBindings.front();
  require(test,
          binding.providerContractRef == synopsysDesignWareContractRef &&
              binding.externalInputs ==
                  std::vector<ExternalInputBinding>{exactDesignWareInput()} &&
              binding.fabricResourceRefs ==
                  std::vector<loom::fabric::FabricPhysicalOccurrenceOwnerRef>{
                      fabric.physicalOccurrence} &&
              binding.representationLocators ==
                  std::vector<RepresentationLocator>{
                      {RepresentationObjectKind::Module, "DW_fp_mac"}} &&
              binding.blackBoxContractPayload ==
                  ImplementationPayloadKey{
                      PayloadRole::BlackBoxContract,
                      "black-box/synopsys-designware-dw-fp-mac-f32-rne-ieee-"
                      "v1"},
          "native binding is not closed over the exact physical occurrence");

  const std::vector<std::uint8_t> rtlBytes(firstResult.rtl.begin(),
                                           firstResult.rtl.end());
  const BlobDigest rtlDigest = take(test, blobs.put(rtlBytes));
  std::vector<ImplementationPayload> payloads{
      {PayloadRole::RtlSource, "rtl/designware_fma.sv", rtlDigest}};
  for (const FabricOperationProviderPayload &providerPayload :
       firstResult.output.payloads) {
    const ImplementationPayload descriptor = providerPayload.descriptor();
    require(test,
            take(test, blobs.put(providerPayload.bytes)) ==
                descriptor.blobDigest,
            "provider payload publication changed its content identity");
    payloads.push_back(descriptor);
  }
  const RepresentationFormatDescriptorRef format =
      take(test, RepresentationFormatDescriptorRef::get(
                     RepresentationFormatKind::SystemVerilogRtl));
  ImplementationRepresentationRoot representation =
      take(test, createImplementationRepresentationRoot(
                     RepresentationRootVariant::Rtl, std::nullopt, format,
                     {RepresentationObjectKind::Module, "designware_fma"},
                     std::move(payloads)));
  HardwareImplementationDraft draft{
      fabric.system.reference(),
      take(test, loom::hardware::test::requireSingleSpatialCoreOccurrence(
                     fabric.system)),
      abi.reference(),
      std::move(representation),
      std::nullopt,
      {},
      firstResult.output.activityPoints,
      {},
      firstResult.output.externalImplementationBindings,
  };
  FinalizedHardwareImplementation implementation =
      take(test, finalizeHardwareImplementation(std::move(draft), catalog,
                                                store, blobs));
  require(
      test,
      implementation.implementation().externalImplementationBindings().size() ==
          1,
      "native binding did not finalize into HardwareImplementation");

  std::unique_ptr<mlir::MLIRContext> portableContext = makeCirctContext();
  SkeletonFixture portable =
      makeSkeleton(test, *portableContext, fabric, abi.abi(), "portable_fma");
  SpecializationResult portableResult =
      take(test, specialize(portable, fabric, abi, registry, catalog,
                            BackendRecipeKey::PortableSystemVerilog, {}));
  require(
      test,
      portableResult.output.payloads.empty() &&
          portableResult.output.externalImplementationBindings.empty() &&
          !llvm::StringRef(portableResult.rtl).contains("DW_fp_mac") &&
          llvm::StringRef(portableResult.rtl).contains("function automatic"),
      "portable recipe did not remain independently selectable");
}

void unverifiedInputsRollbackWithoutFallback(
    const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  FabricOperationProviderRegistry registry = makeRegistry(test);
  ExternalImplementationContractCatalog catalog = makeContractCatalog(test);

  unsigned ordinal = 0;
  for (llvm::StringRef format : {R"("f16")", R"("bf16")", R"("f64")"}) {
    CapabilitySpec spec;
    spec.formats = format;
    expectUnsupported(test, root / ("format-" + std::to_string(ordinal++)),
                      spec, registry, catalog);
  }
  for (llvm::StringRef rounding :
       {R"("downward")", R"("upward")", R"("toward_zero")",
        R"("to_nearest_away")"}) {
    CapabilitySpec spec;
    spec.roundings = rounding;
    expectUnsupported(test, root / ("rounding-" + std::to_string(ordinal++)),
                      spec, registry, catalog);
  }

  CapabilitySpec nan;
  nan.nanBehaviors = R"("number_preferred")";
  expectUnsupported(test, root / "nan", nan, registry, catalog);
  CapabilitySpec multiFormat;
  multiFormat.formats = R"("f32", "f64")";
  expectUnsupported(test, root / "multi-format", multiFormat, registry,
                    catalog);
  CapabilitySpec multiRounding;
  multiRounding.roundings = R"("to_nearest_even", "downward")";
  expectUnsupported(test, root / "multi-rounding", multiRounding, registry,
                    catalog);
  CapabilitySpec signedZero;
  signedZero.signedZeroBehaviors = R"("ignore_sign")";
  expectUnsupported(test, root / "signed-zero", signedZero, registry, catalog);
  CapabilitySpec fastmath;
  fastmath.fastmath = "nnan";
  expectUnsupported(test, root / "fastmath", fastmath, registry, catalog);
  CapabilitySpec width;
  width.portWidth = 32;
  expectUnsupported(test, root / "width", width, registry, catalog);
  CapabilitySpec wideWidth;
  wideWidth.portWidth = 128;
  expectUnsupported(test, root / "wide-width", wideWidth, registry, catalog);
  CapabilitySpec contract;
  contract.wrongContract = true;
  expectUnsupported(test, root / "contract", contract, registry, catalog);

  ExternalInputBinding wrongBuild = exactDesignWareInput();
  std::get<ToolBundledResourceDependency>(wrongBuild.dependencyIdentity)
      .stableProviderBuildIdentity = "synopsys.designware:unverified";
  expectUnsupported(test, root / "build", {}, registry, catalog,
                    {std::move(wrongBuild)});
  ExternalInputBinding wrongResource = exactDesignWareInput();
  std::get<ToolBundledResourceDependency>(wrongResource.dependencyIdentity)
      .resourceKey = "foundation.unverified";
  expectUnsupported(test, root / "component", {}, registry, catalog,
                    {std::move(wrongResource)});

  CapabilitySpec divide;
  divide.family = Family::Divide;
  expectUnsupported(test, root / "family", divide, registry, catalog);
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one temporary directory argument");
  const std::filesystem::path root(argv[1]);
  exactOccurrenceBindingAndDeterminism(root / "exact");
  unverifiedInputsRollbackWithoutFallback(root / "unsupported");
  return 0;
}
