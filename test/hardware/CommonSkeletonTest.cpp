#include "Hardware/RTL/CommonSkeleton.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/ScalarIntegerAddSub.h"
#include "Hardware/RTL/Specialization.h"

#include "ConfigurationABI3TestSupport.h"
#include "ConfigurationTransportTestSupport.h"
#include "PortableProviderTestSupport.h"

#include "ADG/Builder.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "Fabric/Identity/FabricPeConfiguration.h"
#include "Fabric/Identity/FabricTemporalPeConfiguration.h"
#include "Hardware/Implementation/HardwareImplementation.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using loom::ArtifactStore;
using loom::fabric::FinalizedFabricRoot;
using loom::hardware::ExternalImplementationContractCatalog;
using loom::hardware::FinalizedConfigurationABI;
using loom::hardware::rtl::FabricOperationLeafAssociation;
using loom::hardware::rtl::FabricOperationProviderRegistry;
using loom::hardware::rtl::ResolvedFabricPhysicalOperation;

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

void expectError(llvm::StringRef test, llvm::Error error,
                 llvm::StringRef expected) {
  if (!error)
    fail(test, "accepted invalid common CIRCT skeleton");
  const std::string message = llvm::toString(std::move(error));
  require(
      test, llvm::StringRef(message).contains(expected),
      (llvm::Twine("expected '") + expected + "', received '" + message + "'")
          .str());
}

template <typename T>
void expectError(llvm::StringRef test, llvm::Expected<T> value,
                 llvm::StringRef expected) {
  if (value)
    fail(test, "accepted invalid common CIRCT skeleton");
  expectError(test, value.takeError(), expected);
}

template <typename T>
void expectStructuralUnsupported(llvm::StringRef test,
                                 llvm::Expected<T> value) {
  if (value)
    fail(test, "accepted unsupported Fabric structural topology");
  std::string reason;
  std::string unexpected;
  llvm::handleAllErrors(
      value.takeError(),
      [&](const loom::hardware::rtl::FabricStructuralLoweringUnsupportedError
              &error) { reason = error.reason().str(); },
      [&](const llvm::ErrorInfoBase &error) {
        llvm::raw_string_ostream stream(unexpected);
        error.log(stream);
      });
  require(test, unexpected.empty(),
          "unsupported topology returned the wrong typed error: " + unexpected);
  require(test, !reason.empty(), "unsupported topology has no diagnostic");
}

class TemporaryDirectory final {
public:
  explicit TemporaryDirectory(llvm::StringRef test) : test_(test.str()) {
    llvm::SmallString<128> path;
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-common-skeleton-test", path))
      fail(test, error.message());
    path_ = path.str().str();
  }

  ~TemporaryDirectory() {
    if (std::error_code error = llvm::sys::fs::remove_directories(path_))
      llvm::errs() << test_ << ": unable to remove temporary directory: "
                   << error.message() << '\n';
  }

  llvm::StringRef path() const { return path_; }

private:
  std::string test_;
  std::string path_;
};

FinalizedFabricRoot makeOperationFabric(llvm::StringRef test,
                                        const ArtifactStore &store,
                                        bool twoOccurrences = false) {
  mlir::DialectRegistry registry;
  registry.insert<::dataflow::DataflowDialect, ::fabric::FabricDialect,
                  mlir::arith::ArithDialect, mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  const llvm::StringRef sourceText = twoOccurrences ? R"mlir(
    module {
      fabric.module @two_integer_adds(
          %a0: !fabric.bits<8>, %b0: !fabric.bits<8>,
          %a1: !fabric.bits<8>, %b1: !fabric.bits<8>)
          -> (!fabric.bits<8>, !fabric.bits<8>) {
        %pe0 = fabric.pe [spatial]
            (%pa0 = %a0 : !fabric.bits<8>, %pb0 = %b0 : !fabric.bits<8>)
            -> !fabric.bits<8> {
          %fu0 = fabric.fu
              (%fa0 = %pa0 : !fabric.bits<8>, %fb0 = %pb0 : !fabric.bits<8>)
              -> !fabric.bits<8> {
            %value0 = fabric.op [@arith.addi] (%fa0, %fb0)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerAddSub>,
               hw_params = {integer_widths = [8 : i32]}}
              : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
            fabric.yield %value0 : !fabric.bits<8>
          }
        }
        %pe1 = fabric.pe [spatial]
            (%pa1 = %a1 : !fabric.bits<8>, %pb1 = %b1 : !fabric.bits<8>)
            -> !fabric.bits<8> {
          %fu1 = fabric.fu
              (%fa1 = %pa1 : !fabric.bits<8>, %fb1 = %pb1 : !fabric.bits<8>)
              -> !fabric.bits<8> {
            %value1 = fabric.op [@arith.addi] (%fa1, %fb1)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerAddSub>,
               hw_params = {integer_widths = [8 : i32]}}
              : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
            fabric.yield %value1 : !fabric.bits<8>
          }
        }
        fabric.yield %pe0, %pe1 : !fabric.bits<8>, !fabric.bits<8>
      }
    }
  )mlir"
                                                    : R"mlir(
    module {
      fabric.module @integer_add(%a: !fabric.bits<8>, %b: !fabric.bits<8>)
          -> !fabric.bits<8> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<8>, %pb = %b : !fabric.bits<8>)
            -> !fabric.bits<8> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<8>, %fb = %pb : !fabric.bits<8>)
              -> !fabric.bits<8> {
            %value = fabric.op [@arith.addi] (%fa, %fb)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerAddSub>,
               hw_params = {integer_widths = [8 : i32]}}
              : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
            fabric.yield %value : !fabric.bits<8>
          }
        }
        fabric.yield %pe : !fabric.bits<8>
      }
    }
  )mlir";
  mlir::OwningOpRef<mlir::ModuleOp> source =
      mlir::parseSourceString<mlir::ModuleOp>(sourceText, &context);
  require(test, static_cast<bool>(source),
          "unable to parse operation Fabric fixture");
  const std::vector<std::uint8_t> contract =
      take(test, ::fabric::encodeResourceContractRecord(
                     ::fabric::oneCycleElasticOperationResourceContract()));
  const std::vector<std::int8_t> signedContract(contract.begin(),
                                                contract.end());
  source->walk([&](::fabric::OpOp operation) {
    operation->setAttr(::fabric::kResourceContractRecordAttrName,
                       mlir::DenseI8ArrayAttr::get(&context, signedContract));
  });
  ::fabric::ModuleOp root;
  source->walk([&](::fabric::ModuleOp candidate) { root = candidate; });
  require(test, static_cast<bool>(root),
          "operation Fabric fixture has no Module root");
  return take(test, loom::fabric::finalizeFabricRoot(root, store));
}

FinalizedFabricRoot makeBoundaryOnlyFabric(llvm::StringRef test,
                                           const ArtifactStore &store) {
  mlir::DialectRegistry registry;
  registry.insert<::fabric::FabricDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  mlir::OwningOpRef<mlir::ModuleOp> source =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
        module {
          fabric.module @passthrough(
              %data: !fabric.bits<32>,
              %tagged: !fabric.bits_tag<4, 5>)
              -> (!fabric.bits<16>, !fabric.bits_tag<0, 3>) {
            fabric.yield %data : !fabric.bits<32> to !fabric.bits<16>,
                         %tagged : !fabric.bits_tag<4, 5>
                             to !fabric.bits_tag<0, 3>
          }
        }
      )mlir",
                                              &context);
  require(test, static_cast<bool>(source),
          "unable to parse boundary-only Fabric fixture");
  ::fabric::ModuleOp root;
  for (::fabric::ModuleOp candidate : source->getOps<::fabric::ModuleOp>()) {
    require(test, !root, "boundary fixture has multiple Module roots");
    root = candidate;
  }
  require(test, static_cast<bool>(root), "boundary fixture has no Module root");
  return take(test, loom::fabric::finalizeFabricRoot(root, store));
}

FinalizedFabricRoot makeSpatialHierarchyFabric(llvm::StringRef test,
                                               const ArtifactStore &store) {
  using namespace loom::adg;
  DesignBuilder design(store);
  const PortType bits2 = take(test, PortType::bits(2));
  const PortType bits8 = take(test, PortType::bits(8));
  const PortType tagged8x2 = take(test, PortType::taggedBits(8, 2));
  auto spatial =
      take(test, design.createSpatialCore("spatial-hierarchy",
                                          {bits8, bits8, bits2}, {tagged8x2}));
  auto routed = take(
      test, spatial.addSwitch(
                {take(test, spatial.input(0)), take(test, spatial.input(1))},
                SwitchSpec::spatial({bits8, bits8}, {bits8, bits8},
                                    {{0, 1}, {0, 1}})));
  auto pe = take(test, spatial.addPe(routed.values(),
                                     PeSpec::spatial({bits8, bits8}, {bits8})));
  auto fu =
      take(test, pe.addFu({take(test, pe.input(0)), take(test, pe.input(1))},
                          FuSpec{{bits8, bits8}, {bits8}}));
  const auto width =
      llvm::find_if(::fabric::integerWidthDomain, [](auto candidate) {
        return ::fabric::getBitWidth(candidate) == 8;
      });
  require(test, width != ::fabric::integerWidthDomain.end(),
          "integer width catalog omitted i8");
  auto operation = take(
      test, fu.addOperation(
                {take(test, fu.input(0)), take(test, fu.input(1))},
                OperationCapabilitySpec{
                    ::fabric::ImplementationFamilyId::ScalarIntegerAddSub,
                    ::fabric::ScalarIntegerParams{
                        ::fabric::IntegerWidthSet::get({*width})},
                    {::dataflow::OperationSchemaId::ArithAddI},
                    {bits8},
                    ::fabric::oneCycleElasticOperationResourceContract()}));
  if (llvm::Error error =
          fu.addCapabilityTemplate(FuCapabilityTemplateSpec{{operation}, {}}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = fu.close({take(test, operation.output(0))}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = pe.close())
    fail(test, llvm::toString(std::move(error)));
  auto fifo = take(test, spatial.addFifo(take(test, pe.output(0)),
                                         FifoSpec{bits8, 2, true}));
  auto boundary = take(
      test, spatial.addBoundary({fifo.value(), take(test, spatial.input(2))},
                                BoundarySpec::s2t(bits8, bits2, tagged8x2)));
  if (llvm::Error error = spatial.close({boundary.front()}))
    fail(test, llvm::toString(std::move(error)));
  auto finalized = take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 1,
          "hierarchy fixture did not finalize one Module");
  return std::move(finalized.roots().front());
}

FinalizedFabricRoot makeTemporalHierarchyFabric(llvm::StringRef test,
                                                const ArtifactStore &store) {
  using namespace loom::adg;
  DesignBuilder design(store);
  const PortType bits8 = take(test, PortType::bits(8));
  const PortType tagged8x2 = take(test, PortType::taggedBits(8, 2));
  auto spatial =
      take(test, design.createSpatialCore("temporal-hierarchy",
                                          {tagged8x2, tagged8x2}, {tagged8x2}));
  auto pe = take(
      test,
      spatial.addPe(
          {take(test, spatial.input(0)), take(test, spatial.input(1))},
          PeSpec::temporal(
              {bits8, bits8}, {tagged8x2},
              TemporalPeParameters{2, FuConfigurationMode::PerInstruction,
                                   ::fabric::OperandBufferMode::PerInputPort, 2,
                                   TemporalRegisterFifoParameters{1, 2, 2}})));
  auto fu =
      take(test, pe.addFu({take(test, pe.input(0)), take(test, pe.input(1))},
                          FuSpec{{bits8, bits8}, {bits8}}));
  const auto width =
      llvm::find_if(::fabric::integerWidthDomain, [](auto candidate) {
        return ::fabric::getBitWidth(candidate) == 8;
      });
  require(test, width != ::fabric::integerWidthDomain.end(),
          "integer width catalog omitted i8");
  auto operation = take(
      test, fu.addOperation(
                {take(test, fu.input(0)), take(test, fu.input(1))},
                OperationCapabilitySpec{
                    ::fabric::ImplementationFamilyId::ScalarIntegerAddSub,
                    ::fabric::ScalarIntegerParams{
                        ::fabric::IntegerWidthSet::get({*width})},
                    {::dataflow::OperationSchemaId::ArithAddI},
                    {bits8},
                    ::fabric::oneCycleElasticOperationResourceContract()}));
  if (llvm::Error error =
          fu.addCapabilityTemplate(FuCapabilityTemplateSpec{{operation}, {}}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = fu.close({take(test, operation.output(0))}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = pe.close())
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = spatial.close({take(test, pe.output(0))}))
    fail(test, llvm::toString(std::move(error)));
  auto finalized = take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 1,
          "temporal hierarchy fixture did not finalize one Module");
  return std::move(finalized.roots().front());
}

struct SystemFixture final {
  FinalizedFabricRoot module;
  FinalizedFabricRoot system;
  FinalizedConfigurationABI abi;
  loom::fabric::SpatialCoreOccurrenceRef spatialCore;
  std::vector<ResolvedFabricPhysicalOperation> operations;
};

loom::fabric::FabricPhysicalConfigurationFieldRef qualifyConfigurationField(
    llvm::StringRef test, loom::fabric::SpatialCoreOccurrenceRef spatialCore,
    const loom::fabric::FabricSemanticConfigFieldRef &field) {
  auto target =
      take(test, loom::fabric::FabricModulePhysicalTargetRef::create(field));
  return take(test, loom::fabric::FabricPhysicalConfigurationFieldRef::create(
                        loom::fabric::SpatialCoreInternalOccurrenceRef{
                            spatialCore, std::move(target)}));
}

const loom::hardware::ProgrammingUnit *findProgrammingOwner(
    llvm::StringRef test, const loom::hardware::ConfigurationABI &abi,
    const loom::fabric::FabricPhysicalConfigurationSlotRef &slot) {
  const loom::hardware::ProgrammingUnit *result = nullptr;
  for (const auto &unit : abi.programmingUnits())
    for (const auto &field : unit.fields)
      if (field.slot == slot) {
        require(test, result == nullptr,
                "configuration field has duplicate programming owners");
        result = &unit;
      }
  require(test, result != nullptr,
          "configuration field has no programming owner");
  return result;
}

const loom::fabric::FabricTransportEndpointRef &
boundaryEndpoint(llvm::StringRef test,
                 const loom::fabric::FabricArtifactView &module,
                 loom::fabric::FabricPortDirection direction,
                 loom::fabric::FabricOrdinal ordinal) {
  const loom::fabric::FabricTransportEndpointRef *result = nullptr;
  for (const auto &attachment : module.moduleBoundaryTransportAttachments()) {
    if (attachment.boundary.direction != direction ||
        attachment.boundary.ordinal != ordinal)
      continue;
    require(test, result == nullptr,
            "Module boundary endpoint has duplicate attachments");
    result = &attachment.endpoint;
  }
  require(test, result != nullptr, "Module boundary endpoint is unattached");
  return *result;
}

struct ConfigurationImages final {
  loom::hardware::test::PortableConfigurationTarget target;
  loom::hardware::ProgrammingUnitId unitId = 0;
  std::uint64_t bitCount = 0;
  std::vector<std::uint8_t> inactive;
  std::vector<std::uint8_t> route;
  std::vector<std::uint8_t> discard;
};

ConfigurationImages makeConfigurationImages(llvm::StringRef test,
                                            const SystemFixture &fixture) {
  const auto &module = fixture.module.view();
  require(test,
          module.peOccurrences().size() == 1 &&
              module.fuOccurrences().size() == 1,
          "configuration fixture changed its PE/FU shape");
  const auto pe = module.peOccurrences().front();
  const auto fu = module.fuOccurrences().front();
  auto schema = take(test, module.spatialPeConfigurationSchema(pe));

  std::vector<loom::hardware::SemanticConfigurationValue> routeValues;
  std::vector<loom::hardware::SemanticConfigurationValue> discardValues;
  const loom::hardware::ProgrammingUnit *owner = nullptr;
  for (const auto &descriptor : schema.fields()) {
    loom::fabric::FabricPeConfigurationValue routeValue;
    loom::fabric::FabricPeConfigurationValue discardValue;
    if (descriptor.kind ==
        loom::fabric::FabricPeConfigurationFieldKind::Activation) {
      routeValue = loom::fabric::FabricPeActive{fu};
      discardValue = loom::fabric::FabricPeActive{fu};
    } else {
      require(test, descriptor.port.has_value(),
              "selector field has no FU port");
      const auto &port = *descriptor.port;
      const auto &endpoint =
          boundaryEndpoint(test, module, port.direction, port.ordinal);
      routeValue = loom::fabric::FabricPeRoute{endpoint};
      if (descriptor.kind ==
              loom::fabric::FabricPeConfigurationFieldKind::InputSelector &&
          port.ordinal == 0)
        discardValue = loom::fabric::FabricPeInputDiscard{endpoint};
      else if (descriptor.kind ==
               loom::fabric::FabricPeConfigurationFieldKind::InputSelector)
        discardValue = loom::fabric::FabricPeDisconnected{};
      else
        discardValue = loom::fabric::FabricPeRoute{endpoint};
    }

    const auto physical = qualifyConfigurationField(test, fixture.spatialCore,
                                                    descriptor.reference);
    const auto slot =
        take(test,
             loom::fabric::qualifyFabricConfigurationSlot(
                 physical, loom::fabric::FabricStaticConfigurationResidency{}));
    const loom::hardware::ProgrammingUnit *fieldOwner = nullptr;
    for (const auto &unit : fixture.abi.abi().programmingUnits())
      for (const auto &field : unit.fields)
        if (field.slot == slot) {
          require(test, fieldOwner == nullptr,
                  "configuration field has duplicate programming owners");
          fieldOwner = &unit;
        }
    require(test, fieldOwner != nullptr,
            "configuration field has no programming owner");
    if (owner)
      require(test, owner->id == fieldOwner->id,
              "fixture PE fields span multiple programming units");
    else
      owner = fieldOwner;

    const auto routeBytes =
        take(test, schema.encode(descriptor.reference, routeValue));
    routeValues.push_back(
        {slot, std::vector<std::uint8_t>(routeBytes.bytes().begin(),
                                         routeBytes.bytes().end())});
    const auto discardBytes =
        take(test, schema.encode(descriptor.reference, discardValue));
    discardValues.push_back(
        {slot, std::vector<std::uint8_t>(discardBytes.bytes().begin(),
                                         discardBytes.bytes().end())});
  }
  auto fuActivation =
      take(test, loom::hardware::test::deriveSpatialSingleTemplateFuActivation(
                     module, fixture.abi, fixture.spatialCore, fu));
  const auto *fuOwner =
      fixture.abi.abi().findProgrammingUnit(fuActivation.unitId);
  require(test, fuOwner != nullptr, "FU activation has no programming owner");
  if (owner)
    require(test, owner->id == fuOwner->id,
            "fixture configuration spans programming units");
  else
    owner = fuOwner;
  routeValues.push_back(fuActivation.value);
  discardValues.push_back(std::move(fuActivation.value));
  require(test, owner != nullptr, "fixture has no programming unit");
  return ConfigurationImages{
      take(test, loom::hardware::test::derivePortableConfigurationTarget(
                     fixture.abi, fixture.spatialCore, owner->id)),
      owner->id,
      owner->payloadBitCount,
      take(test, fixture.abi.abi().encode(owner->id, {})),
      take(test, fixture.abi.abi().encode(owner->id, routeValues)),
      take(test, fixture.abi.abi().encode(owner->id, discardValues))};
}

SystemFixture makeSystemFixture(llvm::StringRef test,
                                const ArtifactStore &store,
                                FinalizedFabricRoot module,
                                std::uint64_t spatialCoreCount = 1) {
  FinalizedFabricRoot system =
      take(test, loom::hardware::test::makeSpatialCoreSystem(module, store,
                                                             spatialCoreCount));
  auto systemView = take(test, loom::fabric::requireSystemRoot(system.view()));
  require(test,
          systemView.artifact().accCoreOccurrences().size() == spatialCoreCount,
          "test System changed its accelerator core count");
  const loom::fabric::SpatialCoreOccurrenceRef spatialCore{
      systemView.artifact().accCoreOccurrences().front()};
  std::vector<loom::hardware::test::ConfigurationFieldEncodingOverride>
      overrides;
  const auto addDirectOverrides = [&](const auto &occurrences) {
    for (const auto occurrence : occurrences) {
      const loom::fabric::FabricInventoryOwnerRef owner =
          loom::fabric::FabricInventoryOwnerRef::of(occurrence);
      const std::uint64_t fieldCount = module.view().inventorySize(
          owner, loom::fabric::FabricInventoryKind::SemanticConfigField);
      for (std::uint64_t ordinal = 0; ordinal < fieldCount; ++ordinal) {
        const loom::fabric::FabricSemanticConfigFieldRef field{
            loom::fabric::FabricConfigurationOwnerRef(owner), ordinal};
        auto relation =
            take(test, module.view().semanticFieldRelation(
                           field, *const_cast<mlir::Operation *>(
                                       module.view().canonicalOperation())
                                       ->getContext()));
        if (relation.kind() !=
            loom::fabric::FabricSemanticFieldRelationKind::Direct)
          continue;
        const std::uint64_t bitCount = *relation.directEncodedBitCount();
        overrides.push_back(
            {qualifyConfigurationField(test, spatialCore, field),
             loom::hardware::DirectBitsEncoding{bitCount},
             std::vector<std::uint8_t>((bitCount + 7) / 8, 0)});
      }
    }
  };
  addDirectOverrides(module.view().switchOccurrences());
  addDirectOverrides(module.view().boundaryOccurrences());
  addDirectOverrides(module.view().memoryOccurrences());
  addDirectOverrides(module.view().peOccurrences());
  auto abiDraft =
      take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                     system, overrides));
  FinalizedConfigurationABI abi =
      take(test, loom::hardware::finalizeConfigurationABI(std::move(abiDraft),
                                                          store));
  auto operations = take(
      test, loom::hardware::rtl::enumerateFabricPhysicalOperations(systemView));
  return SystemFixture{std::move(module), std::move(system), std::move(abi),
                       spatialCore, std::move(operations)};
}

void spatialHierarchyBuildsStructuralSkeleton() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  SystemFixture fabric =
      makeSystemFixture(test, store, makeSpatialHierarchyFabric(test, store));

  mlir::MLIRContext context;
  context.loadDialect<circt::comb::CombDialect, circt::hw::HWDialect,
                      circt::seq::SeqDialect, circt::sv::SVDialect>();
  auto skeleton = take(test, loom::hardware::rtl::buildModuleRootCirctSkeleton(
                                 context, fabric.spatialCore, fabric.abi));
  require(test, skeleton.operationLeaves.size() == 1,
          "hierarchy skeleton omitted its physical operation");
  std::string text;
  llvm::raw_string_ostream(text) << *skeleton.module;
  require(test,
          llvm::StringRef(text).contains("loom_spatial_pe_") &&
              llvm::StringRef(text).contains("loom_fabric_fu_") &&
              llvm::StringRef(text).contains("loom_fabric_switch_") &&
              llvm::StringRef(text).contains("loom_fabric_fifo_") &&
              llvm::StringRef(text).contains("loom_fabric_boundary_"),
          "hierarchy skeleton flattened or omitted a Fabric resource owner");
}

struct TemporalToolArtifact final {
  std::string systemVerilog;
  loom::hardware::test::PortableConfigurationTarget target;
  std::vector<std::uint8_t> activeImage;
};

TemporalToolArtifact temporalHierarchyBuildsStructuralSkeleton() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  SystemFixture fabric =
      makeSystemFixture(test, store, makeTemporalHierarchyFabric(test, store));

  mlir::MLIRContext context;
  context.loadDialect<circt::comb::CombDialect, circt::hw::HWDialect,
                      circt::seq::SeqDialect, circt::sv::SVDialect>();
  auto skeleton = take(test, loom::hardware::rtl::buildModuleRootCirctSkeleton(
                                 context, fabric.spatialCore, fabric.abi));
  require(test, skeleton.operationLeaves.size() == 1,
          "temporal hierarchy omitted its physical operation");
  FabricOperationProviderRegistry providers;
  if (llvm::Error error =
          loom::hardware::rtl::registerPortableScalarIntegerAddSubProvider(
              providers))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;
  auto conformance = take(
      test, loom::hardware::test::specializeAndExportPortableProvider(
                std::move(skeleton), fabric.abi, providers, externalContracts));
  const llvm::StringRef rtl(conformance.systemVerilog);
  require(test,
          rtl.contains("loom_temporal_pe_") && rtl.contains("operand_pool") &&
              rtl.contains("register_fifo") &&
              rtl.contains("output_0_context") &&
              rtl.contains("result_context_reg"),
          "temporal hierarchy omitted context-local scheduling or storage");

  const auto &module = fabric.module.view();
  require(test,
          module.peOccurrences().size() == 1 &&
              module.fuOccurrences().size() == 1 &&
              fabric.operations.size() == 1,
          "temporal hierarchy fixture changed its PE/FU/operation shape");
  const auto pe = module.peOccurrences().front();
  const auto fu = module.fuOccurrences().front();
  auto schema = take(test, module.temporalPeConfigurationSchema(pe));
  require(test,
          schema.layout().contextCount == 2 &&
              schema.layout().inputPortCount == 2 &&
              schema.layout().outputPortCount == 1,
          "temporal hierarchy fixture changed its direct-carrier shape");

  const auto routeInput = [&](std::uint32_t port) {
    return loom::fabric::FabricTemporalPeOperandSelection{
        loom::fabric::FabricTemporalPeSelectorKind::Route,
        loom::fabric::FabricTemporalPeSelectorTarget{
            loom::fabric::FabricTemporalPePortTarget{port}},
        llvm::APInt(schema.layout().tagWidthBits, 1)};
  };
  loom::fabric::FabricTemporalPeResultSelection routeOutput{
      loom::fabric::FabricTemporalPeSelectorKind::Route,
      loom::fabric::FabricTemporalPeSelectorTarget{
          loom::fabric::FabricTemporalPePortTarget{0}},
      llvm::APInt(schema.layout().tagWidthBits, 2)};
  loom::fabric::FabricTemporalPeActive active;
  active.rows.resize(schema.layout().contextCount);
  active.rows.front() = loom::fabric::FabricTemporalPeInstructionEntry{
      fu, {routeInput(0), routeInput(1)}, {routeOutput}};
  auto peSemantic = take(test, schema.encode(active));

  std::vector<loom::hardware::SemanticConfigurationValue> values;
  const auto pePhysical =
      qualifyConfigurationField(test, fabric.spatialCore, schema.field());
  const auto peSlot =
      take(test,
           loom::fabric::qualifyFabricConfigurationSlot(
               pePhysical, loom::fabric::FabricStaticConfigurationResidency{}));
  const loom::hardware::ProgrammingUnit *owner =
      findProgrammingOwner(test, fabric.abi.abi(), peSlot);
  values.push_back(
      {peSlot, std::vector<std::uint8_t>(peSemantic.bytes().begin(),
                                         peSemantic.bytes().end())});

  const loom::fabric::FabricSemanticConfigFieldRef fuField{
      loom::fabric::FabricConfigurationOwnerRef(
          loom::fabric::FabricInventoryOwnerRef::of(fu)),
      0};
  const auto definition = module.fuTemplateOf(fu);
  require(test, definition.has_value(), "temporal FU has no definition");
  const auto templates = module.fuCapabilityTemplates(*definition);
  require(test, templates.size() == 1,
          "temporal FU fixture changed its capability-template domain");
  auto fuSemantic = take(
      test, loom::fabric::encodeFabricFuConfiguration(
                module, fuField,
                loom::fabric::FabricFuCapabilityTemplateRef{*definition, 0}));
  const auto fuPhysical =
      qualifyConfigurationField(test, fabric.spatialCore, fuField);
  const auto fuSlot =
      take(test, loom::fabric::qualifyFabricConfigurationSlot(
                     fuPhysical, loom::fabric::InstructionContextRef{pe, 0}));
  const auto *fuOwner = findProgrammingOwner(test, fabric.abi.abi(), fuSlot);
  require(test, fuOwner->id == owner->id,
          "temporal PE and FU fields span programming units");
  values.push_back(
      {fuSlot, std::vector<std::uint8_t>(fuSemantic.bytes().begin(),
                                         fuSemantic.bytes().end())});
  require(
      test,
      fabric.operations.front().capability->configurationFieldSchema.empty(),
      "fixed add fixture unexpectedly requires operation configuration");

  return TemporalToolArtifact{
      std::move(conformance.systemVerilog),
      take(test, loom::hardware::test::derivePortableConfigurationTarget(
                     fabric.abi, fabric.spatialCore, owner->id)),
      take(test, fabric.abi.abi().encode(owner->id, values))};
}

void repeatedSpatialCoreBuildsOccurrenceLocalSkeleton() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  SystemFixture fabric =
      makeSystemFixture(test, store, makeOperationFabric(test, store), 2);
  require(test, fabric.operations.size() == 2,
          "repeated Module did not produce two physical operations");
  const auto system =
      take(test, loom::fabric::requireSystemRoot(fabric.system.view()));
  require(test, system.artifact().accCoreOccurrences().size() == 2,
          "repeated Module did not produce two SpatialCores");
  for (const auto core : system.artifact().accCoreOccurrences()) {
    const loom::fabric::SpatialCoreOccurrenceRef spatialCore{core};
    mlir::MLIRContext context;
    context.loadDialect<circt::comb::CombDialect, circt::hw::HWDialect,
                        circt::seq::SeqDialect, circt::sv::SVDialect>();
    auto skeleton =
        take(test, loom::hardware::rtl::buildModuleRootCirctSkeleton(
                       context, spatialCore, fabric.abi));
    require(test, skeleton.operationLeaves.size() == 1,
            "one SpatialCore skeleton covered a foreign occurrence");
    const auto &internal =
        std::get<loom::fabric::SpatialCoreInternalOccurrenceRef>(
            skeleton.operationLeaves.front().occurrence.payload());
    require(test, internal.spatialCore == spatialCore,
            "one SpatialCore skeleton associated a foreign operation");
  }
}

void configurationAbiIncludesFuTopology() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  SystemFixture fabric =
      makeSystemFixture(test, store, makeOperationFabric(test, store));
  require(test, fabric.module.view().fuOccurrences().size() == 1,
          "configuration fixture changed its FU shape");
  const auto fu = fabric.module.view().fuOccurrences().front();
  const loom::fabric::FabricInventoryOwnerRef owner =
      loom::fabric::FabricInventoryOwnerRef::of(fu);
  require(test,
          fabric.module.view().inventorySize(
              owner, loom::fabric::FabricInventoryKind::SemanticConfigField) ==
              1,
          "FU topology is absent from the Fabric configuration inventory");

  const loom::fabric::FabricSemanticConfigFieldRef local{
      loom::fabric::FabricConfigurationOwnerRef(owner), 0};
  const auto physical =
      qualifyConfigurationField(test, fabric.spatialCore, local);
  const auto slot = take(
      test, loom::fabric::qualifyFabricConfigurationSlot(
                physical, loom::fabric::FabricStaticConfigurationResidency{}));
  const auto *encoding = fabric.abi.abi().findField(slot);
  require(test, encoding != nullptr,
          "ConfigurationABI omitted the FU topology slot");
  const auto *codebook = std::get_if<loom::hardware::FiniteCodebookEncoding>(
      &encoding->semanticEncoding);
  require(test, codebook != nullptr && codebook->entries.size() == 2,
          "single-template FU topology does not have Disabled and Active "
          "codes");
}

void commonSkeletonRejectsUnresolvedOrUnboundLeaves() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  SystemFixture fabric =
      makeSystemFixture(test, store, makeOperationFabric(test, store));
  require(test, !fabric.operations.empty(),
          "System has no physical operation occurrence");

  mlir::MLIRContext context;
  context.loadDialect<circt::comb::CombDialect, circt::hw::HWDialect,
                      circt::seq::SeqDialect, circt::sv::SVDialect>();
  mlir::OpBuilder builder(&context);
  const mlir::Location location = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(module->getBody());

  auto schema = circt::hw::HWGeneratorSchemaOp::create(
      builder, location,
      loom::hardware::rtl::fabricOperationGeneratorSchemaSymbol,
      loom::hardware::rtl::fabricOperationGeneratorDescriptor,
      builder.getArrayAttr({}));
  std::vector<circt::hw::HWModuleGeneratedOp> leaves;
  std::vector<FabricOperationLeafAssociation> association;
  for (std::size_t index = 0; index < fabric.operations.size(); ++index) {
    const ResolvedFabricPhysicalOperation &operation = fabric.operations[index];
    auto leaf = circt::hw::HWModuleGeneratedOp::create(
        builder, location,
        mlir::FlatSymbolRefAttr::get(
            &context,
            loom::hardware::rtl::fabricOperationGeneratorSchemaSymbol),
        builder.getStringAttr(
            (llvm::Twine("loom_fabric_operation_") + llvm::Twine(index)).str()),
        take(test, loom::hardware::rtl::deriveFabricOperationLeafPorts(
                       builder, operation.physicalOccurrence,
                       *operation.capability, fabric.abi.abi())));
    leaves.push_back(leaf);
    association.push_back({leaf, operation.physicalOccurrence});
  }
  circt::hw::HWModuleGeneratedOp leaf = leaves.front();
  const llvm::SmallVector<circt::hw::PortInfo> firstLeafPorts =
      leaf.getPortList();
  const std::vector<circt::hw::PortInfo> operationPorts(firstLeafPorts.begin(),
                                                        firstLeafPorts.end());
  circt::hw::HWModuleOp::create(
      builder, location, builder.getStringAttr("loom_common_skeleton_test"),
      circt::hw::ModulePortInfo({}, {}),
      [](mlir::OpBuilder &, circt::hw::HWModulePortAccessor &) {});

  if (llvm::Error error = loom::hardware::rtl::verifyCommonCirctSkeleton(
          *module, fabric.abi.abi(), association))
    fail(test, llvm::toString(std::move(error)));

  const circt::hw::PortInfo unresolvedInput{
      {builder.getStringAttr("input"), builder.getI1Type(),
       circt::hw::ModulePort::Direction::Input}};
  const circt::hw::PortInfo unresolvedOutput{
      {builder.getStringAttr("output"), builder.getI1Type(),
       circt::hw::ModulePort::Direction::Output}};
  auto unresolvedTop = circt::hw::HWModuleOp::create(
      builder, location, builder.getStringAttr("unresolved_structural_top"),
      circt::hw::ModulePortInfo({unresolvedInput}, {unresolvedOutput}),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        llvm::SmallVector<mlir::Type> resultTypes{bodyBuilder.getI1Type()};
        llvm::SmallVector<mlir::Value> operands{accessor.getInput("input")};
        auto unresolved = mlir::UnrealizedConversionCastOp::create(
            bodyBuilder, location, resultTypes, operands);
        accessor.setOutput("output", unresolved.getResult(0));
      });
  expectError(test,
              loom::hardware::rtl::verifyCommonCirctSkeleton(
                  *module, fabric.abi.abi(), association),
              "unresolved structural lowering");
  expectError(test, loom::hardware::rtl::verifySpecializedCirctModule(*module),
              "unresolved structural lowering");
  expectError(
      test,
      loom::hardware::rtl::lowerAndExportSpecializedSystemVerilog(*module),
      "unresolved structural lowering");
  unresolvedTop.erase();

  const circt::hw::ModuleType exactLeafType = leaf.getModuleType();
  std::vector<circt::hw::ModulePort> wrongLeafPorts;
  wrongLeafPorts.reserve(operationPorts.size());
  for (const circt::hw::PortInfo &port : operationPorts)
    wrongLeafPorts.push_back(port);
  wrongLeafPorts.front().type = builder.getI1Type();
  leaf.setModuleType(circt::hw::ModuleType::get(&context, wrongLeafPorts));
  expectError(test,
              loom::hardware::rtl::verifyCommonCirctSkeleton(
                  *module, fabric.abi.abi(), association),
              "does not match its derived contract");
  leaf.setModuleType(exactLeafType);

  SystemFixture foreignFabric =
      makeSystemFixture(test, store, makeBoundaryOnlyFabric(test, store));
  expectError(test,
              loom::hardware::rtl::verifyCommonCirctSkeleton(
                  *module, foreignFabric.abi.abi(), association),
              "does not resolve to a concrete Fabric operation capability");

  expectError(test,
              loom::hardware::rtl::verifyCommonCirctSkeleton(
                  *module, fabric.abi.abi(), {}),
              "has no exact Fabric occurrence association");
  std::vector<FabricOperationLeafAssociation> duplicate = association;
  duplicate.push_back({leaf, fabric.operations.front().physicalOccurrence});
  expectError(test,
              loom::hardware::rtl::verifyCommonCirctSkeleton(
                  *module, fabric.abi.abi(), duplicate),
              "associated more than once");

  auto secondLeaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(
          &context, loom::hardware::rtl::fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr("loom_fabric_operation_1"), operationPorts);
  std::vector<FabricOperationLeafAssociation> duplicateOccurrence = association;
  duplicateOccurrence.push_back(
      {secondLeaf, fabric.operations.front().physicalOccurrence});
  expectError(test,
              loom::hardware::rtl::verifyCommonCirctSkeleton(
                  *module, fabric.abi.abi(), duplicateOccurrence),
              "occurrence is associated more than once");
  secondLeaf.erase();

  mlir::OwningOpRef<mlir::ModuleOp> foreignModule =
      mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(foreignModule->getBody());
  circt::hw::HWGeneratorSchemaOp::create(
      builder, location,
      loom::hardware::rtl::fabricOperationGeneratorSchemaSymbol,
      loom::hardware::rtl::fabricOperationGeneratorDescriptor,
      builder.getArrayAttr({}));
  auto foreignLeaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(
          &context, loom::hardware::rtl::fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr("foreign_fabric_operation"), operationPorts);
  std::vector<FabricOperationLeafAssociation> foreignAssociation = association;
  foreignAssociation.front().module = foreignLeaf;
  expectError(test,
              loom::hardware::rtl::verifyCommonCirctSkeleton(
                  *module, fabric.abi.abi(), foreignAssociation),
              "does not name a Loom leaf in this module");

  schema.setDescriptor("unexpected.fabric.operation");
  expectError(test,
              loom::hardware::rtl::verifyCommonCirctSkeleton(
                  *module, fabric.abi.abi(), association),
              "schema has an unexpected descriptor");
  schema.setDescriptor(loom::hardware::rtl::fabricOperationGeneratorDescriptor);

  SystemFixture twoOccurrence =
      makeSystemFixture(test, store, makeOperationFabric(test, store, true));
  require(test, twoOccurrence.operations.size() == 2,
          "two-operation System changed its operation count");
  std::vector<FabricOperationLeafAssociation> invalid = association;
  invalid.front().occurrence =
      twoOccurrence.operations.back().physicalOccurrence;
  expectError(test,
              loom::hardware::rtl::verifyCommonCirctSkeleton(
                  *module, fabric.abi.abi(), invalid),
              "does not resolve to a concrete Fabric operation capability");

  const ResolvedFabricPhysicalOperation &firstOfTwo =
      twoOccurrence.operations.front();
  mlir::OwningOpRef<mlir::ModuleOp> incompleteModule =
      mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(incompleteModule->getBody());
  circt::hw::HWGeneratorSchemaOp::create(
      builder, location,
      loom::hardware::rtl::fabricOperationGeneratorSchemaSymbol,
      loom::hardware::rtl::fabricOperationGeneratorDescriptor,
      builder.getArrayAttr({}));
  auto incompleteLeaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(
          &context, loom::hardware::rtl::fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr("incomplete_fabric_operation"),
      take(test, loom::hardware::rtl::deriveFabricOperationLeafPorts(
                     builder, firstOfTwo.physicalOccurrence,
                     *firstOfTwo.capability, twoOccurrence.abi.abi())));
  expectError(test,
              loom::hardware::rtl::verifyCommonCirctSkeleton(
                  *incompleteModule, twoOccurrence.abi.abi(),
                  {{incompleteLeaf, firstOfTwo.physicalOccurrence}}),
              "does not exactly cover Fabric operation occurrences");

  expectError(
      test,
      loom::hardware::rtl::lowerAndExportSpecializedSystemVerilog(*module),
      "unresolved Loom Fabric operation leaf");

  for (circt::hw::HWModuleGeneratedOp operationLeaf : leaves)
    operationLeaf.erase();
  const std::string systemVerilog = take(
      test,
      loom::hardware::rtl::lowerAndExportSpecializedSystemVerilog(*module));
  require(test,
          llvm::StringRef(systemVerilog)
              .contains("module loom_common_skeleton_test"),
          "specialized CIRCT module did not export SystemVerilog");
}

std::string moduleBoundaryPassthroughBuildsDeterministicSkeleton() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  SystemFixture fabric =
      makeSystemFixture(test, store, makeBoundaryOnlyFabric(test, store));

  mlir::MLIRContext firstContext;
  firstContext.loadDialect<circt::comb::CombDialect, circt::hw::HWDialect,
                           circt::seq::SeqDialect, circt::sv::SVDialect>();
  auto first = take(test, loom::hardware::rtl::buildModuleRootCirctSkeleton(
                              firstContext, fabric.spatialCore, fabric.abi));
  require(test, first.operationLeaves.empty(),
          "boundary-only skeleton invented an operation leaf");
  if (llvm::Error error = loom::hardware::rtl::verifyCommonCirctSkeleton(
          *first.module, fabric.abi.abi(), first.operationLeaves))
    fail(test, llvm::toString(std::move(error)));

  mlir::MLIRContext secondContext;
  secondContext.loadDialect<circt::comb::CombDialect, circt::hw::HWDialect,
                            circt::seq::SeqDialect, circt::sv::SVDialect>();
  auto second = take(test, loom::hardware::rtl::buildModuleRootCirctSkeleton(
                               secondContext, fabric.spatialCore, fabric.abi));
  std::string firstText;
  std::string secondText;
  llvm::raw_string_ostream(firstText) << *first.module;
  llvm::raw_string_ostream(secondText) << *second.module;
  require(test, firstText == secondText,
          "equal Fabric roots produced different CIRCT skeletons");

  const std::string systemVerilog =
      take(test, loom::hardware::rtl::lowerAndExportSpecializedSystemVerilog(
                     *first.module));
  const llvm::StringRef rtl(systemVerilog);
  require(test,
          rtl.contains("input_0_data") && rtl.contains("input_1_tag") &&
              rtl.contains("output_0_data") && rtl.contains("output_1_tag") &&
              rtl.contains("[15:0]") && rtl.contains("[2:0]"),
          "boundary skeleton omitted canonical transport signals");

  const loom::fabric::SpatialCoreOccurrenceRef invalidSpatialCore{
      loom::fabric::AccCoreOccurrenceRef{fabric.spatialCore.core.id() +
                                         1000000}};
  expectError(test,
              loom::hardware::rtl::buildModuleRootCirctSkeleton(
                  secondContext, invalidSpatialCore, fabric.abi),
              "SpatialCore");
  return systemVerilog;
}

struct InternalToolArtifact final {
  std::string systemVerilog;
  ConfigurationImages configuration;
};

InternalToolArtifact internalOperationBuildsStructuralSkeleton() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  SystemFixture fabric =
      makeSystemFixture(test, store, makeOperationFabric(test, store));
  ConfigurationImages configuration = makeConfigurationImages(test, fabric);

  mlir::MLIRContext context;
  context.loadDialect<circt::comb::CombDialect, circt::hw::HWDialect,
                      circt::seq::SeqDialect, circt::sv::SVDialect>();
  auto skeleton = take(test, loom::hardware::rtl::buildModuleRootCirctSkeleton(
                                 context, fabric.spatialCore, fabric.abi));
  require(test, skeleton.operationLeaves.size() == 1,
          "internal operation skeleton did not expose one exact leaf");
  require(test,
          skeleton.operationLeaves.front().occurrence ==
              fabric.operations.front().physicalOccurrence,
          "internal operation skeleton associated a different occurrence");

  FabricOperationProviderRegistry providers;
  if (llvm::Error error =
          loom::hardware::rtl::registerPortableScalarIntegerAddSubProvider(
              providers))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;
  auto conformance = take(
      test, loom::hardware::test::specializeAndExportPortableProvider(
                std::move(skeleton), fabric.abi, providers, externalContracts));
  const std::string systemVerilog = std::move(conformance.systemVerilog);
  const llvm::StringRef rtl(systemVerilog);
  require(test,
          rtl.contains("input_0_data") && rtl.contains("input_1_data") &&
              rtl.contains("output_0_data") && rtl.contains("clock") &&
              rtl.contains("reset") && rtl.contains("result_data_reg") &&
              rtl.contains("result_valid_reg") && rtl.contains("cfg_awaddr") &&
              rtl.contains("cfg_rdata"),
          "internal operation RTL omitted its structural elastic slot");

  const std::filesystem::path blobRoot =
      std::filesystem::path(directory.path().str()) / "blobs";
  std::error_code directoryError;
  if (!std::filesystem::create_directory(blobRoot, directoryError) ||
      directoryError)
    fail(test, "unable to create the implementation BlobStore: " +
                   directoryError.message());
  loom::BlobStore blobs(blobRoot.string());
  const std::vector<std::uint8_t> rtlBytes(systemVerilog.begin(),
                                           systemVerilog.end());
  const loom::BlobDigest rtlDigest = take(test, blobs.put(rtlBytes));

  auto system =
      take(test, loom::fabric::requireSystemRoot(fabric.system.view()));
  const loom::fabric::FabricInventoryOwnerRef spatialCoreOwner =
      loom::fabric::FabricInventoryOwnerRef::of(fabric.spatialCore);
  std::optional<loom::fabric::HardwareDomainRef> clockDomain;
  std::optional<loom::fabric::HardwareDomainRef> resetDomain;
  for (const loom::fabric::HardwareDomainRef domain :
       system.hardwareDomains()) {
    const auto *contract = system.hardwareDomainContract(domain);
    if (!contract || !llvm::is_contained(contract->members(), spatialCoreOwner))
      continue;
    if (contract->kind() == loom::fabric::FabricHardwareDomainKind::Clock)
      clockDomain = domain;
    else if (contract->kind() == loom::fabric::FabricHardwareDomainKind::Reset)
      resetDomain = domain;
  }
  require(test, clockDomain.has_value() && resetDomain.has_value(),
          "internal operation System has no exact Clock/Reset domains");

  using loom::hardware::ImplementationClockInterfaceRef;
  using loom::hardware::ImplementationConfigurationInterfaceRef;
  using loom::hardware::ImplementationInterface;
  using loom::hardware::ImplementationResetInterfaceRef;
  using loom::hardware::RepresentationLocator;
  using loom::hardware::RepresentationObjectKind;
  std::vector<ImplementationInterface> interfaces{
      {ImplementationClockInterfaceRef{*clockDomain},
       {RepresentationObjectKind::Port, "loom_module.clock"},
       std::nullopt},
      {ImplementationResetInterfaceRef{*resetDomain},
       {RepresentationObjectKind::Port, "loom_module.reset"},
       std::nullopt},
      {ImplementationConfigurationInterfaceRef{
           {fabric.abi.reference(), configuration.unitId}},
       {RepresentationObjectKind::Module, "loom_module"},
       std::nullopt}};
  auto format = take(
      test, loom::hardware::RepresentationFormatDescriptorRef::get(
                loom::hardware::RepresentationFormatKind::SystemVerilogRtl));
  auto representation = take(
      test, loom::hardware::createImplementationRepresentationRoot(
                loom::hardware::RepresentationRootVariant::Rtl, std::nullopt,
                format, {RepresentationObjectKind::Module, "loom_module"},
                {{loom::hardware::PayloadRole::RtlSource,
                  "rtl/internal_module.sv", rtlDigest}}));
  loom::hardware::HardwareImplementationDraft implementationDraft{
      fabric.system.reference(),
      fabric.abi.reference(),
      {},
      std::move(representation),
      std::nullopt,
      std::move(interfaces),
      {{{RepresentationObjectKind::Instance,
         "loom_module.pe_" +
             std::to_string(fabric.module.view().peOccurrences().front().id()) +
             ".fu_" +
             std::to_string(fabric.module.view().fuOccurrences().front().id()) +
             ".operation_" +
             std::to_string(fabric.operations.front().localOccurrence.ordinal)},
        fabric.operations.front().physicalOccurrence}},
      {},
      {}};
  const auto implementation =
      take(test, loom::hardware::finalizeHardwareImplementation(
                     std::move(implementationDraft), store, blobs));
  require(test,
          implementation.implementation().interfaces().size() == 3 &&
              implementation.implementation().activityPoints().size() == 1 &&
              implementation.implementation()
                      .activityPoints()
                      .front()
                      .semanticFabricRef ==
                  fabric.operations.front().physicalOccurrence,
          "internal RTL publication lost its exact System bindings");
  const auto imported =
      take(test, loom::hardware::importHardwareImplementation(
                     implementation.reference(), store, blobs));
  require(test,
          imported.canonicalBytes().bytes() ==
              implementation.canonicalBytes().bytes(),
          "internal RTL HardwareImplementation did not round-trip");
  return InternalToolArtifact{systemVerilog, std::move(configuration)};
}

void writeBoundaryToolArtifacts(const std::filesystem::path &root,
                                llvm::StringRef systemVerilog) {
  std::filesystem::create_directories(root);
  std::ofstream(root / "loom_module.sv") << systemVerilog.str();
  std::ofstream testbench(root / "testbench.sv");
  testbench << R"sv(
module testbench;
  logic        clock;
  logic        reset;
  logic [31:0] input_0_data;
  logic        input_0_valid;
  logic [3:0]  input_1_data;
  logic [4:0]  input_1_tag;
  logic        input_1_valid;
  logic        output_0_ready;
  logic        output_1_ready;
  logic        input_0_ready;
  logic        input_1_ready;
  logic [15:0] output_0_data;
  logic        output_0_valid;
  logic [2:0]  output_1_tag;
  logic        output_1_valid;
  integer      control;

)sv";
  testbench << loom::hardware::test::portableAxiLiteSignalDeclarations();
  testbench << R"sv(

  loom_module dut(.*);

  initial begin
    clock = 0;
    reset = 0;
)sv";
  testbench << loom::hardware::test::portableAxiLiteInitialization();
  testbench << R"sv(
    for (control = 0; control < 16; control = control + 1) begin
      input_0_data = 32'hcafe0000 ^ control;
      input_0_valid = control[3];
      input_1_data = control[3:0];
      input_1_tag = 5'h18 ^ control[4:0];
      input_1_valid = control[2];
      output_0_ready = control[1];
      output_1_ready = control[0];
      #1;
      if (input_0_ready !== output_0_ready ||
          input_1_ready !== output_1_ready ||
          output_0_data !== input_0_data[15:0] ||
          output_0_valid !== input_0_valid ||
          output_1_tag !== input_1_tag[2:0] ||
          output_1_valid !== input_1_valid)
        $fatal(1, "Module boundary passthrough changed transport semantics");
    end
    $finish;
  end
endmodule
)sv";
  std::ofstream(root / "common_skeleton.ys") << R"ys(
read_verilog -sv loom_module.sv
hierarchy -check -top loom_module
check -assert
select -assert-none loom_module/t:$*latch* loom_module/t:$_*LATCH* loom_module/t:$mem*
synth -top loom_module
check -assert
select -assert-none loom_module/t:$*latch* loom_module/t:$_*LATCH* loom_module/t:$mem*
)ys";
}

void writeInternalToolArtifacts(const std::filesystem::path &root,
                                const InternalToolArtifact &artifact) {
  const ConfigurationImages &configuration = artifact.configuration;
  std::ofstream(root / "internal_module.sv") << artifact.systemVerilog;
  std::ofstream testbench(root / "internal_testbench.sv");
  testbench << R"sv(
module internal_testbench;
  logic       clock;
  logic       reset;
  logic [7:0] input_0_data;
  logic       input_0_valid;
  logic       input_0_ready;
  logic [7:0] input_1_data;
  logic       input_1_valid;
  logic       input_1_ready;
  logic [7:0] output_0_data;
  logic       output_0_valid;
  logic       output_0_ready;
)sv";
  testbench << loom::hardware::test::portableAxiLiteSignalDeclarations()
            << "\n";
  testbench << "  loom_module dut(.*);\n\n";
  testbench << R"sv(  always #5 clock = ~clock;

  task check(bit condition, string message);
    if (!condition)
      $fatal(1, "%s", message);
  endtask

)sv";
  testbench << loom::hardware::test::portableAxiLiteDriverTasks();
  testbench << loom::hardware::test::portableCycleWatchdog();
  testbench << R"sv(

  initial begin
    clock = 0;
    reset = 1;
    input_0_data = 8'h05;
    input_1_data = 8'h07;
    input_0_valid = 0;
    input_1_valid = 0;
    output_0_ready = 1;
)sv";
  testbench << loom::hardware::test::portableAxiLiteInitialization();
  testbench << R"sv(    repeat (2) @(posedge clock);
    #1;
    reset = 0;
    #1;
    check(!input_0_ready && !input_1_ready && !output_0_valid,
          "Disabled PE consumed or published a token");

)sv";
  testbench << take("writeInternalToolArtifacts",
                    loom::hardware::test::portableAxiLiteProgramAndVerify(
                        configuration.target, configuration.discard));
  testbench << R"sv(    input_0_valid = 1;
    #1;
    check(input_0_ready && !input_1_ready && !output_0_valid,
          "Input Discard did not drain only its selected PE input");

    input_0_valid = 0;
)sv";
  testbench << take("writeInternalToolArtifacts",
                    loom::hardware::test::portableAxiLiteProgramAndVerify(
                        configuration.target, configuration.route));
  testbench << R"sv(    input_0_valid = 1;
    input_1_valid = 1;
    #1;
    check(input_0_ready && input_1_ready && !output_0_valid,
          "Routed operands were not accepted into an empty slot");
    @(posedge clock);
    #1;
    check(output_0_valid && output_0_data == 8'h0c,
          "Accepted add did not publish exactly one cycle later");

    output_0_ready = 0;
    input_0_data = 8'h09;
    input_1_data = 8'h0a;
    repeat (3) begin
      @(posedge clock);
      #1;
      check(output_0_valid && output_0_data == 8'h0c &&
                !input_0_ready && !input_1_ready,
            "Stalled result or backpressure was not stable");
    end

    output_0_ready = 1;
    #1;
    check(input_0_ready && input_1_ready,
          "Final handoff did not admit a same-cycle replacement");
    @(posedge clock);
    #1;
    check(output_0_valid && output_0_data == 8'h13,
          "Same-cycle replacement did not preserve one-cycle publication");

    input_0_valid = 0;
    input_1_valid = 0;
    @(posedge clock);
    #1;
    check(!output_0_valid, "Released result slot remained valid");
    $finish;
  end
endmodule
)sv";
  std::ofstream(root / "internal_skeleton.ys") << R"ys(
read_verilog -sv internal_module.sv
hierarchy -check -top loom_module
check -assert
proc
select -assert-count 2 loom_operation_shell_0/t:$adff
synth -top loom_module
check -assert
select -assert-none loom_module/t:$dlatch loom_module/t:$_DLATCH_*
)ys";
}

void writeTemporalToolArtifacts(const std::filesystem::path &root,
                                const TemporalToolArtifact &artifact) {
  std::ofstream(root / "temporal_module.sv") << artifact.systemVerilog;
  std::ofstream testbench(root / "temporal_testbench.sv");
  testbench << R"sv(
module temporal_testbench;
  logic       clock;
  logic       reset;
  logic [7:0] input_0_data;
  logic [1:0] input_0_tag;
  logic       input_0_valid;
  logic       input_0_ready;
  logic [7:0] input_1_data;
  logic [1:0] input_1_tag;
  logic       input_1_valid;
  logic       input_1_ready;
  logic [7:0] output_0_data;
  logic [1:0] output_0_tag;
  logic       output_0_valid;
  logic       output_0_ready;
)sv";
  testbench << loom::hardware::test::portableAxiLiteSignalDeclarations()
            << "\n";
  testbench << R"sv(
  loom_module dut(.*);
  always #5 clock = ~clock;

  task automatic check(bit condition, string message);
    if (!condition)
      $fatal(1, "%s", message);
  endtask

  task automatic send_input_0(input logic [7:0] data);
    integer wait_cycles;
    begin
      @(negedge clock);
      input_0_data = data;
      input_0_tag = 2'd1;
      input_0_valid = 1;
      wait_cycles = 0;
      do begin
        @(posedge clock);
        wait_cycles = wait_cycles + 1;
        if (wait_cycles == 32 && !input_0_ready)
          $fatal(1, "Temporal input 0 handshake timed out");
      end while (!input_0_ready);
      @(negedge clock);
      input_0_valid = 0;
    end
  endtask

  task automatic send_input_1(input logic [7:0] data);
    integer wait_cycles;
    begin
      @(negedge clock);
      input_1_data = data;
      input_1_tag = 2'd1;
      input_1_valid = 1;
      wait_cycles = 0;
      do begin
        @(posedge clock);
        wait_cycles = wait_cycles + 1;
        if (wait_cycles == 32 && !input_1_ready)
          $fatal(1, "Temporal input 1 handshake timed out");
      end while (!input_1_ready);
      @(negedge clock);
      input_1_valid = 0;
    end
  endtask

)sv";
  testbench << loom::hardware::test::portableAxiLiteDriverTasks();
  testbench << loom::hardware::test::portableCycleWatchdog();
  testbench << R"sv(

  initial begin
    clock = 0;
    reset = 1;
    input_0_data = 0;
    input_0_tag = 0;
    input_0_valid = 0;
    input_1_data = 0;
    input_1_tag = 0;
    input_1_valid = 0;
    output_0_ready = 0;
)sv";
  testbench << loom::hardware::test::portableAxiLiteInitialization();
  testbench << R"sv(    repeat (2) @(posedge clock);
    #1 reset = 0;
    #1;
    check(!input_0_ready && !input_1_ready && !output_0_valid,
          "Disabled Temporal PE exchanged a token");

)sv";
  testbench << take("writeTemporalToolArtifacts",
                    loom::hardware::test::portableAxiLiteProgramAndVerify(
                        artifact.target, artifact.activeImage));
  testbench << R"sv(    input_0_tag = 2'd3;
    input_0_valid = 1;
    #1;
    check(!input_0_ready,
          "Temporal PE accepted a tag without an active instruction row");
    input_0_valid = 0;

    send_input_0(8'd5);
    repeat (2) begin
      @(posedge clock);
      #1;
      check(!output_0_valid,
            "Temporal operation fired before its independent input arrived");
    end
    send_input_1(8'd7);
    while (!output_0_valid)
      @(posedge clock);
    #1;
    check(output_0_data == 8'd12 && output_0_tag == 2'd2,
          "Temporal operation produced the wrong data or configured tag");
    repeat (3) begin
      @(posedge clock);
      #1;
      check(output_0_valid && output_0_data == 8'd12 &&
                output_0_tag == 2'd2,
            "Stalled Temporal result was not stable");
    end
    output_0_ready = 1;
    @(posedge clock);
    #1;
    check(!output_0_valid, "Consumed Temporal result remained valid");

    fork
      begin : producer
        integer index;
        for (index = 0; index < 4; index = index + 1) begin
          @(negedge clock);
          input_0_data = 8'(10 + index);
          input_1_data = 8'(20 + index);
          input_0_tag = 2'd1;
          input_1_tag = 2'd1;
          input_0_valid = 1;
          input_1_valid = 1;
          @(posedge clock);
          #1;
          check(input_0_ready && input_1_ready,
                "Temporal operand buffers did not sustain one pair per cycle");
        end
        @(negedge clock);
        input_0_valid = 0;
        input_1_valid = 0;
      end
      begin : consumer
        integer index;
        while (!output_0_valid) begin
          @(posedge clock);
          #1;
        end
        for (index = 0; index < 4; index = index + 1) begin
          check(output_0_valid && output_0_data == 8'(30 + 2 * index) &&
                    output_0_tag == 2'd2,
                "Temporal PE did not publish one ordered result per cycle");
          @(posedge clock);
          #1;
        end
      end
    join
    check(!output_0_valid, "Temporal result stream did not terminate");
    $finish;
  end
endmodule
)sv";
  std::ofstream(root / "temporal_skeleton.ys") << R"ys(
read_verilog -sv temporal_module.sv
hierarchy -check -top loom_module
check -assert
proc
synth -top loom_module
check -assert
select -assert-none loom_module/t:$dlatch loom_module/t:$_DLATCH_*
)ys";
}

} // namespace

int main(int argc, char **argv) {
  require("main", argc == 1 || argc == 2,
          "expected at most one output directory");
  spatialHierarchyBuildsStructuralSkeleton();
  const TemporalToolArtifact temporal =
      temporalHierarchyBuildsStructuralSkeleton();
  repeatedSpatialCoreBuildsOccurrenceLocalSkeleton();
  configurationAbiIncludesFuTopology();
  commonSkeletonRejectsUnresolvedOrUnboundLeaves();
  const std::string systemVerilog =
      moduleBoundaryPassthroughBuildsDeterministicSkeleton();
  const InternalToolArtifact internal =
      internalOperationBuildsStructuralSkeleton();
  if (argc == 2) {
    writeBoundaryToolArtifacts(argv[1], systemVerilog);
    writeInternalToolArtifacts(argv[1], internal);
    writeTemporalToolArtifacts(argv[1], temporal);
  }
  return 0;
}
