#include "ADGBuilderTestSupport.h"

#include "Fabric/IR/FabricAttrs.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/FabricTypes.h"
#include "Fabric/IR/FuCapabilityDomain.h"
#include "Fabric/IR/ImplementationFamily.h"
#include "Fabric/IR/MemoryConnectivityContract.h"
#include "Fabric/IR/MemoryOperationPort.h"
#include "Fabric/IR/MemoryServiceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <iterator>
#include <utility>
#include <vector>

namespace loom::adg::test {
namespace {

template <typename T>
void expectRejected(llvm::StringRef test, llvm::Expected<T> result) {
  if (result)
    fail(test, "accepted an invalid typed template handle");
  llvm::consumeError(result.takeError());
}

void assignMember(llvm::StringRef test, SpatialCoreBuilder &root,
                  const ModuleDomainMemberHandle &member,
                  const ModuleDomainSlotHandle &clock,
                  const ModuleDomainSlotHandle &reset) {
  if (llvm::Error error = root.assignDomainSlot(member, clock))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = root.assignDomainSlot(member, reset))
    fail(test, llvm::toString(std::move(error)));
}

ArtifactRootReference buildRepeatedPeFu(llvm::StringRef test,
                                        const ArtifactStore &store,
                                        const PortType &bits32,
                                        bool useTemplates) {
  DesignBuilder design(store);
  auto root = take(test, design.createSpatialCore(
                             "repeated-pe-fu", {bits32, bits32, bits32, bits32},
                             {bits32, bits32, bits32, bits32}));
  const auto clock = take(
      test, root.declareDomainSlot(loom::fabric::FabricClockResetKind::Clock));
  const auto reset = take(
      test, root.declareDomainSlot(loom::fabric::FabricClockResetKind::Reset));
  const auto assign = [&](const ModuleDomainMemberHandle &member) {
    assignMember(test, root, member, clock, reset);
  };
  for (std::size_t ordinal = 0; ordinal != 4; ++ordinal) {
    assign(take(test, root.inputDomainMember(ordinal)));
    assign(take(test, root.outputDomainMember(ordinal)));
  }

  std::vector<SpatialValue> outputs;
  if (!useTemplates) {
    for (std::size_t peOrdinal = 0; peOrdinal != 2; ++peOrdinal) {
      const std::size_t inputBase = peOrdinal * 2;
      auto pe =
          take(test,
               root.addPe({take(test, root.input(inputBase)),
                           take(test, root.input(inputBase + 1))},
                          PeSpec::spatial({bits32, bits32}, {bits32, bits32})));
      for (std::size_t fuOrdinal = 0; fuOrdinal != 2; ++fuOrdinal) {
        auto fu = take(test, pe.addFu({take(test, pe.input(fuOrdinal))},
                                      FuSpec{{bits32}, {bits32}}));
        auto add = take(
            test, fu.addOperation(
                      {take(test, fu.input(0)), take(test, fu.input(0))},
                      integerCapability(
                          ::fabric::ImplementationFamilyId::ScalarIntegerAddSub,
                          ::dataflow::OperationSchemaId::ArithAddI, bits32)));
        if (llvm::Error error =
                fu.addCapabilityTemplate(FuCapabilityTemplateSpec{{add}, {}}))
          fail(test, llvm::toString(std::move(error)));
        if (llvm::Error error = fu.close({take(test, add.output(0))}))
          fail(test, llvm::toString(std::move(error)));
        assign(fu.domainMember());
        assign(add.domainMember());
      }
      if (llvm::Error error = pe.close())
        fail(test, llvm::toString(std::move(error)));
      assign(pe.domainMember());
      assign(take(test, pe.instructionContextMember(0)));
      outputs.push_back(take(test, pe.output(0)));
      outputs.push_back(take(test, pe.output(1)));
    }
  } else {
    auto pe =
        take(test, root.createPeTemplate(
                       "pe", {bits32, bits32},
                       PeSpec::spatial({bits32, bits32}, {bits32, bits32})));
    auto fu = take(test, pe.createFuTemplate("fu", FuSpec{{bits32}, {bits32}}));
    auto add = take(
        test, fu.addOperation(
                  {take(test, fu.input(0)), take(test, fu.input(0))},
                  integerCapability(
                      ::fabric::ImplementationFamilyId::ScalarIntegerAddSub,
                      ::dataflow::OperationSchemaId::ArithAddI, bits32)));
    const auto addOwner = take(test, add.templateOwner());
    if (llvm::Error error =
            fu.addCapabilityTemplate(FuCapabilityTemplateSpec{{add}, {}}))
      fail(test, llvm::toString(std::move(error)));
    const auto fuTemplate =
        take(test, fu.closeTemplate({take(test, add.output(0))}));
    auto firstFu =
        take(test, pe.instantiate(fuTemplate, {take(test, pe.input(0))}));
    auto secondFu =
        take(test, pe.instantiate(fuTemplate, {take(test, pe.input(1))}));
    expectRejected(test, firstFu.project(secondFu.occurrenceOwner()));
    const auto peTemplate = take(test, pe.closeTemplate());

    const auto assignPeUse = [&](const SpatialTemplateInstanceResult &use) {
      assign(take(test, root.moduleMember(use.occurrenceOwner())));
      assign(take(
          test, root.moduleMember(take(
                    test, use.project(take(
                              test, peTemplate.instructionContextOwner(0)))))));
      for (const PeTemplateInstanceResult *fuUse : {&firstFu, &secondFu}) {
        assign(take(test, root.moduleMember(take(
                              test, use.project(fuUse->occurrenceOwner())))));
        assign(take(
            test,
            root.moduleMember(take(
                test, use.project(take(test, fuUse->project(addOwner)))))));
      }
    };

    auto firstPe =
        take(test, root.instantiate(peTemplate, {take(test, root.input(0)),
                                                 take(test, root.input(1))}));
    auto secondPe =
        take(test, root.instantiate(peTemplate, {take(test, root.input(2)),
                                                 take(test, root.input(3))}));
    assignPeUse(firstPe);
    assignPeUse(secondPe);
    outputs.assign(firstPe.values().begin(), firstPe.values().end());
    outputs.insert(outputs.end(), secondPe.values().begin(),
                   secondPe.values().end());
  }

  if (llvm::Error error = root.close(outputs))
    fail(test, llvm::toString(std::move(error)));
  auto finalized = take(test, std::move(design).finalize());
  const auto &view = finalized.roots().front().view();
  require(
      test,
      entityCount(view, loom::fabric::FabricEntityKind::FabricPeOccurrence) ==
              2 &&
          entityCount(view,
                      loom::fabric::FabricEntityKind::FabricFuOccurrence) == 4,
      "repeated template use lost an exact physical occurrence");
  return finalized.roots().front().reference();
}

void repeatedPeFuTemplatesHaveExactOccurrenceIdentity() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  const PortType bits32 = take(test, PortType::bits(32));
  const ArtifactRootReference direct =
      buildRepeatedPeFu(test, store, bits32, false);
  const ArtifactRootReference instantiated =
      buildRepeatedPeFu(test, store, bits32, true);
  require(test, direct == instantiated,
          "repeated PE or FU template use changed canonical identity");
}

void switchTemplatesMatchIndependentFabricMlir() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  const PortType bits32 = take(test, PortType::bits(32));

  DesignBuilder design(store);
  auto root =
      take(test, design.createSpatialCore("independent-switch-oracle",
                                          {bits32, bits32}, {bits32, bits32}));
  const auto target =
      take(test, root.createSwitchTemplate(
                     "switch", SwitchSpec::spatial({bits32}, {bits32}, {{0}})));
  auto first =
      take(test, root.instantiate(target, {take(test, root.input(0))}));
  auto second =
      take(test, root.instantiate(target, {take(test, root.input(1))}));
  if (llvm::Error error = root.close({first[0], second[0]}))
    fail(test, llvm::toString(std::move(error)));
  const auto authored = take(test, std::move(design).finalize());

  mlir::DialectRegistry registry;
  registry.insert<::fabric::FabricDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      fabric.module @independent_switch_oracle(
          %a: !fabric.bits<32>, %b: !fabric.bits<32>)
          -> (!fabric.bits<32>, !fabric.bits<32>) {
        %x = fabric.switch [spatial] %a
          [{connectivity_table = ["1"]}]
          : (!fabric.bits<32>) -> !fabric.bits<32>
        %y = fabric.switch [spatial] %b
          [{connectivity_table = ["1"]}]
          : (!fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %x, %y : !fabric.bits<32>, !fabric.bits<32>
      }
    }
  )mlir",
                                                        &context);
  require(test, static_cast<bool>(source),
          "independent Fabric MLIR oracle did not parse");
  auto roots = source->getOps<::fabric::ModuleOp>();
  require(test, std::distance(roots.begin(), roots.end()) == 1,
          "independent Fabric MLIR oracle has the wrong root count");
  auto independent =
      take(test, loom::fabric::finalizeFabricRoot(*roots.begin(), store));
  require(test, authored.roots().front().reference() == independent.reference(),
          "Builder template encoding disagrees with independent Fabric MLIR");
}

void templateHandlesRejectForeignAndStaleUseAtomically() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  const PortType bits32 = take(test, PortType::bits(32));
  const SwitchSpec identity = SwitchSpec::spatial({bits32}, {bits32}, {{0}});

  DesignBuilder firstDesign(store);
  auto firstRoot = take(
      test, firstDesign.createSpatialCore("handle-first", {bits32}, {bits32}));
  auto secondRoot = take(
      test, firstDesign.createSpatialCore("handle-second", {bits32}, {bits32}));
  const auto firstTarget =
      take(test, firstRoot.createSwitchTemplate("switch", identity));
  const auto alternateTarget =
      take(test, firstRoot.createSwitchTemplate("alternate", identity));
  const auto secondTarget =
      take(test, secondRoot.createSwitchTemplate("switch", identity));

  expectRejected(test, secondRoot.instantiate(
                           firstTarget, {take(test, secondRoot.input(0))}));
  auto secondUse =
      take(test, secondRoot.instantiate(secondTarget,
                                        {take(test, secondRoot.input(0))}));
  auto firstUse =
      take(test, firstRoot.instantiate(firstTarget,
                                       {take(test, firstRoot.input(0))}));
  expectRejected(test, firstUse.project(alternateTarget.occurrenceOwner()));
  expectRejected(test, firstRoot.moduleMember(firstTarget.occurrenceOwner()));
  const auto projected =
      take(test, firstUse.project(firstTarget.occurrenceOwner()));
  expectRejected(test, firstUse.project(projected));

  DesignBuilder foreignDesign(store);
  auto foreignRoot = take(test, foreignDesign.createSpatialCore(
                                    "handle-foreign", {bits32}, {bits32}));
  expectRejected(test, foreignRoot.instantiate(
                           firstTarget, {take(test, foreignRoot.input(0))}));
  const auto foreignTarget =
      take(test, foreignRoot.createSwitchTemplate("switch", identity));
  auto foreignUse =
      take(test, foreignRoot.instantiate(foreignTarget,
                                         {take(test, foreignRoot.input(0))}));

  if (llvm::Error error = firstRoot.close(firstUse.values()))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = secondRoot.close(secondUse.values()))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = foreignRoot.close(foreignUse.values()))
    fail(test, llvm::toString(std::move(error)));
  const auto firstFinalized = take(test, std::move(firstDesign).finalize());
  const auto foreignFinalized = take(test, std::move(foreignDesign).finalize());
  require(test,
          firstFinalized.roots()[0].reference() ==
                  firstFinalized.roots()[1].reference() &&
              firstFinalized.roots()[0].reference() ==
                  foreignFinalized.roots()[0].reference(),
          "rejected foreign use changed the surviving design identity");
  expectRejected(test, firstUse.project(firstTarget.occurrenceOwner()));
  expectRejected(test, firstRoot.moduleMember(firstUse.occurrenceOwner()));
}

} // namespace

ArtifactRootReference
buildIndependentNonModuleTemplateOracle(llvm::StringRef test,
                                        const ArtifactStore &store) {
  mlir::DialectRegistry registry;
  registry.insert<::fabric::FabricDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  const mlir::Location location = mlir::UnknownLoc::get(&context);
  auto source = mlir::ModuleOp::create(location);
  mlir::OpBuilder builder(&context);
  builder.setInsertionPointToEnd(source.getBody());

  const mlir::Type bits32 = ::fabric::BitsType::get(&context, 32);
  const mlir::Type tagged0 = ::fabric::BitsTagType::get(&context, 0, 4);
  const mlir::Type tagged32 = ::fabric::BitsTagType::get(&context, 32, 4);
  const llvm::SmallVector<mlir::Type, 6> rootInputs = {
      bits32, bits32, bits32, bits32, tagged32, tagged0};
  const llvm::SmallVector<mlir::Type, 4> rootOutputs = {bits32, bits32,
                                                        tagged32, tagged0};
  const mlir::FunctionType rootType =
      mlir::FunctionType::get(&context, rootInputs, rootOutputs);
  auto root = ::fabric::ModuleOp::create(
      builder, location, "non-module-template-equivalence", rootType,
      mlir::IntegerAttr(), mlir::IntegerAttr(), mlir::ArrayAttr(),
      mlir::ArrayAttr());
  mlir::Block *rootBody = new mlir::Block();
  root.getBody().push_back(rootBody);
  for (mlir::Type type : rootInputs)
    rootBody->addArgument(type, location);

  builder.setInsertionPointToEnd(rootBody);
  mlir::Value peInputs[] = {rootBody->getArgument(0), rootBody->getArgument(1)};
  auto pe = ::fabric::PeOp::create(
      builder, location, mlir::TypeRange{bits32}, mlir::StringAttr(),
      mlir::TypeAttr(), ::fabric::Schedule::Spatial, peInputs,
      mlir::IntegerAttr(), mlir::IntegerAttr(), mlir::IntegerAttr(),
      mlir::IntegerAttr(), mlir::IntegerAttr(), ::fabric::FuConfigModeAttr(),
      ::fabric::OperandBufferModeAttr(), mlir::IntegerAttr());
  mlir::Block *peBody = new mlir::Block();
  pe.getBody().push_back(peBody);
  peBody->addArgument(bits32, location);
  peBody->addArgument(bits32, location);

  builder.setInsertionPointToEnd(peBody);
  mlir::Value fuInputs[] = {peBody->getArgument(0), peBody->getArgument(1)};
  auto fu = ::fabric::FuOp::create(
      builder, location, mlir::TypeRange{bits32}, mlir::StringAttr(),
      mlir::TypeAttr(), ::fabric::FuCapabilityDomainAttr(), fuInputs);
  mlir::Block *fuBody = new mlir::Block();
  fu.getBody().push_back(fuBody);
  fuBody->addArgument(bits32, location);
  fuBody->addArgument(bits32, location);

  const PortType builderBits32 = take(test, PortType::bits(32));
  const OperationCapabilitySpec capability = integerCapability(
      ::fabric::ImplementationFamilyId::ScalarIntegerAddSub,
      ::dataflow::OperationSchemaId::ArithAddI, builderBits32);
  mlir::Attribute operationName = mlir::FlatSymbolRefAttr::get(
      &context, ::dataflow::operationSchemaSpelling(
                    ::dataflow::OperationSchemaId::ArithAddI));
  builder.setInsertionPointToEnd(fuBody);
  mlir::Value operationInputs[] = {fuBody->getArgument(0),
                                   fuBody->getArgument(1)};
  auto operation = ::fabric::OpOp::create(
      builder, location, mlir::TypeRange{bits32}, operationInputs,
      ::fabric::ImplementationFamilyIdAttr::get(
          &context, ::fabric::ImplementationFamilyId::ScalarIntegerAddSub),
      mlir::ArrayAttr::get(&context, {operationName}),
      ::fabric::getFamilyCapabilityParamsAttr(&context,
                                              capability.hardwareParameters));
  auto resourceBytes =
      take(test,
           ::fabric::encodeResourceContractRecord(capability.resourceContract));
  llvm::SmallVector<std::int8_t, 64> signedResourceBytes;
  for (std::uint8_t byte : resourceBytes)
    signedResourceBytes.push_back(static_cast<std::int8_t>(byte));
  operation->setAttr(
      ::fabric::kResourceContractRecordAttrName,
      mlir::DenseI8ArrayAttr::get(&context, signedResourceBytes));
  ::fabric::FuCapabilityTemplateSelection selection;
  selection.activeOperationNodeOrdinals = {0};
  auto capabilityDomain =
      take(test, ::fabric::FuCapabilityDomainRecord::create({selection}));
  auto capabilityBytes =
      take(test, ::fabric::encodeFuCapabilityDomainRecord(capabilityDomain));
  llvm::SmallVector<std::int8_t, 64> signedCapabilityBytes;
  for (std::uint8_t byte : capabilityBytes)
    signedCapabilityBytes.push_back(static_cast<std::int8_t>(byte));
  fu.setCapabilityTemplatesAttr(::fabric::FuCapabilityDomainAttr::get(
      &context, mlir::DenseI8ArrayAttr::get(&context, signedCapabilityBytes)));
  ::fabric::YieldOp::create(builder, location, operation.getResults());

  builder.setInsertionPointToEnd(rootBody);
  mlir::NamedAttrList switchHardware;
  switchHardware.set(
      "connectivity_table",
      mlir::ArrayAttr::get(&context, {mlir::StringAttr::get(&context, "11")}));
  auto switchHardwareParameters =
      mlir::ArrayAttr::get(&context, {switchHardware.getDictionary(&context)});
  mlir::Value switchInputs[] = {rootBody->getArgument(2),
                                rootBody->getArgument(3)};
  auto fabricSwitch = ::fabric::SwitchOp::create(
      builder, location, mlir::TypeRange{bits32}, switchInputs,
      mlir::StringAttr(), mlir::TypeAttr(), ::fabric::Schedule::Spatial,
      llvm::ArrayRef<mlir::Type>{}, switchHardwareParameters,
      mlir::DictionaryAttr());

  const llvm::SmallVector<mlir::Type, 2> memoryInputs = {tagged32, tagged0};
  const llvm::SmallVector<mlir::Type, 2> memoryOutputs = {tagged32, tagged0};
  const mlir::FunctionType memoryType =
      mlir::FunctionType::get(&context, memoryInputs, memoryOutputs);
  auto endpoints =
      take(test, ::fabric::deriveMemoryTransportEndpointInventory(memoryType));
  const auto portDeclaration = loadPortDeclaration();
  auto portRecord =
      take(test, ::fabric::MemoryOperationPortRecord::fromCanonical(
                     &context, ::fabric::Schedule::Temporal, endpoints,
                     portDeclaration));
  auto portBytes =
      take(test, ::fabric::encodeMemoryOperationPortRecord(portRecord));
  llvm::SmallVector<std::int8_t, 64> signedPortBytes;
  for (std::uint8_t byte : portBytes)
    signedPortBytes.push_back(static_cast<std::int8_t>(byte));
  const mlir::ArrayAttr operationPorts = mlir::ArrayAttr::get(
      &context, {mlir::DenseI8ArrayAttr::get(&context, signedPortBytes)});

  const auto serviceRecord = localMemoryContract(test, context);
  auto serviceBytes =
      take(test, ::fabric::encodeMemoryServiceContractRecord(serviceRecord));
  llvm::SmallVector<std::int8_t, 64> signedServiceBytes;
  for (std::uint8_t byte : serviceBytes)
    signedServiceBytes.push_back(static_cast<std::int8_t>(byte));
  auto serviceContract = ::fabric::MemoryServiceContractAttr::get(
      &context, mlir::DenseI8ArrayAttr::get(&context, signedServiceBytes));
  auto localService =
      ::fabric::LocalMemoryServiceAttr::get(&context, 4096, serviceContract);

  ::fabric::MemoryConnectivityDeclaration connectivityDeclaration;
  connectivityDeclaration.operationPorts = {{{{localMemoryTarget()}}}};
  auto connectivityRecord =
      take(test, ::fabric::MemoryConnectivityContractRecord::create(
                     std::move(connectivityDeclaration)));
  auto connectivityBytes = take(
      test,
      ::fabric::encodeMemoryConnectivityContractRecord(connectivityRecord));
  llvm::SmallVector<std::int8_t, 64> signedConnectivityBytes;
  for (std::uint8_t byte : connectivityBytes)
    signedConnectivityBytes.push_back(static_cast<std::int8_t>(byte));
  auto connectivity = ::fabric::MemoryConnectivityContractAttr::get(
      &context, mlir::DenseI8ArrayAttr::get(&context, signedConnectivityBytes));
  auto memoryEngine = ::fabric::MemoryEngineAttr::get(
      &context, ::fabric::Schedule::Temporal,
      ::fabric::MemoryResidentContextsAttr::get(&context, 4));
  const auto emptyOrdinals =
      mlir::DenseI32ArrayAttr::get(&context, llvm::ArrayRef<std::int32_t>{});
  auto memoryContract = ::fabric::MemoryContractAttr::get(
      &context, memoryEngine, localService, connectivity, emptyOrdinals,
      emptyOrdinals);
  mlir::Value memoryOperands[] = {rootBody->getArgument(4),
                                  rootBody->getArgument(5)};
  auto memory = ::fabric::MemOp::create(
      builder, location, memoryOutputs, memoryOperands, mlir::StringAttr(),
      mlir::TypeAttr(), memoryContract, llvm::ArrayRef<mlir::Type>{},
      mlir::ArrayAttr(), operationPorts);

  mlir::Value results[] = {pe.getResult(0), fabricSwitch.getResult(0),
                           memory.getResult(0), memory.getResult(1)};
  ::fabric::YieldOp::create(builder, location, results);
  if (mlir::failed(mlir::verify(source)))
    fail(test, "independent Fabric MLIR oracle failed verification");

  std::string assembly;
  llvm::raw_string_ostream stream(assembly);
  source.print(stream);
  stream.flush();
  auto parsed = mlir::parseSourceString<mlir::ModuleOp>(assembly, &context);
  require(test, static_cast<bool>(parsed),
          "independent Fabric MLIR oracle did not parse");
  auto roots = parsed->getOps<::fabric::ModuleOp>();
  require(test, std::distance(roots.begin(), roots.end()) == 1,
          "independent Fabric MLIR oracle has the wrong root count");
  return take(test, loom::fabric::finalizeFabricRoot(*roots.begin(), store))
      .reference();
}

void runBuilderTemplateTests() {
  repeatedPeFuTemplatesHaveExactOccurrenceIdentity();
  switchTemplatesMatchIndependentFabricMlir();
  templateHandlesRejectForeignAndStaleUseAtomically();
}

} // namespace loom::adg::test
