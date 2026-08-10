#include "Hardware/RTL/SystemSkeleton.h"

#include "Hierarchy/Support.h"

#include "Hardware/RTL/CommonSkeleton.h"
#include "Hardware/RTL/ConfigurationTransport.h"

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace loom::hardware::rtl {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "rtl_system_skeleton_invalid: " + message);
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendFramed(std::vector<std::uint8_t> &bytes,
                  llvm::ArrayRef<std::uint8_t> value) {
  appendU64(bytes, value.size());
  bytes.insert(bytes.end(), value.begin(), value.end());
}

void appendFramed(std::vector<std::uint8_t> &bytes, llvm::StringRef value) {
  appendFramed(bytes, llvm::ArrayRef<std::uint8_t>(
                          reinterpret_cast<const std::uint8_t *>(value.data()),
                          value.size()));
}

std::string moduleText(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  return text;
}

llvm::Expected<fabric::HardwareDomainRef>
findDomain(const fabric::FabricSystemRootView &system,
           fabric::SpatialCoreOccurrenceRef spatialCore,
           fabric::FabricHardwareDomainKind kind) {
  const fabric::FabricInventoryOwnerRef owner =
      fabric::FabricInventoryOwnerRef::of(spatialCore);
  std::optional<fabric::HardwareDomainRef> result;
  for (fabric::HardwareDomainRef domain : system.hardwareDomains()) {
    const auto *contract = system.hardwareDomainContract(domain);
    if (!contract || contract->kind() != kind ||
        !llvm::is_contained(contract->members(), owner))
      continue;
    if (result)
      return invalid("SpatialCore belongs to more than one required domain");
    result = domain;
  }
  if (!result)
    return invalid("SpatialCore has no required Clock or Reset domain");
  return *result;
}

llvm::Error
appendDefinitionDomainKey(std::vector<std::uint8_t> &key,
                          const fabric::FabricSystemRootView &system,
                          fabric::HardwareDomainRef domain,
                          fabric::FabricHardwareDomainKind expectedKind) {
  const auto *contract = system.hardwareDomainContract(domain);
  if (!contract || contract->kind() != expectedKind)
    return invalid("definition domain does not resolve with the required kind");
  appendU64(key, static_cast<std::uint64_t>(expectedKind));
  if (expectedKind == fabric::FabricHardwareDomainKind::Clock) {
    const auto *clock =
        std::get_if<fabric::ClockDomainContractRecord>(&contract->contract());
    if (!clock)
      return invalid("Clock domain has no Clock contract");
    appendU64(key, clock->periodFs());
    appendU64(key, clock->phaseFs());
    return llvm::Error::success();
  }
  const auto *reset =
      std::get_if<fabric::ResetDomainContractRecord>(&contract->contract());
  if (!reset)
    return invalid("Reset domain has no Reset contract");
  appendU64(key, static_cast<std::uint64_t>(reset->polarity()));
  appendU64(key, static_cast<std::uint64_t>(reset->assertion()));
  appendU64(key, static_cast<std::uint64_t>(reset->deassertion()));
  appendU64(key, static_cast<std::uint64_t>(reset->initialState()));
  appendU64(key, reset->synchronousTo().has_value());
  appendU64(key, reset->releaseLatencyCycles());
  return llvm::Error::success();
}

llvm::Expected<std::vector<std::uint8_t>>
definitionKey(const fabric::FabricSystemRootView &system,
              fabric::SpatialCoreOccurrenceRef spatialCore,
              fabric::HardwareDomainRef clockDomain,
              fabric::HardwareDomainRef resetDomain, mlir::ModuleOp module) {
  const auto target = system.spatialCoreTarget(spatialCore.core);
  if (!target ||
      target->dependencyOrdinal >= system.artifact().importedModules().size())
    return invalid("SpatialCore imported Module target does not resolve");
  const fabric::FabricArtifactView &imported =
      system.artifact().importedModules()[target->dependencyOrdinal];
  if (imported.moduleRootTemplate() != target->target)
    return invalid("SpatialCore target and imported Module disagree");

  std::vector<std::uint8_t> key;
  appendFramed(key, imported.identity().bytes());
  if (llvm::Error error = appendDefinitionDomainKey(
          key, system, clockDomain, fabric::FabricHardwareDomainKind::Clock))
    return std::move(error);
  if (llvm::Error error = appendDefinitionDomainKey(
          key, system, resetDomain, fabric::FabricHardwareDomainKind::Reset))
    return std::move(error);
  appendFramed(key, moduleText(module));
  return key;
}

llvm::Expected<circt::hw::HWModuleOp>
renameDefinition(mlir::ModuleOp module, std::size_t definitionOrdinal) {
  circt::hw::HWModuleOp root =
      module.lookupSymbol<circt::hw::HWModuleOp>("loom_module");
  if (!root)
    return invalid("SpatialCore skeleton has no loom_module root");

  std::vector<mlir::Operation *> symbols;
  std::vector<mlir::Operation *> schemas;
  for (mlir::Operation &operation : *module.getBody()) {
    if (llvm::isa<circt::hw::HWGeneratorSchemaOp>(operation)) {
      schemas.push_back(&operation);
      continue;
    }
    if (mlir::SymbolTable::getSymbolName(&operation))
      symbols.push_back(&operation);
  }
  for (mlir::Operation *schema : schemas)
    schema->erase();

  const std::string prefix =
      "loom_spatial_definition_" + std::to_string(definitionOrdinal);
  for (mlir::Operation *symbol : symbols) {
    const mlir::StringAttr oldName = mlir::SymbolTable::getSymbolName(symbol);
    const std::string replacement =
        symbol == root.getOperation()
            ? prefix
            : prefix + "__" + oldName.getValue().str();
    const mlir::StringAttr newName =
        mlir::StringAttr::get(module.getContext(), replacement);
    if (mlir::failed(mlir::SymbolTable::replaceAllSymbolUses(
            oldName, newName, module.getOperation())))
      return invalid("cannot rebase SpatialCore definition symbol uses");
    symbol->setAttr(mlir::SymbolTable::getSymbolAttrName(), newName);
  }
  return root;
}

std::string corePrefix(fabric::SpatialCoreOccurrenceRef core) {
  return "core_" + std::to_string(core.core.id());
}

std::string exposedPortName(fabric::SpatialCoreOccurrenceRef core,
                            llvm::StringRef localPort) {
  return corePrefix(core) + "_" + localPort.str();
}

std::string domainPortName(fabric::HardwareDomainRef domain,
                           fabric::FabricHardwareDomainKind kind) {
  return (kind == fabric::FabricHardwareDomainKind::Clock ? "clock_"
                                                          : "reset_") +
         std::to_string(domain.id());
}

circt::hw::PortInfo renamedPort(mlir::OpBuilder &builder,
                                const circt::hw::PortInfo &port,
                                llvm::StringRef name) {
  return circt::hw::PortInfo{{builder.getStringAttr(name), port.type,
                              port.isOutput()
                                  ? circt::hw::ModulePort::Direction::Output
                                  : circt::hw::ModulePort::Direction::Input}};
}

llvm::Expected<fabric::SpatialCoreOccurrenceRef>
attachmentCore(const fabric::FabricSpatialAttachmentEndpointRef &endpoint) {
  if (const auto *transport = endpoint.transport()) {
    if (transport->owner.kind() !=
        fabric::FabricTransportEndpointOwnerKind::SpatialCoreOccurrence)
      return invalid("System transport attachment is not SpatialCore-owned");
    return std::get<fabric::SpatialCoreOccurrenceRef>(transport->owner.payload);
  }
  const auto *memory = endpoint.memory();
  if (!memory ||
      memory->owner.kind() !=
          fabric::FabricMemoryEndpointOwnerKind::SpatialCoreOccurrence)
    return invalid("System memory attachment is not SpatialCore-owned");
  return std::get<fabric::SpatialCoreOccurrenceRef>(memory->owner.payload);
}

std::string attachmentLocalPort(
    const fabric::FabricSpatialAttachmentRecordView &attachment) {
  const bool input = attachment.moduleEndpoint.target.direction ==
                     fabric::FabricPortDirection::Input;
  const std::string direction = input ? "input_" : "output_";
  const std::string ordinal =
      std::to_string(attachment.moduleEndpoint.target.ordinal);
  if (attachment.spatialEndpoint.transport())
    return direction + ordinal + "_valid";
  return "memory_" + direction + ordinal + "_request_valid";
}

ImplementationInterface
topPortInterface(ImplementationInterfaceSemanticRef semanticRef,
                 llvm::StringRef port) {
  return ImplementationInterface{
      std::move(semanticRef),
      {RepresentationObjectKind::Port, "loom_system." + port.str()},
      std::nullopt};
}

struct Definition final {
  std::vector<std::uint8_t> key;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  circt::hw::HWModuleOp root;
};

struct CorePlan final {
  fabric::SpatialCoreOccurrenceRef core;
  fabric::HardwareDomainRef clockDomain;
  fabric::HardwareDomainRef resetDomain;
  std::size_t definitionOrdinal = 0;
};

} // namespace

llvm::Expected<SystemRootCirctSkeleton> buildPortableSystemRootCirctSkeleton(
    mlir::MLIRContext &context,
    const FinalizedConfigurationABI &configurationAbi,
    const FabricOperationProviderRegistry &providers,
    const ExternalImplementationContractCatalog &externalContracts) {
  const fabric::FabricSystemRootView &system =
      configurationAbi.abi().fabricSystem();
  if (system.artifact().accCoreOccurrences().empty())
    return invalid("System has no SpatialCore occurrence");

  std::vector<Definition> definitions;
  std::vector<CorePlan> cores;
  cores.reserve(system.artifact().accCoreOccurrences().size());
  for (fabric::AccCoreOccurrenceRef accCore :
       system.artifact().accCoreOccurrences()) {
    const fabric::SpatialCoreOccurrenceRef spatialCore{accCore};
    auto clockDomain = findDomain(system, spatialCore,
                                  fabric::FabricHardwareDomainKind::Clock);
    if (!clockDomain)
      return clockDomain.takeError();
    auto resetDomain = findDomain(system, spatialCore,
                                  fabric::FabricHardwareDomainKind::Reset);
    if (!resetDomain)
      return resetDomain.takeError();

    auto skeleton =
        buildModuleRootCirctSkeleton(context, spatialCore, configurationAbi);
    if (!skeleton)
      return skeleton.takeError();
    std::vector<FabricOperationRecipeBinding> recipes;
    recipes.reserve(skeleton->operationLeaves.size());
    for (const FabricOperationLeafAssociation &association :
         skeleton->operationLeaves)
      recipes.push_back({association.occurrence,
                         BackendRecipeKey::PortableSystemVerilog,
                         {}});
    auto output = specializeFabricOperationLeaves(
        *skeleton->module, configurationAbi, skeleton->operationLeaves, recipes,
        providers, externalContracts);
    if (!output)
      return output.takeError();
    if (!output->payloads.empty() || !output->activityPoints.empty() ||
        !output->externalImplementationBindings.empty())
      return invalid("portable provider returned non-self-contained material");
    if (llvm::Error error = verifySpecializedCirctModule(*skeleton->module))
      return std::move(error);

    auto key = definitionKey(system, spatialCore, *clockDomain, *resetDomain,
                             *skeleton->module);
    if (!key)
      return key.takeError();
    auto existing =
        llvm::find_if(definitions, [&](const Definition &candidate) {
          return candidate.key == *key;
        });
    std::size_t definitionOrdinal = 0;
    if (existing == definitions.end()) {
      definitionOrdinal = definitions.size();
      definitions.push_back(
          Definition{std::move(*key), std::move(skeleton->module), {}});
    } else {
      definitionOrdinal =
          static_cast<std::size_t>(existing - definitions.begin());
    }
    cores.push_back(
        {spatialCore, *clockDomain, *resetDomain, definitionOrdinal});
  }

  mlir::OpBuilder builder(&context);
  const mlir::Location location = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> combined = mlir::ModuleOp::create(location);
  for (auto [ordinal, definition] : llvm::enumerate(definitions)) {
    auto root = renameDefinition(*definition.module, ordinal);
    if (!root)
      return root.takeError();
    definition.root = *root;
    while (!definition.module->getBody()->empty())
      definition.module->getBody()->front().moveBefore(
          combined->getBody(), combined->getBody()->end());
  }

  builder.setInsertionPointToEnd(combined->getBody());
  llvm::SmallVector<circt::hw::PortInfo, 32> inputs;
  llvm::SmallVector<circt::hw::PortInfo, 32> outputs;
  std::set<std::string> portNames;
  for (fabric::HardwareDomainRef domain : system.hardwareDomains()) {
    const auto *contract = system.hardwareDomainContract(domain);
    if (!contract ||
        (contract->kind() != fabric::FabricHardwareDomainKind::Clock &&
         contract->kind() != fabric::FabricHardwareDomainKind::Reset))
      continue;
    const bool used = llvm::any_of(cores, [&](const CorePlan &core) {
      return core.clockDomain == domain || core.resetDomain == domain;
    });
    if (!used)
      continue;
    const std::string name = domainPortName(domain, contract->kind());
    if (!portNames.insert(name).second)
      return invalid("System domain port name is duplicated");
    const mlir::Type type =
        contract->kind() == fabric::FabricHardwareDomainKind::Clock
            ? mlir::Type(circt::seq::ClockType::get(&context))
            : mlir::Type(builder.getI1Type());
    inputs.push_back(
        circt::hw::PortInfo{{builder.getStringAttr(name), type,
                             circt::hw::ModulePort::Direction::Input}});
  }
  for (const CorePlan &core : cores) {
    circt::hw::HWModuleOp definition = definitions[core.definitionOrdinal].root;
    for (const circt::hw::PortInfo &port : definition.getPortList()) {
      if (port.getName() == "clock" || port.getName() == "reset")
        continue;
      const std::string name = exposedPortName(core.core, port.getName());
      if (!portNames.insert(name).second)
        return invalid("System occurrence port name is duplicated");
      (port.isOutput() ? outputs : inputs)
          .push_back(renamedPort(builder, port, name));
    }
  }

  std::optional<std::string> materializationError;
  circt::hw::HWModuleOp::create(
      builder, location, builder.getStringAttr("loom_system"),
      circt::hw::ModulePortInfo(inputs, outputs),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        for (const CorePlan &core : cores) {
          circt::hw::HWModuleOp definition =
              definitions[core.definitionOrdinal].root;
          std::map<std::string, mlir::Value> instanceInputs;
          instanceInputs.emplace(
              "clock",
              accessor.getInput(domainPortName(
                  core.clockDomain, fabric::FabricHardwareDomainKind::Clock)));
          instanceInputs.emplace(
              "reset",
              accessor.getInput(domainPortName(
                  core.resetDomain, fabric::FabricHardwareDomainKind::Reset)));
          for (const circt::hw::PortInfo &port : definition.getPortList())
            if (!port.isOutput() && port.getName() != "clock" &&
                port.getName() != "reset")
              instanceInputs.emplace(port.getName().str(),
                                     accessor.getInput(exposedPortName(
                                         core.core, port.getName())));
          auto instance = hierarchy::instantiateModule(
              bodyBuilder, location, definition, corePrefix(core.core),
              instanceInputs);
          if (!instance) {
            if (!materializationError)
              materializationError = llvm::toString(instance.takeError());
            else
              llvm::consumeError(instance.takeError());
            return;
          }
          for (const circt::hw::PortInfo &port : definition.getPortList())
            if (port.isOutput())
              accessor.setOutput(exposedPortName(core.core, port.getName()),
                                 instance->at(port.getName().str()));
        }
      });
  if (materializationError)
    return invalid(*materializationError);
  if (mlir::failed(mlir::verify(*combined)))
    return invalid("combined System hierarchy does not verify");
  if (llvm::Error error = verifySpecializedCirctModule(*combined))
    return std::move(error);

  std::vector<ImplementationInterface> interfaces;
  for (fabric::HardwareDomainRef domain : system.hardwareDomains()) {
    const auto *contract = system.hardwareDomainContract(domain);
    if (!contract)
      continue;
    const bool used = llvm::any_of(cores, [&](const CorePlan &core) {
      return core.clockDomain == domain || core.resetDomain == domain;
    });
    if (!used)
      continue;
    if (contract->kind() == fabric::FabricHardwareDomainKind::Clock)
      interfaces.push_back(
          topPortInterface(ImplementationClockInterfaceRef{domain},
                           domainPortName(domain, contract->kind())));
    else if (contract->kind() == fabric::FabricHardwareDomainKind::Reset)
      interfaces.push_back(
          topPortInterface(ImplementationResetInterfaceRef{domain},
                           domainPortName(domain, contract->kind())));
  }
  std::set<ProgrammingUnitId> assignedUnits;
  for (const CorePlan &core : cores) {
    auto layout =
        derivePortableConfigurationTransportLayout(configurationAbi, core.core);
    if (!layout)
      return layout.takeError();
    for (const ConfigurationTransportUnitLayout &unit : layout->units) {
      if (!assignedUnits.insert(unit.programmingUnit.unitId).second)
        return invalid("Programming Unit belongs to multiple SpatialCores");
      interfaces.push_back(topPortInterface(
          ImplementationConfigurationInterfaceRef{unit.programmingUnit},
          exposedPortName(core.core, "cfg_awaddr")));
    }
  }
  if (assignedUnits.size() != configurationAbi.abi().programmingUnits().size())
    return invalid("System hierarchy does not expose every Programming Unit");

  for (const auto &attachment : system.spatialAttachments()) {
    auto core = attachmentCore(attachment.spatialEndpoint);
    if (!core)
      return core.takeError();
    const auto target = system.spatialCoreTarget(core->core);
    if (!target ||
        target->dependencyOrdinal !=
            attachment.moduleEndpoint.dependencyOrdinal ||
        target->target != attachment.moduleEndpoint.target.module)
      return invalid("System attachment targets a foreign Module occurrence");
    const std::string port =
        exposedPortName(*core, attachmentLocalPort(attachment));
    if (attachment.spatialEndpoint.transport())
      interfaces.push_back(topPortInterface(
          ImplementationDataInterfaceRef{attachment.spatialEndpoint}, port));
    else
      interfaces.push_back(topPortInterface(
          ImplementationMemoryInterfaceRef{attachment.spatialEndpoint}, port));
  }

  return SystemRootCirctSkeleton{std::move(combined), std::move(interfaces),
                                 definitions.size(), cores.size()};
}

} // namespace loom::hardware::rtl
