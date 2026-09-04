#include "RepresentationIndexInternal.h"

#include "slang/ast/Symbol.h"
#include "slang/ast/symbols/BlockSymbols.h"
#include "slang/ast/symbols/CompilationUnitSymbols.h"
#include "slang/ast/symbols/InstanceSymbols.h"
#include "slang/ast/symbols/MemberSymbols.h"
#include "slang/ast/symbols/PortSymbols.h"
#include "slang/ast/symbols/VariableSymbols.h"
#include "slang/ast/types/Type.h"
#include "slang/syntax/AllSyntax.h"

#include "llvm/ADT/StringSet.h"

#include <string>
#include <utility>

namespace loom::hardware::detail {
namespace {

llvm::Expected<RepresentationSignalDirection>
primitiveDirection(slang::ast::PrimitivePortDirection direction) {
  switch (direction) {
  case slang::ast::PrimitivePortDirection::In:
    return RepresentationSignalDirection::Input;
  case slang::ast::PrimitivePortDirection::Out:
  case slang::ast::PrimitivePortDirection::OutReg:
    return RepresentationSignalDirection::Output;
  case slang::ast::PrimitivePortDirection::InOut:
    return RepresentationSignalDirection::Inout;
  }
  return unsupportedIndex("UDP pin direction is outside the descriptor");
}

class GateIndexBuilder final {
public:
  explicit GateIndexBuilder(RepresentationFormatDescriptorRef formatRef)
      : catalog_(formatRef) {}

  llvm::Expected<RawIndex> build(const slang::ast::InstanceSymbol &top,
                                 const RepresentationLocator &exactRoot) {
    if (llvm::Error error = catalog_.addEntry(
            exactRoot, RepresentationObjectFacts{
                           RepresentationObjectKind::Module, std::nullopt}))
      return std::move(error);
    if (llvm::Error error = collectModuleBody(top.body, exactRoot.canonicalName,
                                              RepresentationObjectKind::Port))
      return std::move(error);
    return catalog_.finish();
  }

private:
  llvm::Error collectModuleTerminals(const slang::ast::InstanceBodySymbol &body,
                                     llvm::StringRef path,
                                     RepresentationObjectKind terminalKind,
                                     llvm::StringSet<> &terminalPaths) {
    for (const slang::ast::Symbol *portMember : body.getPortList()) {
      if (portMember->kind == slang::ast::SymbolKind::InterfacePort)
        return unsupportedIndex(
            "gate interface terminals are outside the descriptor");
      const slang::ast::Type *type = nullptr;
      slang::ast::ArgumentDirection portDirection;
      if (const auto *port = portMember->as_if<slang::ast::PortSymbol>()) {
        if (port->isNullPort)
          return unsupportedIndex("null gate terminals are outside the descriptor");
        type = &port->getType();
        portDirection = port->direction;
      } else if (const auto *aggregate =
                     portMember->as_if<slang::ast::MultiPortSymbol>()) {
        // A public aggregate has one fixed bit-stream interface only when all
        // of its ordered internal port expressions have the same direction.
        // Slang owns the concatenation order and resulting packed type.
        if (aggregate->ports.empty() ||
            llvm::any_of(aggregate->ports, [&](const auto *part) {
              return part->isNullPort || part->direction != aggregate->direction;
            }))
          return unsupportedIndex(
              "empty or mixed-direction aggregate gate terminals are outside "
              "the descriptor");
        type = &aggregate->getType();
        portDirection = aggregate->direction;
      } else {
        return unsupportedIndex(
            "gate module terminal kind is outside the descriptor");
      }
      if (portMember->name.empty())
        return unsupportedIndex(
            "unnamed or null gate terminals are outside the descriptor");
      auto direction = signalDirection(portDirection, "gate terminal");
      if (!direction)
        return direction.takeError();
      auto width = packedIntegralWidth(*type, "gate terminal");
      if (!width)
        return width.takeError();

      const std::string terminalPath = childPath(path, portMember->name);
      terminalPaths.insert(terminalPath);
      if (llvm::Error error = catalog_.addEntry(
              {terminalKind, terminalPath},
              RepresentationObjectFacts{
                  terminalKind,
                  RepresentationSignalGeometry{*direction, *width}}))
        return error;
    }
    return llvm::Error::success();
  }

  llvm::Error collectNet(const slang::ast::NetSymbol &net,
                         llvm::StringRef path) {
    if (net.name.empty() || net.isImplicit)
      return unsupportedIndex(
          "implicit or unnamed gate nets are outside the descriptor");
    if (net.getDelay() || net.getChargeStrength() ||
        net.getDriveStrength().first || net.getDriveStrength().second)
      return unsupportedIndex(
          "delayed or strengthened gate nets are outside the descriptor");
    auto width = packedIntegralWidth(net.getType(), "gate net");
    if (!width)
      return width.takeError();
    return catalog_.addEntry(
        {RepresentationObjectKind::Net, childPath(path, net.name)},
        RepresentationObjectFacts{RepresentationObjectKind::Net, std::nullopt});
  }

  llvm::Error collectResolvedCell(const slang::ast::InstanceSymbol &instance,
                                  llvm::StringRef path) {
    if (instance.name.empty() || !instance.arrayPath.empty())
      return unsupportedIndex(
          "unnamed or arrayed gate cells are outside the descriptor");
    if (instance.getDefinition().definitionKind !=
        slang::ast::DefinitionKind::Module)
      return unsupportedIndex(
          "non-module resolved gate cells are outside the descriptor");
    const std::string cellPath = childPath(path, instance.name);
    if (llvm::Error error = catalog_.addEntry(
            {RepresentationObjectKind::Cell, cellPath},
            RepresentationObjectFacts{RepresentationObjectKind::Cell,
                                      std::nullopt}))
      return error;
    if (atRootModule_)
      catalog_.addRootModuleInstance({RepresentationObjectKind::Cell, cellPath},
                                     instance.getDefinition().name);
    const slang::ast::InstanceBodySymbol &body =
        instance.getCanonicalBody() ? *instance.getCanonicalBody()
                                    : instance.body;
    const bool wasRoot = atRootModule_;
    atRootModule_ = false;
    llvm::Error result =
        collectModuleBody(body, cellPath, RepresentationObjectKind::Pin);
    atRootModule_ = wasRoot;
    return result;
  }

  llvm::Error
  collectUnknownCell(const slang::ast::UninstantiatedDefSymbol &instance,
                     llvm::StringRef path) {
    if (instance.name.empty() || instance.isChecker())
      return unsupportedIndex(
          "unnamed or checker gate cells are outside the descriptor");
    const slang::syntax::SyntaxNode *origin = instance.getSyntax();
    const auto *syntax =
        origin ? origin->as_if<slang::syntax::HierarchicalInstanceSyntax>()
               : nullptr;
    if (!syntax || !syntax->decl || !syntax->decl->dimensions.empty())
      return unsupportedIndex(
          "arrayed unknown gate cells are outside the descriptor");
    if (llvm::Error error = catalog_.addEntry(
            {RepresentationObjectKind::Cell, childPath(path, instance.name)},
            RepresentationObjectFacts{RepresentationObjectKind::Cell,
                                      std::nullopt}))
      return error;
    if (atRootModule_)
      catalog_.addRootModuleInstance(
          {RepresentationObjectKind::Cell, childPath(path, instance.name)},
          instance.definitionName);
    return catalog_.addUnresolvedModule(instance.definitionName);
  }

  llvm::Error
  collectUdpCell(const slang::ast::PrimitiveInstanceSymbol &instance,
                 llvm::StringRef path) {
    if (instance.name.empty() || !instance.arrayPath.empty())
      return unsupportedIndex(
          "unnamed or arrayed UDP cells are outside the descriptor");
    if (instance.primitiveType.primitiveKind !=
            slang::ast::PrimitiveSymbol::UserDefined ||
        instance.primitiveType.name.empty())
      return unsupportedIndex(
          "built-in gate or switch primitives are outside the descriptor");
    if (instance.getDelay() || instance.getDriveStrength().first ||
        instance.getDriveStrength().second)
      return unsupportedIndex(
          "delayed or strengthened UDP cells are outside the descriptor");

    const std::string cellPath = childPath(path, instance.name);
    if (llvm::Error error = catalog_.addEntry(
            {RepresentationObjectKind::Cell, cellPath},
            RepresentationObjectFacts{RepresentationObjectKind::Cell,
                                      std::nullopt}))
      return error;
    for (const slang::ast::PrimitivePortSymbol *port :
         instance.primitiveType.ports) {
      if (!port || port->name.empty())
        return unsupportedIndex(
            "unnamed UDP terminals are outside the descriptor");
      auto direction = primitiveDirection(port->direction);
      if (!direction)
        return direction.takeError();
      auto width = packedIntegralWidth(port->getType(), "UDP terminal");
      if (!width)
        return width.takeError();
      if (llvm::Error error = catalog_.addEntry(
              {RepresentationObjectKind::Pin, childPath(cellPath, port->name)},
              RepresentationObjectFacts{
                  RepresentationObjectKind::Pin,
                  RepresentationSignalGeometry{*direction, *width}}))
        return error;
    }
    return llvm::Error::success();
  }

  llvm::Error collectGenerateBlock(const slang::ast::GenerateBlockSymbol &block,
                                   llvm::StringRef path) {
    if (block.isUninstantiated)
      return llvm::Error::success();
    if (block.isUnnamed || block.name.empty() || block.getArrayIndex())
      return unsupportedIndex("implicit or arrayed gate generate scopes are "
                              "outside the descriptor");
    return collectScope(block, childPath(path, block.name));
  }

  llvm::Error collectScope(const slang::ast::Scope &scope,
                           llvm::StringRef path) {
    for (const slang::ast::Symbol &member : scope.members())
      if (llvm::Error error = collectMember(member, path))
        return error;
    return llvm::Error::success();
  }

  llvm::Error collectModuleBody(const slang::ast::InstanceBodySymbol &body,
                                llvm::StringRef path,
                                RepresentationObjectKind terminalKind) {
    llvm::StringSet<> terminalPaths;
    if (llvm::Error error =
            collectModuleTerminals(body, path, terminalKind, terminalPaths))
      return error;

    for (const slang::ast::Symbol &member : body.members()) {
      switch (member.kind) {
      case slang::ast::SymbolKind::Port:
      case slang::ast::SymbolKind::MultiPort:
      case slang::ast::SymbolKind::InterfacePort:
        continue;
      case slang::ast::SymbolKind::Net:
      case slang::ast::SymbolKind::Variable:
        if (!member.name.empty() &&
            terminalPaths.contains(childPath(path, member.name)))
          continue;
        break;
      default:
        break;
      }
      if (llvm::Error error = collectMember(member, path))
        return error;
    }
    return llvm::Error::success();
  }

  llvm::Error collectMember(const slang::ast::Symbol &member,
                            llvm::StringRef path) {
    switch (member.kind) {
    case slang::ast::SymbolKind::Port:
    case slang::ast::SymbolKind::MultiPort:
    case slang::ast::SymbolKind::InterfacePort:
    case slang::ast::SymbolKind::ContinuousAssign:
    case slang::ast::SymbolKind::Parameter:
    case slang::ast::SymbolKind::TypeParameter:
    case slang::ast::SymbolKind::Genvar:
      return llvm::Error::success();
    case slang::ast::SymbolKind::Net:
      return collectNet(member.as<slang::ast::NetSymbol>(), path);
    case slang::ast::SymbolKind::Variable:
      return unsupportedIndex(
          "runtime gate variables and memories are outside the descriptor");
    case slang::ast::SymbolKind::Instance:
      return collectResolvedCell(member.as<slang::ast::InstanceSymbol>(), path);
    case slang::ast::SymbolKind::UninstantiatedDef:
      return collectUnknownCell(
          member.as<slang::ast::UninstantiatedDefSymbol>(), path);
    case slang::ast::SymbolKind::PrimitiveInstance:
      return collectUdpCell(member.as<slang::ast::PrimitiveInstanceSymbol>(),
                            path);
    case slang::ast::SymbolKind::GenerateBlock:
      return collectGenerateBlock(member.as<slang::ast::GenerateBlockSymbol>(),
                                  path);
    case slang::ast::SymbolKind::InstanceArray:
    case slang::ast::SymbolKind::GenerateBlockArray:
      return unsupportedIndex(
          "arrayed gate hierarchy is outside the descriptor");
    case slang::ast::SymbolKind::ProceduralBlock:
    case slang::ast::SymbolKind::Subroutine:
    case slang::ast::SymbolKind::SpecifyBlock:
    case slang::ast::SymbolKind::SystemTimingCheck:
    case slang::ast::SymbolKind::Checker:
    case slang::ast::SymbolKind::CheckerInstance:
    case slang::ast::SymbolKind::CheckerInstanceBody:
      return unsupportedIndex(
          "behavioral or timed gate members are outside the descriptor");
    default:
      return llvm::Error::success();
    }
  }

  RawIndexBuilder catalog_;
  bool atRootModule_ = true;
};

} // namespace

llvm::Expected<RawIndex>
indexStructuralVerilogGateNetlist(RepresentationFormatDescriptorRef formatRef,
                                  const slang::ast::InstanceSymbol &top,
                                  const RepresentationLocator &exactRoot) {
  return GateIndexBuilder(formatRef).build(top, exactRoot);
}

} // namespace loom::hardware::detail
