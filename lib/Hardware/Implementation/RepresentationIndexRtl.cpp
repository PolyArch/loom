#include "RepresentationIndexInternal.h"

#include "slang/ast/Symbol.h"
#include "slang/ast/symbols/BlockSymbols.h"
#include "slang/ast/symbols/CompilationUnitSymbols.h"
#include "slang/ast/symbols/InstanceSymbols.h"
#include "slang/ast/symbols/PortSymbols.h"
#include "slang/ast/symbols/VariableSymbols.h"
#include "slang/ast/types/Type.h"
#include "slang/syntax/AllSyntax.h"

#include "llvm/ADT/StringSet.h"

#include <string>
#include <utility>

namespace loom::hardware::detail {
namespace {

class RtlIndexBuilder final {
public:
  explicit RtlIndexBuilder(RepresentationFormatDescriptorRef formatRef)
      : catalog_(formatRef) {}

  llvm::Expected<RawIndex> build(const slang::ast::InstanceSymbol &top,
                                 const RepresentationLocator &exactRoot) {
    if (llvm::Error error = catalog_.addEntry(
            exactRoot, RepresentationObjectFacts{
                           RepresentationObjectKind::Module, std::nullopt}))
      return std::move(error);
    if (llvm::Error error =
            collectInstanceBody(top.body, exactRoot.canonicalName))
      return std::move(error);
    return catalog_.finish();
  }

private:
  llvm::Error collectPorts(const slang::ast::InstanceBodySymbol &body,
                           llvm::StringRef path,
                           llvm::StringSet<> &terminalPaths) {
    for (const slang::ast::Symbol *portMember : body.getPortList()) {
      if (portMember->kind == slang::ast::SymbolKind::MultiPort)
        return unsupportedIndex(
            "ports backed by multiple expressions are outside the descriptor");
      if (portMember->kind == slang::ast::SymbolKind::InterfacePort)
        return unsupportedIndex("interface ports are outside the descriptor");
      const auto *port = portMember->as_if<slang::ast::PortSymbol>();
      if (!port)
        return unsupportedIndex("module port kind is outside the descriptor");
      if (port->name.empty() || port->isNullPort)
        return unsupportedIndex(
            "unnamed or null ports are outside the descriptor");
      auto direction = signalDirection(port->direction, "port");
      if (!direction)
        return direction.takeError();
      auto width = packedIntegralWidth(port->getType(), "port");
      if (!width)
        return width.takeError();

      const std::string terminalPath = childPath(path, port->name);
      terminalPaths.insert(terminalPath);
      if (llvm::Error error = catalog_.addEntry(
              {RepresentationObjectKind::Port, terminalPath},
              RepresentationObjectFacts{
                  RepresentationObjectKind::Port,
                  RepresentationSignalGeometry{*direction, *width}}))
        return error;
    }
    return llvm::Error::success();
  }

  llvm::Error collectNet(const slang::ast::NetSymbol &net,
                         llvm::StringRef path) {
    if (net.name.empty() || net.isImplicit)
      return unsupportedIndex(
          "implicit or unnamed nets are outside the descriptor");
    auto width = packedIntegralWidth(net.getType(), "net");
    if (!width)
      return width.takeError();
    return catalog_.addEntry(
        {RepresentationObjectKind::Net, childPath(path, net.name)},
        RepresentationObjectFacts{RepresentationObjectKind::Net, std::nullopt});
  }

  llvm::Error collectVariable(const slang::ast::VariableSymbol &variable,
                              llvm::StringRef path) {
    if (variable.flags.has(slang::ast::VariableFlags::CompilerGenerated))
      return llvm::Error::success();
    if (variable.name.empty())
      return unsupportedIndex("unnamed variables are outside the descriptor");
    const slang::ast::Type *element = &variable.getType();
    bool hasUnpackedDimension = false;
    while (element->getCanonicalType().kind ==
           slang::ast::SymbolKind::FixedSizeUnpackedArrayType) {
      hasUnpackedDimension = true;
      if (!element->isFixedSize())
        return unsupportedIndex("variable unpacked dimension is not fixed");
      element = element->getArrayElementType();
      if (!element)
        return unsupportedIndex("variable has no fixed unpacked element type");
    }
    if (element->isUnpackedArray() || !element->isFixedSize() ||
        !element->isIntegral() || element->getBitWidth() == 0)
      return unsupportedIndex(
          "variable does not have a fixed packed-integral element type");
    const RepresentationObjectKind kind =
        hasUnpackedDimension ? RepresentationObjectKind::Memory
                             : RepresentationObjectKind::Register;
    return catalog_.addEntry({kind, childPath(path, variable.name)},
                             RepresentationObjectFacts{kind, std::nullopt});
  }

  llvm::Error
  collectResolvedInstance(const slang::ast::InstanceSymbol &instance,
                          llvm::StringRef path) {
    if (instance.name.empty() || !instance.arrayPath.empty())
      return unsupportedIndex(
          "unnamed or arrayed instances are outside the descriptor");
    if (instance.getDefinition().definitionKind !=
        slang::ast::DefinitionKind::Module)
      return unsupportedIndex(
          "interface and program instances are outside the descriptor");
    const std::string instancePath = childPath(path, instance.name);
    if (llvm::Error error = catalog_.addEntry(
            {RepresentationObjectKind::Instance, instancePath},
            RepresentationObjectFacts{RepresentationObjectKind::Instance,
                                      std::nullopt}))
      return error;
    if (atRootModule_)
      catalog_.addRootModuleInstance(
          {RepresentationObjectKind::Instance, instancePath},
          instance.getDefinition().name);
    const slang::ast::InstanceBodySymbol &body =
        instance.getCanonicalBody() ? *instance.getCanonicalBody()
                                    : instance.body;
    const bool wasRoot = atRootModule_;
    atRootModule_ = false;
    llvm::Error result = collectInstanceBody(body, instancePath);
    atRootModule_ = wasRoot;
    return result;
  }

  llvm::Error
  collectUnknownInstance(const slang::ast::UninstantiatedDefSymbol &instance,
                         llvm::StringRef path) {
    if (instance.name.empty() || instance.isChecker())
      return unsupportedIndex(
          "unnamed or checker occurrences are outside the descriptor");
    const slang::syntax::SyntaxNode *origin = instance.getSyntax();
    const auto *syntax =
        origin ? origin->as_if<slang::syntax::HierarchicalInstanceSyntax>()
               : nullptr;
    if (!syntax || !syntax->decl || !syntax->decl->dimensions.empty())
      return unsupportedIndex(
          "arrayed unknown instances are outside the descriptor");
    if (llvm::Error error = catalog_.addEntry(
            {RepresentationObjectKind::Instance,
             childPath(path, instance.name)},
            RepresentationObjectFacts{RepresentationObjectKind::Instance,
                                      std::nullopt}))
      return error;
    if (atRootModule_)
      catalog_.addRootModuleInstance(
          {RepresentationObjectKind::Instance, childPath(path, instance.name)},
          instance.definitionName);
    return catalog_.addUnresolvedModule(instance.definitionName);
  }

  llvm::Error collectGenerateBlock(const slang::ast::GenerateBlockSymbol &block,
                                   llvm::StringRef path) {
    if (block.isUninstantiated)
      return llvm::Error::success();
    if (block.isUnnamed || block.name.empty() || block.getArrayIndex())
      return unsupportedIndex(
          "implicit or arrayed generate scopes are outside the descriptor");
    return collectScope(block, childPath(path, block.name));
  }

  llvm::Error collectScope(const slang::ast::Scope &scope,
                           llvm::StringRef path) {
    for (const slang::ast::Symbol &member : scope.members())
      if (llvm::Error error = collectMember(member, path))
        return error;
    return llvm::Error::success();
  }

  llvm::Error collectInstanceBody(const slang::ast::InstanceBodySymbol &body,
                                  llvm::StringRef path) {
    llvm::StringSet<> terminalPaths;
    if (llvm::Error error = collectPorts(body, path, terminalPaths))
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
      return llvm::Error::success();
    case slang::ast::SymbolKind::Net:
      return collectNet(member.as<slang::ast::NetSymbol>(), path);
    case slang::ast::SymbolKind::Variable:
      return collectVariable(member.as<slang::ast::VariableSymbol>(), path);
    case slang::ast::SymbolKind::Instance:
      return collectResolvedInstance(member.as<slang::ast::InstanceSymbol>(),
                                     path);
    case slang::ast::SymbolKind::UninstantiatedDef:
      return collectUnknownInstance(
          member.as<slang::ast::UninstantiatedDefSymbol>(), path);
    case slang::ast::SymbolKind::GenerateBlock:
      return collectGenerateBlock(member.as<slang::ast::GenerateBlockSymbol>(),
                                  path);
    case slang::ast::SymbolKind::InstanceArray:
    case slang::ast::SymbolKind::GenerateBlockArray:
    case slang::ast::SymbolKind::PrimitiveInstance:
      return unsupportedIndex(
          "arrayed or primitive hierarchy is outside the descriptor");
    case slang::ast::SymbolKind::GenericClassDef:
    case slang::ast::SymbolKind::ClassType:
    case slang::ast::SymbolKind::Checker:
    case slang::ast::SymbolKind::CheckerInstance:
    case slang::ast::SymbolKind::CheckerInstanceBody:
    case slang::ast::SymbolKind::AnonymousProgram:
      return unsupportedIndex(
          "class, checker, or program constructs are outside the descriptor");
    default:
      return llvm::Error::success();
    }
  }

  RawIndexBuilder catalog_;
  bool atRootModule_ = true;
};

} // namespace

llvm::Expected<RawIndex>
indexSystemVerilogRtl(RepresentationFormatDescriptorRef formatRef,
                      const slang::ast::InstanceSymbol &top,
                      const RepresentationLocator &exactRoot) {
  return RtlIndexBuilder(formatRef).build(top, exactRoot);
}

} // namespace loom::hardware::detail
