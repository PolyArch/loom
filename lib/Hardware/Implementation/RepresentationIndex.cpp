#include "Hardware/Implementation/RepresentationIndex.h"

#include "RepresentationIndexInternal.h"

#include "slang/ast/Compilation.h"
#include "slang/ast/Symbol.h"
#include "slang/ast/symbols/CompilationUnitSymbols.h"
#include "slang/ast/symbols/InstanceSymbols.h"
#include "slang/ast/symbols/MemberSymbols.h"
#include "slang/ast/types/Type.h"
#include "slang/diagnostics/Diagnostics.h"
#include "slang/diagnostics/PreprocessorDiags.h"
#include "slang/parsing/Lexer.h"
#include "slang/parsing/Parser.h"
#include "slang/parsing/Preprocessor.h"
#include "slang/syntax/AllSyntax.h"
#include "slang/syntax/SyntaxKind.h"
#include "slang/syntax/SyntaxTree.h"
#include "slang/text/SourceManager.h"
#include "slang/util/Bag.h"
#include "slang/util/BumpAllocator.h"
#include "slang/util/LanguageVersion.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace boost {
BOOST_NORETURN void throw_exception(const std::exception &) { std::abort(); }
} // namespace boost

namespace loom::hardware {
namespace detail {

llvm::Error invalidIndex(const llvm::Twine &reason) {
  return llvm::make_error<RepresentationIndexFailure>(
      RepresentationIndexFailureKind::Invalid, reason.str());
}

llvm::Error unsupportedIndex(const llvm::Twine &reason) {
  return llvm::make_error<RepresentationIndexFailure>(
      RepresentationIndexFailureKind::Unsupported, reason.str());
}

std::string childPath(llvm::StringRef parent, std::string_view child) {
  return (parent + "." + llvm::StringRef(child)).str();
}

llvm::Error RawIndexBuilder::addEntry(RepresentationLocator locator,
                                      RepresentationObjectFacts facts) {
  if (llvm::Error error =
          validateRepresentationLocatorSyntax(formatRef_, locator))
    return unsupportedIndex(
        "HDL object has a locator outside the descriptor: " +
        llvm::toString(std::move(error)));
  raw_.entries.push_back(RawIndexEntry{std::move(locator), std::move(facts)});
  return llvm::Error::success();
}

llvm::Error
RawIndexBuilder::addUnresolvedModule(std::string_view definitionName) {
  RepresentationLocator locator{RepresentationObjectKind::Module,
                                std::string(definitionName)};
  if (llvm::is_contained(raw_.unresolved, locator))
    return llvm::Error::success();
  if (llvm::Error error =
          validateRepresentationLocatorSyntax(formatRef_, locator))
    return unsupportedIndex(
        "unresolved definition name is outside the descriptor: " +
        llvm::toString(std::move(error)));
  raw_.unresolved.push_back(locator);
  return addEntry(std::move(locator),
                  RepresentationObjectFacts{RepresentationObjectKind::Module,
                                            std::nullopt});
}

llvm::Expected<RawIndex> RawIndexBuilder::finish() {
  llvm::sort(
      raw_.entries, [](const RawIndexEntry &lhs, const RawIndexEntry &rhs) {
        return representationLocatorCanonicalLess(lhs.locator, rhs.locator);
      });
  for (std::size_t index = 1; index < raw_.entries.size(); ++index) {
    if (raw_.entries[index - 1].locator == raw_.entries[index].locator)
      return invalidIndex("two HDL objects have the same canonical locator");
  }
  llvm::sort(raw_.unresolved, representationLocatorCanonicalLess);
  return std::move(raw_);
}

llvm::Expected<RepresentationSignalDirection>
signalDirection(slang::ast::ArgumentDirection direction,
                llvm::StringRef description) {
  switch (direction) {
  case slang::ast::ArgumentDirection::In:
    return RepresentationSignalDirection::Input;
  case slang::ast::ArgumentDirection::Out:
    return RepresentationSignalDirection::Output;
  case slang::ast::ArgumentDirection::InOut:
    return RepresentationSignalDirection::Inout;
  case slang::ast::ArgumentDirection::Ref:
    return unsupportedIndex(
        description + " has a reference direction outside the descriptor");
  }
  return unsupportedIndex(description +
                          " has a direction outside the descriptor");
}

llvm::Expected<std::uint64_t> packedIntegralWidth(const slang::ast::Type &type,
                                                  llvm::StringRef description) {
  if (type.isUnpackedArray() || !type.isFixedSize() || !type.isIntegral() ||
      type.getBitWidth() == 0)
    return unsupportedIndex(description + " does not have one fixed positive "
                                          "packed-integral width");
  return static_cast<std::uint64_t>(type.getBitWidth());
}

} // namespace detail
namespace {

using detail::BuiltinRepresentationIndexer;
using detail::RawIndex;
using slang::LanguageVersion;

struct LoadedSource final {
  std::string logicalName;
  std::string bytes;
};

LanguageVersion languageVersion(RepresentationLanguageProfile profile) {
  switch (profile) {
  case RepresentationLanguageProfile::Ieee1800_2017:
    return LanguageVersion::v1800_2017;
  case RepresentationLanguageProfile::Ieee1364_2005:
    return LanguageVersion::v1364_2005;
  }
  llvm_unreachable("closed representation language profile");
}

slang::parsing::LexerOptions lexerOptions(LanguageVersion version) {
  slang::parsing::LexerOptions options;
  options.commentHandlers.clear();
  options.maxErrors = 16;
  options.languageVersion = version;
  options.enableLegacyProtect = false;
  options.allowMacroTrailingSpace = false;
  return options;
}

slang::parsing::PreprocessorOptions
preprocessorOptions(LanguageVersion version) {
  slang::parsing::PreprocessorOptions options;
  // isLanguageError relies on this zero depth: the include-depth report can
  // then only come from the descriptor's fixed no-follow policy.
  options.maxIncludeDepth = 0;
  options.languageVersion = version;
  options.predefineSource = "<api>";
  options.predefines.clear();
  options.undefines.clear();
  options.additionalIncludePaths.clear();
  options.ignoreDirectives.clear();
  options.keywordMapping.clear();
  options.bufferChangeCB = {};
  options.allowMissingProtectedScopeEnd = false;
  return options;
}

slang::parsing::ParserOptions parserOptions(LanguageVersion version) {
  slang::parsing::ParserOptions options;
  options.maxRecursionDepth = 1024;
  options.languageVersion = version;
  return options;
}

slang::ast::CompilationOptions baseCompilationOptions(LanguageVersion version) {
  slang::ast::CompilationOptions options;
  options.flags = slang::ast::CompilationFlags::IgnoreUnknownModules |
                  slang::ast::CompilationFlags::DisallowRefsToUnknownInstances;
  options.maxInstanceDepth = 128;
  options.maxCheckerInstanceDepth = 64;
  options.maxGenerateSteps = 131072;
  options.maxConstexprDepth = 128;
  options.maxConstexprSteps = 1000000;
  options.maxConstexprBacktrace = 10;
  options.maxConstantSize = 8 * 1024 * 1024;
  options.maxDefParamSteps = 128;
  options.maxDefParamBlocks = std::numeric_limits<std::uint32_t>::max();
  options.maxInstanceArray = 65535;
  options.maxEnumValues = 65535;
  options.maxRecursiveClassSpecialization = 8;
  options.maxUDPCoverageNotes = 8;
  options.errorLimit = 64;
  options.typoCorrectionLimit = 32;
  options.minTypMax = slang::ast::MinTypMax::Typ;
  options.languageVersion = version;
  options.defaultTimeScale.reset();
  options.topModules.clear();
  options.paramOverrides.clear();
  options.defaultLiblist.clear();
  return options;
}

// Language validity is evaluated in a context that applies no admission
// policy: no preselected top, and top-level interface or reference ports stay
// legal. Every definition in the closure participates, so an intrinsic
// frontend error is observed wherever it occurs.
slang::ast::CompilationOptions
languageValidationOptions(LanguageVersion version) {
  slang::ast::CompilationOptions options = baseCompilationOptions(version);
  options.flags |= slang::ast::CompilationFlags::AllowTopLevelIfacePorts;
  return options;
}

// Admission elaboration forces the exact root under the descriptor's fixed
// configuration. A failure at this stage is an admission result, never a
// language error.
slang::ast::CompilationOptions
admissionElaborationOptions(LanguageVersion version, llvm::StringRef exactTop) {
  slang::ast::CompilationOptions options = baseCompilationOptions(version);
  options.topModules.emplace(exactTop);
  return options;
}

llvm::Error validateTextPayload(const ImplementationPayload &payload,
                                llvm::ArrayRef<std::uint8_t> bytes) {
  const llvm::StringRef text(reinterpret_cast<const char *>(bytes.data()),
                             bytes.size());
  if (text.contains('\0'))
    return detail::invalidIndex("text payload '" +
                                payload.canonicalLogicalName +
                                "' contains a NUL byte");
  if (text.contains('\r'))
    return detail::invalidIndex("text payload '" +
                                payload.canonicalLogicalName +
                                "' does not use LF line endings");
  if (!llvm::json::isUTF8(text))
    return detail::invalidIndex("text payload '" +
                                payload.canonicalLogicalName +
                                "' is not valid UTF-8");
  return llvm::Error::success();
}

llvm::Expected<std::vector<LoadedSource>>
validateAndLoadClosure(const RepresentationFormatDescriptor &descriptor,
                       llvm::ArrayRef<ImplementationPayload> canonicalPayloads,
                       const BlobStore &blobs) {
  auto canonical = canonicalizeImplementationPayloadCatalog(canonicalPayloads);
  if (!canonical)
    return detail::invalidIndex("payload closure is invalid: " +
                                llvm::toString(canonical.takeError()));
  if (!llvm::equal(*canonical, canonicalPayloads))
    return detail::invalidIndex("payload closure is not in canonical order");

  std::vector<std::uint64_t> counts(descriptor.payloadContracts.size());
  std::vector<LoadedSource> sources;
  for (const ImplementationPayload &payload : canonicalPayloads) {
    const auto contract =
        llvm::find_if(descriptor.payloadContracts,
                      [&](const RepresentationPayloadContract &candidate) {
                        return candidate.role == payload.role;
                      });
    if (contract == descriptor.payloadContracts.end())
      return detail::invalidIndex(
          "payload role is not admitted by the selected format");
    const std::size_t contractIndex = static_cast<std::size_t>(
        contract - descriptor.payloadContracts.begin());
    ++counts[contractIndex];

    auto contents = blobs.get(payload.blobDigest);
    if (!contents)
      return detail::invalidIndex(
          "payload '" + payload.canonicalLogicalName +
          "' could not be loaded: " + llvm::toString(contents.takeError()));
    if (contract->textPolicy == RepresentationTextPolicy::Utf8LfNoNul)
      if (llvm::Error error = validateTextPayload(payload, *contents))
        return std::move(error);
    if (descriptor.frontendSourceRole &&
        payload.role == *descriptor.frontendSourceRole) {
      sources.push_back(LoadedSource{
          payload.canonicalLogicalName,
          std::string(reinterpret_cast<const char *>(contents->data()),
                      contents->size())});
    }
  }

  for (auto [contract, count] :
       llvm::zip_equal(descriptor.payloadContracts, counts)) {
    if (count < contract.minimumCount)
      return detail::invalidIndex(
          "payload role cardinality is below its minimum");
    if (contract.maximumCount && count > *contract.maximumCount)
      return detail::invalidIndex(
          "payload role cardinality is above its maximum");
  }
  return sources;
}

bool isForbiddenRawDirective(slang::syntax::SyntaxKind kind) {
  using slang::syntax::SyntaxKind;
  switch (kind) {
  case SyntaxKind::IncludeDirective:
  case SyntaxKind::BeginKeywordsDirective:
  case SyntaxKind::EndKeywordsDirective:
  case SyntaxKind::PragmaDirective:
  case SyntaxKind::ProtectDirective:
  case SyntaxKind::ProtectedDirective:
  case SyntaxKind::EndProtectDirective:
  case SyntaxKind::EndProtectedDirective:
    return true;
  default:
    return false;
  }
}

/// Descriptor-admission conditions observed while raw-lexing one source unit.
/// They are recorded rather than returned so that language validity of the
/// complete payload closure is always established first.
struct RawAdmissionMarkers final {
  bool forbiddenDirective = false;
  bool escapedIdentifier = false;
};

/// The descriptor never follows an include directive, so the frontend's exact
/// typed report that a well-formed include could not be followed under that
/// fixed policy is an admission fact. Every other error belongs to the
/// selected language profile.
bool isLanguageError(const slang::Diagnostic &diagnostic) {
  return diagnostic.isError() &&
         diagnostic.code != slang::diag::ExceededMaxIncludeDepth;
}

llvm::Error validateRawSource(const slang::SourceBuffer &buffer,
                              slang::SourceManager &sourceManager,
                              LanguageVersion version,
                              RawAdmissionMarkers &markers) {
  slang::BumpAllocator allocator;
  slang::Diagnostics diagnostics;
  slang::parsing::Lexer lexer(buffer, allocator, diagnostics, sourceManager,
                              lexerOptions(version));
  while (true) {
    const slang::parsing::Token token = lexer.lex();
    if (token.kind == slang::parsing::TokenKind::EndOfFile)
      break;
    if (token.kind == slang::parsing::TokenKind::Directive &&
        isForbiddenRawDirective(token.directiveKind()))
      markers.forbiddenDirective = true;
    if (token.kind == slang::parsing::TokenKind::Identifier &&
        llvm::StringRef(token.rawText()).starts_with("\\"))
      markers.escapedIdentifier = true;
  }
  if (llvm::any_of(diagnostics, [](const slang::Diagnostic &diagnostic) {
        return diagnostic.isError();
      }))
    return detail::invalidIndex("source lexing failed");
  return llvm::Error::success();
}

llvm::Error validateWiringExpression(const slang::syntax::SyntaxNode &node) {
  using slang::syntax::SyntaxKind;
  switch (node.kind) {
  case SyntaxKind::IdentifierName:
  case SyntaxKind::IdentifierSelectName:
  case SyntaxKind::ElementSelect:
  case SyntaxKind::ElementSelectExpression:
  case SyntaxKind::BitSelect:
  case SyntaxKind::SimpleRangeSelect:
  case SyntaxKind::AscendingRangeSelect:
  case SyntaxKind::DescendingRangeSelect:
  case SyntaxKind::IntegerLiteralExpression:
  case SyntaxKind::IntegerVectorExpression:
  case SyntaxKind::UnbasedUnsizedLiteralExpression:
  case SyntaxKind::ParenthesizedExpression:
  case SyntaxKind::ConcatenationExpression:
  case SyntaxKind::MultipleConcatenationExpression:
    break;
  default:
    return detail::unsupportedIndex(
        "gate wiring expression is outside the descriptor");
  }
  for (std::uint32_t index = 0; index < node.getChildCount(); ++index) {
    if (const slang::syntax::SyntaxNode *child = node.childNode(index))
      if (llvm::Error error = validateWiringExpression(*child))
        return error;
  }
  return llvm::Error::success();
}

llvm::Error validateGateActualExpression(
    const slang::syntax::PropertyExprSyntax &propertyExpression) {
  using slang::syntax::SyntaxKind;
  if (propertyExpression.kind != SyntaxKind::SimplePropertyExpr)
    return detail::invalidIndex(
        "gate connection is not an IEEE 1364-2005 expression");
  const auto &property =
      propertyExpression.as<slang::syntax::SimplePropertyExprSyntax>();
  if (property.expr->kind != SyntaxKind::SimpleSequenceExpr)
    return detail::invalidIndex(
        "gate connection is not an IEEE 1364-2005 expression");
  const auto &sequence =
      property.expr->as<slang::syntax::SimpleSequenceExprSyntax>();
  if (sequence.repetition)
    return detail::invalidIndex(
        "gate connection is not an IEEE 1364-2005 expression");
  return validateWiringExpression(*sequence.expr);
}

llvm::Error validateGateNetDeclaration(
    const slang::syntax::NetDeclarationSyntax &declaration) {
  if (declaration.strength || declaration.delay)
    return detail::unsupportedIndex(
        "gate net strength or delay is outside the descriptor");
  for (const slang::syntax::DeclaratorSyntax *declarator :
       declaration.declarators) {
    if (!declarator || declarator->name.valueText().empty() ||
        !declarator->dimensions.empty())
      return detail::unsupportedIndex(
          "unnamed or arrayed gate nets are outside the descriptor");
    if (declarator->initializer)
      if (llvm::Error error =
              validateWiringExpression(*declarator->initializer->expr))
        return error;
  }
  return llvm::Error::success();
}

llvm::Error validateGateContinuousAssign(
    const slang::syntax::ContinuousAssignSyntax &assignment) {
  if (assignment.strength || assignment.delay)
    return detail::unsupportedIndex("gate continuous-assignment strength or "
                                    "delay is outside the descriptor");
  for (const slang::syntax::ExpressionSyntax *expression :
       assignment.assignments) {
    if (!expression)
      return detail::unsupportedIndex(
          "empty gate continuous assignment is outside the descriptor");
    if (expression->kind != slang::syntax::SyntaxKind::AssignmentExpression)
      return detail::unsupportedIndex(
          "gate continuous assignment is outside the descriptor");
    const auto &assigned =
        expression->as<slang::syntax::BinaryExpressionSyntax>();
    if (llvm::Error error = validateWiringExpression(*assigned.left))
      return error;
    if (llvm::Error error = validateWiringExpression(*assigned.right))
      return error;
  }
  return llvm::Error::success();
}

bool isGateBehaviorSyntax(slang::syntax::SyntaxKind kind) {
  using slang::syntax::SyntaxKind;
  switch (kind) {
  case SyntaxKind::AlwaysBlock:
  case SyntaxKind::AlwaysCombBlock:
  case SyntaxKind::AlwaysFFBlock:
  case SyntaxKind::AlwaysLatchBlock:
  case SyntaxKind::InitialBlock:
  case SyntaxKind::FinalBlock:
  case SyntaxKind::DataDeclaration:
  case SyntaxKind::UserDefinedNetDeclaration:
  case SyntaxKind::FunctionDeclaration:
  case SyntaxKind::TaskDeclaration:
  case SyntaxKind::FunctionPrototype:
  case SyntaxKind::DPIImport:
  case SyntaxKind::DPIExport:
  case SyntaxKind::Delay3:
  case SyntaxKind::DelayControl:
  case SyntaxKind::EventControl:
  case SyntaxKind::EventControlWithExpression:
  case SyntaxKind::ImplicitEventControl:
  case SyntaxKind::RepeatedEventControl:
  case SyntaxKind::TimingControlExpression:
  case SyntaxKind::TimingControlStatement:
  case SyntaxKind::SpecifyBlock:
  case SyntaxKind::SystemTimingCheck:
    return true;
  default:
    return false;
  }
}

llvm::Error validateSyntaxNode(const slang::syntax::SyntaxNode &node,
                               BuiltinRepresentationIndexer indexer) {
  using slang::syntax::SyntaxKind;
  switch (node.kind) {
  case SyntaxKind::InterfaceDeclaration:
  case SyntaxKind::ProgramDeclaration:
  case SyntaxKind::AnonymousProgram:
  case SyntaxKind::CheckerDeclaration:
  case SyntaxKind::CheckerInstantiation:
  case SyntaxKind::CheckerInstanceStatement:
  case SyntaxKind::ClassDeclaration:
    return detail::unsupportedIndex("interface, program, checker, or class "
                                    "syntax is outside the descriptor");
  case SyntaxKind::LoopGenerate:
    return detail::unsupportedIndex(
        "arrayed generate hierarchy is outside the descriptor");
  case SyntaxKind::MinTypMaxExpression:
    return detail::unsupportedIndex(
        "min:typ:max expressions are outside the descriptor");
  case SyntaxKind::VariablePortHeader:
    if (node.as<slang::syntax::VariablePortHeaderSyntax>().direction.kind ==
        slang::parsing::TokenKind::RefKeyword)
      return detail::unsupportedIndex(
          "reference ports are outside the descriptor");
    break;
  case SyntaxKind::ExplicitAnsiPort:
    if (node.as<slang::syntax::ExplicitAnsiPortSyntax>().direction.kind ==
        slang::parsing::TokenKind::RefKeyword)
      return detail::unsupportedIndex(
          "reference ports are outside the descriptor");
    break;
  case SyntaxKind::HierarchicalInstance: {
    const auto &instance = node.as<slang::syntax::HierarchicalInstanceSyntax>();
    if (!instance.decl || instance.decl->name.valueText().empty() ||
        !instance.decl->dimensions.empty())
      return detail::unsupportedIndex(
          "unnamed or arrayed hierarchy is outside the descriptor");
    break;
  }
  case SyntaxKind::GenerateBlock: {
    const auto &block = node.as<slang::syntax::GenerateBlockSyntax>();
    if (!block.label && !block.beginName)
      return detail::unsupportedIndex(
          "implicit generate hierarchy is outside the descriptor");
    break;
  }
  case SyntaxKind::UdpDeclaration:
    if (indexer == BuiltinRepresentationIndexer::SystemVerilogRtl)
      return detail::unsupportedIndex(
          "UDP declarations are outside the RTL descriptor");
    break;
  default:
    break;
  }

  if (indexer == BuiltinRepresentationIndexer::StructuralVerilogGateNetlist) {
    if (isGateBehaviorSyntax(node.kind))
      return detail::unsupportedIndex(
          "behavioral or timed gate syntax is outside the descriptor");
    if (node.kind == SyntaxKind::NetDeclaration) {
      if (llvm::Error error = validateGateNetDeclaration(
              node.as<slang::syntax::NetDeclarationSyntax>()))
        return error;
    } else if (node.kind == SyntaxKind::ContinuousAssign) {
      if (llvm::Error error = validateGateContinuousAssign(
              node.as<slang::syntax::ContinuousAssignSyntax>()))
        return error;
    } else if (node.kind == SyntaxKind::OrderedPortConnection) {
      if (llvm::Error error = validateGateActualExpression(
              *node.as<slang::syntax::OrderedPortConnectionSyntax>().expr))
        return error;
    } else if (node.kind == SyntaxKind::NamedPortConnection) {
      const auto *expression =
          node.as<slang::syntax::NamedPortConnectionSyntax>().expr;
      if (expression)
        if (llvm::Error error = validateGateActualExpression(*expression))
          return error;
    } else if (node.kind == SyntaxKind::PrimitiveInstantiation) {
      const auto &primitive =
          node.as<slang::syntax::PrimitiveInstantiationSyntax>();
      if (primitive.type.kind != slang::parsing::TokenKind::Identifier ||
          primitive.strength || primitive.delay)
        return detail::unsupportedIndex(
            "built-in, switch, strengthened, or delayed primitives are outside "
            "the descriptor");
    }
  }

  for (std::uint32_t index = 0; index < node.getChildCount(); ++index) {
    if (const slang::syntax::SyntaxNode *child = node.childNode(index))
      if (llvm::Error error = validateSyntaxNode(*child, indexer))
        return error;
  }
  return llvm::Error::success();
}

llvm::Error
validateDefinitionClosure(const slang::ast::Compilation &compilation,
                          BuiltinRepresentationIndexer indexer) {
  for (const slang::ast::Symbol *definition : compilation.getDefinitions()) {
    if (const auto *hdl = definition->as_if<slang::ast::DefinitionSymbol>()) {
      if (hdl->definitionKind != slang::ast::DefinitionKind::Module)
        return detail::unsupportedIndex(
            "non-module HDL definitions are outside the descriptor");
      continue;
    }
    const auto *primitive = definition->as_if<slang::ast::PrimitiveSymbol>();
    if (!primitive)
      continue;
    if (indexer == BuiltinRepresentationIndexer::SystemVerilogRtl ||
        primitive->primitiveKind != slang::ast::PrimitiveSymbol::UserDefined ||
        primitive->name.empty())
      return detail::unsupportedIndex(
          "primitive definition is outside the descriptor");
  }
  return llvm::Error::success();
}

llvm::Expected<RawIndex>
indexInitialHdl(RepresentationFormatDescriptorRef formatRef,
                const RepresentationLocator &exactRoot,
                llvm::ArrayRef<ImplementationPayload> canonicalPayloads,
                const BlobStore &blobs) {
  const detail::StaticRepresentationFormatEntry &entry =
      detail::getStaticRepresentationFormatEntry(formatRef);
  const RepresentationFormatDescriptor &descriptor = entry.descriptor;
  if (exactRoot.kind != descriptor.exactRootKind)
    return detail::invalidIndex(
        "exact root has the wrong representation object kind");
  if (llvm::Error error =
          validateRepresentationLocatorSyntax(formatRef, exactRoot))
    return detail::invalidIndex("exact root locator is invalid: " +
                                llvm::toString(std::move(error)));
  if (!descriptor.frontendSourceRole || !descriptor.languageProfile)
    return detail::invalidIndex(
        "built-in HDL indexer descriptor lacks source or language metadata");

  auto sources = validateAndLoadClosure(descriptor, canonicalPayloads, blobs);
  if (!sources)
    return sources.takeError();
  const LanguageVersion version = languageVersion(*descriptor.languageProfile);
  slang::SourceManager sourceManager;
  sourceManager.setDisableProximatePaths(true);
  sourceManager.setDisableLocalIncludes(true);
  slang::Bag syntaxOptions(lexerOptions(version), preprocessorOptions(version),
                           parserOptions(version));
  std::vector<std::shared_ptr<slang::syntax::SyntaxTree>> trees;
  trees.reserve(sources->size());
  std::vector<slang::SourceBuffer> buffers;
  buffers.reserve(sources->size());
  for (const LoadedSource &source : *sources)
    buffers.push_back(
        sourceManager.assignText(source.logicalName, source.bytes));

  // LanguageValid, raw phase: malformed bytes are language errors, while
  // descriptor-policy observations are only recorded.
  RawAdmissionMarkers markers;
  for (const slang::SourceBuffer &buffer : buffers)
    if (llvm::Error error =
            validateRawSource(buffer, sourceManager, version, markers))
      return std::move(error);

  // LanguageValid, parse phase: every unit must parse under the selected
  // profile before any admission verdict is computed.
  for (const slang::SourceBuffer &buffer : buffers) {
    std::shared_ptr<slang::syntax::SyntaxTree> tree =
        slang::syntax::SyntaxTree::fromBuffer(buffer, sourceManager,
                                              syntaxOptions);
    if (!tree)
      return detail::invalidIndex("HDL source did not produce a syntax tree");
    if (llvm::any_of(tree->diagnostics(), isLanguageError))
      return detail::invalidIndex("HDL parse or elaboration failed");
    trees.push_back(std::move(tree));
  }

  // LanguageValid, semantic phase: elaborate the complete closure in the
  // language-validation context, which applies no admission policy.
  slang::ast::Compilation languageCompilation(
      slang::Bag(languageValidationOptions(version)));
  for (const auto &tree : trees)
    languageCompilation.addSyntaxTree(tree);
  const slang::Diagnostics &languageDiagnostics =
      languageCompilation.getAllDiagnostics();
  if (llvm::any_of(languageDiagnostics, isLanguageError))
    return detail::invalidIndex("HDL parse or elaboration failed");

  // The exact-root claim: absence or ambiguity is Invalid; the root kind is
  // an admission question.
  std::size_t exactDefinitionCount = 0;
  bool exactDefinitionIsModule = false;
  for (const slang::ast::Symbol *definition :
       languageCompilation.getDefinitions()) {
    if (definition->name != exactRoot.canonicalName)
      continue;
    ++exactDefinitionCount;
    const auto *module = definition->as_if<slang::ast::DefinitionSymbol>();
    exactDefinitionIsModule =
        module && module->definitionKind == slang::ast::DefinitionKind::Module;
  }
  if (exactDefinitionCount == 0)
    return detail::invalidIndex(
        "exact top does not resolve to a module definition");
  if (exactDefinitionCount > 1)
    return detail::invalidIndex(
        "exact top resolves to more than one definition");

  // DescriptorAdmitted, over the complete closure.
  if (!exactDefinitionIsModule)
    return detail::unsupportedIndex("exact top is not a module definition");
  if (markers.forbiddenDirective)
    return detail::unsupportedIndex(
        "source directive is outside the descriptor");
  if (markers.escapedIdentifier)
    return detail::unsupportedIndex(
        "escaped identifiers are outside the descriptor");
  for (const auto &tree : trees) {
    if (!tree->getIncludeDirectives().empty())
      return detail::unsupportedIndex(
          "include directives are outside the descriptor");
    if (llvm::Error error = validateSyntaxNode(tree->root(), entry.indexer))
      return std::move(error);
  }
  if (llvm::Error error =
          validateDefinitionClosure(languageCompilation, entry.indexer))
    return std::move(error);

  // Admission elaboration forces the exact root under the fixed descriptor
  // configuration. Language validity is already established, so any error
  // here is an admission result.
  slang::ast::Compilation admissionCompilation(
      slang::Bag(admissionElaborationOptions(version,
                                             exactRoot.canonicalName)));
  for (const auto &tree : trees)
    admissionCompilation.addSyntaxTree(tree);
  const slang::Diagnostics &admissionDiagnostics =
      admissionCompilation.getAllDiagnostics();
  if (llvm::any_of(admissionDiagnostics,
                   [](const slang::Diagnostic &diagnostic) {
                     return diagnostic.isError();
                   }))
    return detail::unsupportedIndex(
        "exact top does not elaborate under the fixed descriptor "
        "configuration");

  const slang::ast::RootSymbol &root = admissionCompilation.getRoot();
  const slang::ast::InstanceSymbol *top = nullptr;
  for (const slang::ast::InstanceSymbol *candidate : root.topInstances) {
    if (candidate->name != exactRoot.canonicalName)
      continue;
    if (top)
      return detail::invalidIndex(
          "exact top resolves to more than one definition");
    top = candidate;
  }
  if (!top)
    return detail::invalidIndex(
        "exact top does not resolve to a module definition");
  if (!top->isModule())
    return detail::unsupportedIndex("exact top is not a module definition");

  switch (entry.indexer) {
  case BuiltinRepresentationIndexer::SystemVerilogRtl:
    return detail::indexSystemVerilogRtl(formatRef, *top, exactRoot);
  case BuiltinRepresentationIndexer::StructuralVerilogGateNetlist:
    return detail::indexStructuralVerilogGateNetlist(formatRef, *top,
                                                     exactRoot);
  }
  llvm_unreachable("closed built-in representation indexer");
}

} // namespace

char RepresentationIndexFailure::ID;

void RepresentationIndexFailure::log(llvm::raw_ostream &stream) const {
  stream << (kind_ == RepresentationIndexFailureKind::Invalid
                 ? "representation_index_invalid: "
                 : "representation_index_unsupported: ")
         << reason_;
}

std::error_code RepresentationIndexFailure::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::Expected<std::optional<RepresentationObjectFacts>>
RepresentationIndex::lookup(const RepresentationLocator &locator) const {
  if (llvm::Error error =
          validateRepresentationLocatorSyntax(formatRef_, locator))
    return detail::invalidIndex("lookup locator is invalid: " +
                                llvm::toString(std::move(error)));
  const llvm::StringRef name(locator.canonicalName);
  const llvm::StringRef root(exactRoot_.canonicalName);
  if (locator.kind != exactRoot_.kind &&
      !(name.starts_with(root) && name.size() > root.size() &&
        name[root.size()] == '.'))
    return detail::invalidIndex(
        "lookup locator is not rooted at the indexed exact root");
  const auto found = llvm::lower_bound(
      entries_, locator,
      [](const Entry &entry, const RepresentationLocator &key) {
        return representationLocatorCanonicalLess(entry.locator, key);
      });
  if (found == entries_.end() || found->locator != locator)
    return std::optional<RepresentationObjectFacts>();
  return std::optional<RepresentationObjectFacts>(found->facts);
}

llvm::Expected<RepresentationIndex>
indexRepresentation(RepresentationFormatDescriptorRef formatRef,
                    const RepresentationLocator &exactRoot,
                    llvm::ArrayRef<ImplementationPayload> canonicalPayloads,
                    const BlobStore &blobs) {
  auto raw = indexInitialHdl(formatRef, exactRoot, canonicalPayloads, blobs);
  if (!raw)
    return raw.takeError();
  std::vector<RepresentationIndex::Entry> entries;
  entries.reserve(raw->entries.size());
  for (detail::RawIndexEntry &entry : raw->entries)
    entries.push_back(RepresentationIndex::Entry{std::move(entry.locator),
                                                 std::move(entry.facts)});
  return RepresentationIndex(formatRef, exactRoot, std::move(entries),
                             std::move(raw->unresolved));
}

} // namespace loom::hardware
