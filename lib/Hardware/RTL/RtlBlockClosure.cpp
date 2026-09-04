#include "Hardware/RTL/RtlBlockClosure.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::hardware::rtl {
namespace {

constexpr llvm::StringLiteral identityDomain = "loom.rtl_block_identity.1";
constexpr llvm::StringLiteral blockNamePrefix = "loom_block_";
constexpr llvm::StringLiteral selfPlaceholder = "loom_block_self";

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "rtl_block_closure_invalid: " + message);
}

bool identifierStart(char character) {
  return std::isalpha(static_cast<unsigned char>(character)) ||
         character == '_';
}

bool identifierContinuation(char character) {
  return std::isalnum(static_cast<unsigned char>(character)) ||
         character == '_' || character == '$';
}

/// Visits every plain identifier token of Verilog text. Comments, string
/// literals, escaped identifiers, system identifiers, and the base letters of
/// sized literals are skipped; each visit receives the token and the text
/// preceding it since the previous visit.
template <typename Visit, typename VisitEscaped>
void forEachIdentifier(llvm::StringRef text, Visit &&visit,
                       VisitEscaped &&visitEscaped) {
  std::size_t index = 0;
  std::size_t flushed = 0;
  const auto skipWhile = [&](auto &&predicate) {
    while (index < text.size() && predicate(text[index]))
      ++index;
  };
  while (index < text.size()) {
    const char character = text[index];
    const char next = index + 1 < text.size() ? text[index + 1] : '\0';
    if (character == '/' && next == '/') {
      skipWhile([](char c) { return c != '\n'; });
    } else if (character == '/' && next == '*') {
      index += 2;
      while (index < text.size() &&
             !(text[index] == '*' && index + 1 < text.size() &&
               text[index + 1] == '/'))
        ++index;
      index = std::min(index + 2, text.size());
    } else if (character == '"') {
      ++index;
      while (index < text.size() && text[index] != '"') {
        if (text[index] == '\\')
          ++index;
        ++index;
      }
      index = std::min(index + 1, text.size());
    } else if (character == '\\') {
      const std::size_t begin = ++index;
      skipWhile(
          [](char c) { return !std::isspace(static_cast<unsigned char>(c)); });
      visitEscaped(text.slice(begin, index));
    } else if (character == '\'' || character == '$' ||
               std::isdigit(static_cast<unsigned char>(character))) {
      ++index;
      skipWhile(identifierContinuation);
    } else if (identifierStart(character)) {
      const std::size_t begin = index;
      skipWhile(identifierContinuation);
      visit(text.slice(flushed, begin), text.slice(begin, index));
      flushed = index;
    } else {
      ++index;
    }
  }
  visit(text.drop_front(flushed), llvm::StringRef());
}

struct RewrittenEmission final {
  std::string text;
  /// Identifier occurrences per referenced projection ordinal.
  std::map<std::size_t, std::uint64_t> references;
  bool reservedNameCollision = false;
};

/// Rewrites every identifier naming a projection definition through
/// replacement while counting the references per definition.
RewrittenEmission
rewriteEmission(llvm::StringRef bytes,
                const llvm::StringMap<std::size_t> &definitionOrdinals,
                const std::map<std::size_t, std::string> &replacement) {
  RewrittenEmission result;
  result.text.reserve(bytes.size());
  forEachIdentifier(
      bytes,
      [&](llvm::StringRef preceding, llvm::StringRef token) {
        result.text.append(preceding.data(), preceding.size());
        if (token.empty())
          return;
        const auto definition = definitionOrdinals.find(token);
        if (definition == definitionOrdinals.end()) {
          result.reservedNameCollision |= token.starts_with(blockNamePrefix);
          result.text.append(token.data(), token.size());
          return;
        }
        ++result.references[definition->second];
        const auto replaced = replacement.find(definition->second);
        if (replaced == replacement.end())
          result.text.append(token.data(), token.size());
        else
          result.text += replaced->second;
      },
      [&](llvm::StringRef escaped) {
        result.reservedNameCollision |= escaped.starts_with(blockNamePrefix);
      });
  return result;
}

llvm::StringRef directionName(RtlModulePortDirection direction) {
  switch (direction) {
  case RtlModulePortDirection::Input:
    return "input";
  case RtlModulePortDirection::Output:
    return "output";
  case RtlModulePortDirection::Inout:
    return "inout";
  }
  llvm_unreachable("unknown RTL port direction");
}

class IdentityEncoder final {
public:
  static llvm::Expected<IdentityEncoder> create() {
    auto builder = BlobDigestBuilder::create();
    if (!builder)
      return builder.takeError();
    IdentityEncoder encoder(std::move(*builder));
    if (llvm::Error error = encoder.text(identityDomain))
      return std::move(error);
    return encoder;
  }

  llvm::Error field(llvm::StringRef key, llvm::StringRef value) {
    if (llvm::Error error = text(key + "=" + llvm::Twine(value.size()) + ":"))
      return error;
    if (llvm::Error error = text(value))
      return error;
    return text("\n");
  }

  llvm::Error ports(llvm::ArrayRef<RtlModulePortProjection> ports) {
    if (llvm::Error error = field("port_count", std::to_string(ports.size())))
      return error;
    for (const RtlModulePortProjection &port : ports) {
      if (llvm::Error error =
              field("port_direction", directionName(port.direction)))
        return error;
      if (llvm::Error error = field("port_name", port.name))
        return error;
      if (llvm::Error error = field("port_type", port.type))
        return error;
      if (llvm::Error error = field("port_attributes", port.attributes))
        return error;
    }
    return llvm::Error::success();
  }

  llvm::Expected<BlobDigest> finish() { return builder_.finish(); }

private:
  explicit IdentityEncoder(BlobDigestBuilder builder)
      : builder_(std::move(builder)) {}

  llvm::Error text(const llvm::Twine &value) {
    llvm::SmallString<128> storage;
    const llvm::StringRef bytes = value.toStringRef(storage);
    return builder_.update(llvm::ArrayRef<std::uint8_t>(
        reinterpret_cast<const std::uint8_t *>(bytes.data()), bytes.size()));
  }

  BlobDigestBuilder builder_;
};

/// The per-definition facts of the closure derivation before members are
/// merged by identity.
struct DefinitionFacts final {
  BlobDigest identity;
  /// Direct children by projection ordinal, mirroring the projection.
  std::vector<RtlModuleDependency> dependencies;
};

llvm::Expected<std::optional<std::string>>
domainPort(const RtlModuleProjection &root,
           const std::optional<std::string> &name, llvm::StringRef role) {
  if (!name)
    return std::optional<std::string>();
  for (const RtlModulePortProjection &port : root.ports) {
    if (port.name != *name)
      continue;
    if (port.direction != RtlModulePortDirection::Input || port.type != "i1")
      return invalid(role + " port '" + *name +
                     "' does not have the domain port shape");
    return std::optional<std::string>(port.name);
  }
  return std::optional<std::string>();
}

} // namespace

std::string rtlBlockName(const BlobDigest &identity) {
  return blockNamePrefix.str() + formatBlobDigestHex(identity);
}

llvm::Expected<RtlBlockClosure>
deriveRtlBlockClosure(const RtlModuleGraphProjection &graph,
                      const RtlModuleGraphSourceBinding &source,
                      std::size_t rootModule,
                      const RtlDomainPortNames &domainPorts) {
  if (rootModule >= graph.modules.size() ||
      source.moduleBytes().size() != graph.modules.size())
    return invalid("root definition is outside the bound module graph");
  if (graph.modules[rootModule].kind != RtlModuleDefinitionKind::Concrete)
    return invalid("root definition is not concrete");

  llvm::StringMap<std::size_t> definitionOrdinals;
  for (std::size_t ordinal = 0; ordinal != graph.modules.size(); ++ordinal)
    if (!definitionOrdinals
             .try_emplace(graph.modules[ordinal].emittedName, ordinal)
             .second)
      return invalid("two definitions share one emitted name");

  // Content identities in postorder from the root; every child's identity is
  // complete before its parents encode it.
  std::vector<std::optional<DefinitionFacts>> facts(graph.modules.size());
  std::vector<std::size_t> postorder;
  std::vector<bool> active(graph.modules.size(), false);
  const auto visit = [&](auto &&self, std::size_t ordinal) -> llvm::Error {
    if (ordinal >= graph.modules.size())
      return invalid("dependency ordinal is outside the module graph");
    if (facts[ordinal])
      return llvm::Error::success();
    if (active[ordinal])
      return invalid("module instance graph contains a cycle");
    active[ordinal] = true;
    const RtlModuleProjection &definition = graph.modules[ordinal];
    for (const RtlModuleDependency &dependency : definition.dependencies) {
      if (dependency.multiplicity == 0)
        return invalid("module dependency has zero multiplicity");
      if (llvm::Error error = self(self, dependency.targetModule))
        return error;
    }
    auto encoder = IdentityEncoder::create();
    if (!encoder)
      return encoder.takeError();
    if (llvm::Error error = encoder->field("preamble", source.preamble()))
      return error;
    if (definition.kind == RtlModuleDefinitionKind::External) {
      if (llvm::Error error = encoder->field("kind", "external"))
        return error;
      if (llvm::Error error = encoder->field("name", definition.emittedName))
        return error;
      if (llvm::Error error = encoder->ports(definition.ports))
        return error;
      if (llvm::Error error =
              encoder->field("parameters", definition.parameters))
        return error;
    } else {
      if (llvm::Error error = encoder->field("kind", "concrete"))
        return error;
      if (llvm::Error error = encoder->ports(definition.ports))
        return error;
      if (llvm::Error error =
              encoder->field("parameters", definition.parameters))
        return error;
      std::map<std::string, std::uint64_t> childrenByIdentity;
      std::map<std::size_t, std::string> replacement{
          {ordinal, selfPlaceholder.str()}};
      for (const RtlModuleDependency &dependency : definition.dependencies) {
        const DefinitionFacts &child = *facts[dependency.targetModule];
        auto &multiplicity =
            childrenByIdentity[formatBlobDigestHex(child.identity)];
        if (dependency.multiplicity >
            std::numeric_limits<std::uint64_t>::max() - multiplicity)
          return invalid("merged child multiplicity overflow");
        multiplicity += dependency.multiplicity;
        if (graph.modules[dependency.targetModule].kind ==
            RtlModuleDefinitionKind::Concrete)
          replacement.emplace(dependency.targetModule,
                              rtlBlockName(child.identity));
      }
      if (llvm::Error error = encoder->field(
              "child_count", std::to_string(childrenByIdentity.size())))
        return error;
      for (const auto &[identity, multiplicity] : childrenByIdentity) {
        if (llvm::Error error = encoder->field("child", identity))
          return error;
        if (llvm::Error error = encoder->field("child_multiplicity",
                                               std::to_string(multiplicity)))
          return error;
      }
      RewrittenEmission rewritten = rewriteEmission(
          source.moduleBytes()[ordinal], definitionOrdinals, replacement);
      if (rewritten.reservedNameCollision)
        return invalid("emission uses the reserved block name namespace");
      // The textual rewrite is exact only when the emitted bytes reference
      // definitions exactly as the projection's instance DAG states.
      for (const auto &[referenced, count] : rewritten.references) {
        if (referenced == ordinal) {
          if (count != 1)
            return invalid("definition '" + definition.emittedName +
                           "' names itself other than once");
          continue;
        }
        const std::size_t referencedOrdinal = referenced;
        const auto dependency =
            llvm::find_if(definition.dependencies, [&](const auto &entry) {
              return entry.targetModule == referencedOrdinal;
            });
        if (dependency == definition.dependencies.end() ||
            dependency->multiplicity != count)
          return invalid("definition '" + definition.emittedName +
                         "' references '" +
                         graph.modules[referenced].emittedName +
                         "' outside its instance multiplicity");
      }
      if (rewritten.references.count(ordinal) == 0)
        return invalid("definition '" + definition.emittedName +
                       "' does not name itself in its emission");
      for (const RtlModuleDependency &dependency : definition.dependencies)
        if (rewritten.references.count(dependency.targetModule) == 0)
          return invalid("definition '" + definition.emittedName +
                         "' emission lacks an instance of '" +
                         graph.modules[dependency.targetModule].emittedName +
                         "'");
      if (llvm::Error error = encoder->field("emission", rewritten.text))
        return error;
    }
    auto identity = encoder->finish();
    if (!identity)
      return identity.takeError();
    facts[ordinal] = DefinitionFacts{*identity, definition.dependencies};
    active[ordinal] = false;
    postorder.push_back(ordinal);
    return llvm::Error::success();
  };
  if (llvm::Error error = visit(visit, rootModule))
    return std::move(error);

  for (std::size_t ordinal : postorder) {
    const RtlModuleProjection &definition = graph.modules[ordinal];
    if (llvm::StringRef(definition.emittedName).starts_with(blockNamePrefix) &&
        (definition.kind == RtlModuleDefinitionKind::External ||
         definition.emittedName != rtlBlockName(facts[ordinal]->identity)))
      return invalid("definition collides with the content block namespace");
  }

  // Members merge every definition of one identity; the first completed
  // definition is the representative and fixes the dependency position.
  RtlBlockClosure closure;
  std::map<std::string, std::size_t> memberByIdentity;
  std::vector<std::size_t> memberOfDefinition(
      graph.modules.size(), std::numeric_limits<std::size_t>::max());
  for (std::size_t ordinal : postorder) {
    const std::string identity = formatBlobDigestHex(facts[ordinal]->identity);
    auto [entry, inserted] =
        memberByIdentity.emplace(identity, closure.members.size());
    if (inserted)
      closure.members.push_back(
          RtlBlockClosureMember{facts[ordinal]->identity, {}, {}, 0});
    closure.members[entry->second].definitions.push_back(ordinal);
    memberOfDefinition[ordinal] = entry->second;
  }
  if (memberOfDefinition[rootModule] != closure.members.size() - 1)
    return invalid("root definition is not the last closure member");
  for (RtlBlockClosureMember &member : closure.members) {
    llvm::sort(member.definitions);
    std::map<std::size_t, std::uint64_t> children;
    for (const RtlModuleDependency &dependency :
         facts[member.definitions.front()]->dependencies) {
      auto &multiplicity =
          children[memberOfDefinition[dependency.targetModule]];
      if (dependency.multiplicity >
          std::numeric_limits<std::uint64_t>::max() - multiplicity)
        return invalid("merged child multiplicity overflow");
      multiplicity += dependency.multiplicity;
    }
    for (const auto &[child, multiplicity] : children)
      member.children.push_back({child, multiplicity});
  }
  // Definition ordinals contain occurrence names. Order the merged DAG by
  // dependency depth and content identity so the complete rendered closure is
  // occurrence-free as well as each individual definition.
  std::vector<std::size_t> depth(closure.members.size(), 0);
  std::vector<std::size_t> order;
  for (std::size_t index = 0; index < closure.members.size(); ++index) {
    for (const auto &child : closure.members[index].children) {
      if (child.member >= index)
        return invalid("closure member order violates the dependency DAG");
      depth[index] = std::max(depth[index], depth[child.member] + 1);
    }
    order.push_back(index);
  }
  llvm::sort(order, [&](std::size_t lhs, std::size_t rhs) {
    if (depth[lhs] != depth[rhs])
      return depth[lhs] < depth[rhs];
    return closure.members[lhs].identity.bytes() <
           closure.members[rhs].identity.bytes();
  });
  std::vector<std::size_t> destination(order.size());
  for (std::size_t index = 0; index < order.size(); ++index)
    destination[order[index]] = index;
  std::vector<RtlBlockClosureMember> ordered;
  for (std::size_t sourceIndex : order) {
    RtlBlockClosureMember member = std::move(closure.members[sourceIndex]);
    for (auto &child : member.children)
      child.member = destination[child.member];
    llvm::sort(member.children, [](const auto &lhs, const auto &rhs) {
      return lhs.member < rhs.member;
    });
    ordered.push_back(std::move(member));
  }
  closure.members = std::move(ordered);
  closure.members.back().instanceCount = 1;
  for (std::size_t index = closure.members.size(); index-- != 0;)
    for (const RtlBlockClosureChild &child : closure.members[index].children) {
      if (child.member >= index)
        return invalid("closure member order violates the dependency DAG");
      const std::uint64_t weighted =
          closure.members[index].instanceCount * child.multiplicity;
      if (closure.members[index].instanceCount != 0 &&
          weighted / closure.members[index].instanceCount != child.multiplicity)
        return invalid("closure instance count overflow");
      if (weighted > std::numeric_limits<std::uint64_t>::max() -
                         closure.members[child.member].instanceCount)
        return invalid("closure instance count overflow");
      closure.members[child.member].instanceCount += weighted;
    }

  const RtlModuleProjection &root = graph.modules[rootModule];
  auto clock = domainPort(root, domainPorts.clock, "clock");
  if (!clock)
    return clock.takeError();
  auto reset = domainPort(root, domainPorts.reset, "reset");
  if (!reset)
    return reset.takeError();
  closure.clockPort = std::move(*clock);
  closure.resetPort = std::move(*reset);
  return closure;
}

llvm::Expected<RtlBlockSourceProjection>
projectRtlBlockClosureSource(const RtlBlockClosure &closure,
                             const RtlModuleGraphProjection &graph,
                             const RtlModuleGraphSourceBinding &source) {
  if (closure.members.empty() ||
      source.moduleBytes().size() != graph.modules.size())
    return invalid("closure is not bound to the module graph source");
  llvm::StringMap<std::size_t> definitionOrdinals;
  for (std::size_t ordinal = 0; ordinal != graph.modules.size(); ++ordinal)
    definitionOrdinals.try_emplace(graph.modules[ordinal].emittedName, ordinal);
  const auto digestOf = [](llvm::StringRef bytes) {
    return computeBlobDigest(llvm::ArrayRef<std::uint8_t>(
        reinterpret_cast<const std::uint8_t *>(bytes.data()), bytes.size()));
  };
  RtlBlockSourceProjection result;
  result.source = source.preamble().str();
  if (!source.preamble().empty())
    result.graph.preamble = RtlModuleEmissionRange{0, source.preamble().size(),
                                                   digestOf(source.preamble())};
  result.graph.topModule = closure.root();
  for (const RtlBlockClosureMember &member : closure.members) {
    if (member.definitions.empty() ||
        member.definitions.front() >= graph.modules.size())
      return invalid("closure member has no representative definition");
    const std::size_t representative = member.definitions.front();
    const RtlModuleProjection &definition = graph.modules[representative];
    const std::string blockName = rtlBlockName(member.identity);
    RtlModuleProjection normalized;
    normalized.emittedName =
        definition.kind == RtlModuleDefinitionKind::External
            ? definition.emittedName
            : blockName;
    normalized.irSymbol = normalized.emittedName;
    normalized.kind = definition.kind;
    normalized.reachable = true;
    normalized.ports = definition.ports;
    normalized.parameters = definition.parameters;
    for (const auto &child : member.children)
      normalized.dependencies.push_back({child.member, child.multiplicity});
    if (definition.kind == RtlModuleDefinitionKind::External) {
      result.graph.modules.push_back(std::move(normalized));
      continue;
    }
    std::map<std::size_t, std::string> replacement;
    for (std::size_t ordinal : member.definitions)
      replacement.emplace(ordinal, blockName);
    for (const RtlBlockClosureChild &child : member.children)
      for (std::size_t ordinal : closure.members[child.member].definitions)
        if (graph.modules[ordinal].kind == RtlModuleDefinitionKind::Concrete)
          replacement.emplace(
              ordinal, rtlBlockName(closure.members[child.member].identity));
    RewrittenEmission rewritten = rewriteEmission(
        source.moduleBytes()[representative], definitionOrdinals, replacement);
    if (rewritten.reservedNameCollision)
      return invalid("emission uses the reserved block name namespace");
    normalized.emission = RtlModuleEmissionRange{
        result.source.size(), rewritten.text.size(), digestOf(rewritten.text)};
    result.source += rewritten.text;
    result.graph.modules.push_back(std::move(normalized));
  }
  result.graph.sourceByteCount = result.source.size();
  result.graph.sourceDigest = digestOf(result.source);
  return result;
}

} // namespace loom::hardware::rtl
