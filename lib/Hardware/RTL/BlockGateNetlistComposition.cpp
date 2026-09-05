#include "Hardware/RTL/BlockGateNetlistComposition.h"

#include "llvm/ADT/STLExtras.h"

#include <map>
#include <set>

namespace loom::hardware::rtl {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "block_gate_composition_invalid: " + message);
}

} // namespace

llvm::Expected<BlockGateNetlistComposition> composeBlockGateNetlistChildren(
    const FinalizedRtlBlockSource &source,
    llvm::ArrayRef<FinalizedBlockGateNetlist> children,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  const auto &graph = source.projection().graph;
  const auto &dependencies = graph.modules[graph.topModule].dependencies;
  if (children.empty() || children.size() != dependencies.size())
    return invalid("compiled children do not cover the exact direct child set");
  std::map<std::string, RtlModuleDependency> required;
  for (const auto &dependency : dependencies)
    required.emplace(graph.modules[dependency.targetModule].emittedName,
                     dependency);
  std::vector<const FinalizedBlockGateNetlist *> ordered;
  for (const auto &child : children)
    ordered.push_back(&child);
  llvm::sort(ordered, [](const auto *left, const auto *right) {
    return artifactRootReferenceLess(left->reference(), right->reference());
  });
  const auto &target = ordered.front()->netlist();
  BlockGateNetlistComposition result;
  std::set<std::string> unitDigests;
  for (const auto *child : ordered) {
    const auto &netlist = child->netlist();
    if (netlist.implementationPlatform != target.implementationPlatform ||
        netlist.corner != target.corner ||
        netlist.standardCellLibrary != target.standardCellLibrary ||
        netlist.standardCellContract != target.standardCellContract)
      return invalid("compiled children have different technology contracts");
    // The normalized definition name selects its unique parent subgraph;
    // exact Source identity and clock are verified independently below.
    auto expected = required.find(netlist.representation.top.canonicalName);
    if (expected == required.end())
      return invalid("compiled child is extra or repeats another direct child");
    auto childSource = importRtlBlockSource(netlist.source, artifacts, blobs);
    if (!childSource)
      return childSource.takeError();
    if (llvm::Error error = verifyRtlBlockSourceSubgraphDerivation(
            source, expected->second.targetModule, *childSource))
      return std::move(error);
    auto childIndex = indexRepresentationRoot(netlist.representation, blobs);
    if (!childIndex)
      return childIndex.takeError();
    result.children.push_back({expected->first, expected->second.multiplicity,
                               childIndex->rootBoundaryPorts()});
    required.erase(expected);
    for (const auto &payload : netlist.representation.payloads) {
      if (payload.role != PayloadRole::Netlist)
        continue;
      const std::string digest = formatBlobDigestHex(payload.blobDigest);
      if (unitDigests.insert(digest).second)
        result.units.push_back(
            {{PayloadRole::Netlist, "netlist/" + digest + ".v",
              payload.blobDigest},
             child->reference()});
    }
  }
  // Preserve canonical child order and each child's admitted payload order for
  // semantic bundle inputs. The representation catalog has its own owner order.
  std::vector<ImplementationPayload> payloads;
  for (const auto &unit : result.units)
    payloads.push_back(unit.payload);
  auto catalog = canonicalizeImplementationPayloadCatalog(payloads);
  if (!catalog)
    return catalog.takeError();
  auto index = indexRepresentation(target.representation.formatRef,
                                   target.representation.top, *catalog, blobs);
  if (!index)
    return index.takeError();
  for (const auto &definition : index->concreteModuleDefinitions()) {
    if (definition.canonicalName == source.top())
      return invalid("compiled child definition shadows the parent root");
    result.definitions.push_back(definition);
  }
  llvm::sort(result.children, [](const auto &left, const auto &right) {
    return left.definition < right.definition;
  });
  return result;
}

} // namespace loom::hardware::rtl
