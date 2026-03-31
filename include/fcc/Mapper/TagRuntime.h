#ifndef FCC_MAPPER_TAGRUNTIME_H
#define FCC_MAPPER_TAGRUNTIME_H

#include "fcc/Mapper/Graph.h"
#include "fcc/Mapper/MappingState.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include <optional>

namespace fcc {

const Node *getPortOwnerNode(const Graph &graph, IdIndex portId);

std::optional<uint64_t> applyMapTagTableValue(const Node *mapTagNode,
                                              std::optional<uint64_t> tag);

std::optional<uint64_t> projectRuntimeTagValueToType(std::optional<uint64_t> tag,
                                                     mlir::Type type);

std::optional<uint64_t> computeRuntimeTagValueAlongPath(
    llvm::ArrayRef<IdIndex> hwPath, size_t uptoIndex, const Graph &adg,
    llvm::function_ref<std::optional<uint64_t>(IdIndex)> externalTagAtPort =
        nullptr);

std::optional<uint64_t> computeRuntimeTagValueAlongMappedPath(
    IdIndex swEdgeId, llvm::ArrayRef<IdIndex> hwPath, size_t uptoIndex,
    const MappingState &state, const Graph &dfg, const Graph &adg);

} // namespace fcc

#endif // FCC_MAPPER_TAGRUNTIME_H
