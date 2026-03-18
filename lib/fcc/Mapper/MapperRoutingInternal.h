#ifndef FCC_MAPPER_MAPPERROUTINGINTERNAL_H
#define FCC_MAPPER_MAPPERROUTINGINTERNAL_H

#include "fcc/Mapper/Graph.h"
#include "fcc/Mapper/MappingState.h"

#include "llvm/ADT/StringRef.h"

namespace fcc {
namespace routing_detail {

/// Returns true if the node has resource_class == "routing".
bool isRoutingNode(const Node *node);

/// Returns true if portId is an output port of a multi-port routing crossbar.
bool isRoutingCrossbarOutputPort(IdIndex portId, const Graph &adg);

/// Returns true if opName is a software memory interface operation.
bool isSoftwareMemoryInterfaceOpName(llvm::StringRef opName);

} // namespace routing_detail
} // namespace fcc

#endif // FCC_MAPPER_MAPPERROUTINGINTERNAL_H
