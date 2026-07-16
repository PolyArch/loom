#ifndef LOOM_PNR_MAPPING_INTERNAL_H
#define LOOM_PNR_MAPPING_INTERNAL_H

#include "PnR/Mapping.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>

namespace loom::pnr::detail {

enum class ResourceKind {
  FabricOp,
  MemLoad,
  MemStore,
};

struct SoftwareNode {
  std::string id;
  std::string operation;
  ResourceKind resourceKind;
  mlir::Operation *op = nullptr;
};

struct HardwareResource {
  std::string id;
  ResourceKind kind;
  std::string schedule;
  mlir::Operation *op = nullptr;
  std::map<std::string, std::set<std::string>> hwParamOptions;
  std::map<std::string, std::string> swConfigs;
  llvm::StringSet<> supportedOps;
  bool used = false;
};

enum class TransportKind {
  Bits,
  BitsTag,
};

struct TransportShape {
  TransportKind kind = TransportKind::Bits;
  unsigned payloadWidth = 0;
};

struct HardwareEndpoint {
  std::string id;
  TransportShape shape;
};

struct HardwareRouteSegment {
  std::string segmentKind;
  std::string sourceEndpoint;
  std::string sinkEndpoint;
  std::string hardwareRef;
  unsigned payloadCapacity = 0;
};

struct HardwareTopology {
  std::map<std::string, llvm::SmallVector<HardwareRouteSegment, 2>>
      segmentsBySource;
  std::map<std::string, llvm::SmallVector<HardwareRouteSegment, 2>>
      segmentsBySink;
};

struct HardwareModel {
  llvm::SmallVector<HardwareResource> resources;
  HardwareTopology topology;
};

struct RouteCollection {
  llvm::SmallVector<RouteRecord, 0> routes;
  llvm::SmallVector<UnroutedEdgeRecord, 0> unroutedEdgeDetails;
  std::uint64_t unroutedEdges = 0;
};

struct RouteEdgeKey {
  std::string producerSoftwareId;
  unsigned producerResultIndex = 0;
  std::string consumerSoftwareId;
  unsigned consumerOperandIndex = 0;

  bool operator<(const RouteEdgeKey &other) const {
    return std::tie(producerSoftwareId, producerResultIndex, consumerSoftwareId,
                    consumerOperandIndex) <
           std::tie(other.producerSoftwareId, other.producerResultIndex,
                    other.consumerSoftwareId, other.consumerOperandIndex);
  }
};

struct SoftwareRouteEdge {
  RouteEdgeKey key;
  std::string payloadKind;
  unsigned requiredPayloadWidth = 0;
  std::optional<unsigned> producerResultIndex;
  std::optional<unsigned> consumerOperandIndex;
};

struct RoutingProblem {
  llvm::SmallVector<SoftwareRouteEdge, 0> edges;
};

struct RouteCacheKey {
  std::string sourceEndpoint;
  std::string sinkEndpoint;
  unsigned requiredPayloadWidth = 0;

  bool operator<(const RouteCacheKey &other) const {
    return std::tie(sourceEndpoint, sinkEndpoint, requiredPayloadWidth) <
           std::tie(other.sourceEndpoint, other.sinkEndpoint,
                    other.requiredPayloadWidth);
  }
};

struct RouteCache {
  std::map<RouteCacheKey, std::optional<llvm::SmallVector<RouteSegment, 2>>>
      routes;
};

llvm::StringRef resourceKindName(ResourceKind kind);
bool isAdapterOp(mlir::Operation *op);
bool shouldMaterializeAdapterOp(mlir::Operation *op);
bool isStructuredContainerOp(mlir::Operation *op);
bool isPointerBookkeepingOp(mlir::Operation *op);
std::map<std::string, std::string> softwareConfigsFor(const SoftwareNode &node);
std::optional<std::string>
resolvedSoftwareConfigValue(const HardwareResource &resource,
                            llvm::StringRef key, llvm::StringRef value);
std::optional<unsigned> softwareBitWidth(mlir::Type type);
HardwareResource *
claimResource(SoftwareNode &node,
              llvm::MutableArrayRef<HardwareResource> resources);
PlacementRecord makePlacementRecord(const SoftwareNode &node,
                                    const HardwareResource &resource);
bool resourceIsCompatible(const SoftwareNode &node,
                          const HardwareResource &resource);
unsigned compatibleResourceCount(const SoftwareNode &node,
                                 llvm::ArrayRef<HardwareResource> resources);
std::optional<std::string> configFor(const HardwareResource &resource,
                                     llvm::StringRef key);

llvm::Expected<HardwareModel> collectHardwareModel(mlir::Operation *hardware,
                                                   llvm::StringRef name);
llvm::Expected<RoutingProblem>
buildRoutingProblem(llvm::ArrayRef<SoftwareNode> nodes, mlir::Operation *graph);
RouteCollection collectRoutes(const RoutingProblem &problem,
                              llvm::ArrayRef<PlacementRecord> placements,
                              const HardwareTopology &topology,
                              RouteCache &routeCache);
bool placeRouteFeasible(llvm::MutableArrayRef<SoftwareNode> nodes,
                        const RoutingProblem &routingProblem,
                        llvm::MutableArrayRef<HardwareResource> resources,
                        const HardwareTopology &topology,
                        llvm::SmallVectorImpl<PlacementRecord> &placements,
                        RouteCache &routeCache);

llvm::Error appendPlacementConfig(MappingSummary &summary,
                                  const SoftwareNode &node,
                                  const HardwareResource &resource);
void appendRouteConfig(MappingSummary &summary);
llvm::Error validateConfigBitstream(const MappingSummary &summary);

} // namespace loom::pnr::detail

#endif // LOOM_PNR_MAPPING_INTERNAL_H
