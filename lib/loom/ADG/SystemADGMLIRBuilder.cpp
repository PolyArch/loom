//===-- SystemADGMLIRBuilder.cpp - MLIR builder for system ADG ----*- C++ -*-===//
//
// Part of the loom project.
//
//===----------------------------------------------------------------------===//
//
// Implementation of SystemADGMLIRBuilder which constructs a proper typed MLIR
// module for system-level ADG representation, using fabric.router,
// fabric.shared_mem, and fabric.noc_link ops. Core type definitions are
// cloned directly from provided ModuleOps, eliminating string parsing.
//
//===----------------------------------------------------------------------===//

#include "loom/ADG/SystemADGMLIRBuilder.h"
#include "loom/Dialect/Fabric/FabricDialect.h"
#include "loom/Dialect/Fabric/FabricOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"

#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <set>

namespace loom {
namespace adg {

namespace {

void setLinkKind(mlir::Operation *op, mlir::OpBuilder &builder,
                 llvm::StringRef kind) {
  if (!kind.empty())
    op->setAttr("link_kind", builder.getStringAttr(kind));
}

} // namespace

mlir::ModuleOp SystemADGMLIRBuilder::build(
    mlir::MLIRContext *ctx, const std::string &systemName,
    const std::vector<CoreType> &coreTypes,
    const std::vector<SystemCoreInstance> &instances, const NoCSpec &nocSpec,
    const SharedMemorySpec &sharedMemSpec,
    const std::vector<NoCLinkSpec> &explicitLinks,
    const ShgMetadata *metadata) {

  mlir::OpBuilder builder(ctx);
  auto loc = builder.getUnknownLoc();

  // Create the top-level MLIR module
  auto wrapper = mlir::ModuleOp::create(loc);
  builder.setInsertionPointToEnd(wrapper.getBody());

  if (metadata && metadata->enabled)
    emitShgMetadata(builder, wrapper.getOperation(), *metadata);

  // Emit per-core fabric.module definitions
  emitCoreTypeDefinitions(builder, wrapper, coreTypes, instances);

  // Emit the system-level fabric.module
  emitSystemModule(builder, wrapper, systemName, coreTypes, instances,
                   nocSpec, sharedMemSpec, explicitLinks, metadata);

  return wrapper;
}

void SystemADGMLIRBuilder::emitCoreTypeDefinitions(
    mlir::OpBuilder &builder, mlir::ModuleOp wrapper,
    const std::vector<CoreType> &coreTypes,
    const std::vector<SystemCoreInstance> &instances) {

  // Determine which core types are actually used
  std::set<unsigned> usedTypeIds;
  for (const auto &inst : instances)
    usedTypeIds.insert(inst.coreType.id);

  for (unsigned typeId : usedTypeIds) {
    if (typeId >= coreTypes.size())
      continue;

    const auto &coreType = coreTypes[typeId];
    mlir::ModuleOp coreModule = coreType.coreModule;

    if (!coreModule)
      continue;

    // Clone fabric.module ops from the provided ModuleOp directly.
    // No string parsing needed -- the caller already has a live ModuleOp.
    for (auto &op : coreModule.getBody()->getOperations()) {
      if (mlir::isa<fabric::ModuleOp>(op)) {
        builder.setInsertionPointToEnd(wrapper.getBody());
        builder.clone(op);
        break;
      }
    }
  }
}

void SystemADGMLIRBuilder::emitShgMetadata(mlir::OpBuilder &builder,
                                           mlir::Operation *module,
                                           const ShgMetadata &metadata) {
  if (!metadata.enabled)
    return;

  module->setAttr("shg_id", builder.getStringAttr(metadata.shgId));
  module->setAttr("domain", builder.getStringAttr(metadata.domain));
  module->setAttr("total_cores",
                  builder.getI64IntegerAttr(metadata.totalCores));
  module->setAttr("mesh_rows", builder.getI64IntegerAttr(metadata.meshRows));
  module->setAttr("mesh_cols", builder.getI64IntegerAttr(metadata.meshCols));
  module->setAttr("express_link_count",
                  builder.getI64IntegerAttr(metadata.expressLinkCount));
  module->setAttr("total_area_budget_mm2",
                  builder.getF64FloatAttr(metadata.totalAreaBudgetMm2));
  module->setAttr("total_allocated_area_mm2",
                  builder.getF64FloatAttr(metadata.totalAllocatedAreaMm2));
}

void SystemADGMLIRBuilder::emitSystemModule(
    mlir::OpBuilder &builder, mlir::ModuleOp wrapper,
    const std::string &systemName,
    const std::vector<CoreType> &coreTypes,
    const std::vector<SystemCoreInstance> &instances,
    const NoCSpec &nocSpec, const SharedMemorySpec &sharedMemSpec,
    const std::vector<NoCLinkSpec> &explicitLinks,
    const ShgMetadata *metadata) {

  mlir::MLIRContext *ctx = wrapper.getContext();
  auto loc = builder.getUnknownLoc();

  builder.setInsertionPointToEnd(wrapper.getBody());

  // Create system-level fabric.module with () -> () signature
  auto emptyFuncType = mlir::FunctionType::get(ctx, {}, {});
  auto sysModule = fabric::ModuleOp::create(
      builder, loc, systemName, emptyFuncType);
  if (metadata && metadata->enabled)
    emitShgMetadata(builder, sysModule.getOperation(), *metadata);

  // Ensure the body region has a block (the generated build only adds a
  // region, not a block). Then add a fabric.yield terminator.
  mlir::Region &bodyRegion = sysModule.getBody();
  if (bodyRegion.empty())
    bodyRegion.emplaceBlock();

  mlir::Block &body = bodyRegion.front();

  // If no terminator exists yet, add a fabric.yield
  if (body.empty() || !body.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
    builder.setInsertionPointToEnd(&body);
    fabric::YieldOp::create(builder, loc, mlir::ValueRange{});
  }

  // Set insertion point before the terminator (fabric.yield)
  mlir::Operation *terminator = body.getTerminator();
  builder.setInsertionPoint(terminator);

  // Emit core instances
  for (const auto &inst : instances) {
    if (inst.coreType.id >= coreTypes.size())
      continue;

    const auto &coreType = coreTypes[inst.coreType.id];

    auto instanceOp = fabric::InstanceOp::create(
        builder, loc,
        /*results=*/mlir::TypeRange{},
        /*module=*/coreType.typeName,
        /*sym_name=*/builder.getStringAttr(inst.instanceName),
        /*operands=*/mlir::ValueRange{});

    // Add grid position metadata
    instanceOp->setAttr("grid_row",
                        builder.getI64IntegerAttr(inst.row));
    instanceOp->setAttr("grid_col",
                        builder.getI64IntegerAttr(inst.col));
    instanceOp->setAttr("core_id", builder.getI64IntegerAttr(inst.coreId));
    if (!inst.khgType.empty())
      instanceOp->setAttr("khg_type", builder.getStringAttr(inst.khgType));
    instanceOp->setAttr("mesh_row",
                        builder.getI64IntegerAttr(inst.meshRow));
    instanceOp->setAttr("mesh_col",
                        builder.getI64IntegerAttr(inst.meshCol));
  }

  // Emit routers
  emitRouters(builder, loc, instances, nocSpec);

  // Emit shared memory
  emitSharedMemory(builder, loc, sharedMemSpec);

  // Emit NoC links based on topology
  switch (nocSpec.topology) {
  case NoCSpec::MESH:
    emitMeshLinks(builder, loc, instances, nocSpec);
    break;
  case NoCSpec::RING:
    emitRingLinks(builder, loc, instances, nocSpec);
    break;
  case NoCSpec::HIERARCHICAL:
    emitHierarchicalLinks(builder, loc, instances, nocSpec);
    break;
  }

  emitExplicitLinks(builder, loc, explicitLinks);
}

void SystemADGMLIRBuilder::emitExplicitLinks(
    mlir::OpBuilder &builder, mlir::Location loc,
    const std::vector<NoCLinkSpec> &links) {
  for (const auto &link : links) {
    auto op = fabric::NoCLinkOp::create(
        builder, loc,
        /*source=*/llvm::StringRef(link.source),
        /*source_port=*/static_cast<uint64_t>(link.sourcePort),
        /*dest=*/llvm::StringRef(link.dest),
        /*dest_port=*/static_cast<uint64_t>(link.destPort),
        /*width_bits=*/static_cast<uint64_t>(link.widthBits),
        /*latency_cycles=*/static_cast<uint64_t>(link.latencyCycles),
        /*bandwidth=*/static_cast<uint64_t>(link.bandwidth));
    setLinkKind(op, builder, link.linkKind.empty() ? "explicit"
                                                  : link.linkKind);
  }
}

void SystemADGMLIRBuilder::emitRouters(
    mlir::OpBuilder &builder, mlir::Location loc,
    const std::vector<SystemCoreInstance> &instances,
    const NoCSpec &nocSpec) {

  // Derive routing_strategy and topology_role from the topology enum
  llvm::StringRef routingStrategy;
  llvm::StringRef topologyRole;
  switch (nocSpec.topology) {
  case NoCSpec::MESH:
    routingStrategy = "xy_dor";
    topologyRole = "mesh";
    break;
  case NoCSpec::RING:
    routingStrategy = "xy_dor";
    topologyRole = "ring";
    break;
  case NoCSpec::HIERARCHICAL:
    routingStrategy = "adaptive";
    topologyRole = "hierarchical";
    break;
  }

  // Create one router per core instance
  for (size_t i = 0; i < instances.size(); ++i) {
    std::string routerName = "router_" + std::to_string(i);
    const auto &inst = instances[i];

    // Mesh routers typically have 5 ports (N, S, E, W, local)
    uint64_t numPorts = 5;
    if (nocSpec.topology == NoCSpec::RING)
      numPorts = 3; // fwd, rev, local
    if (inst.routerPortCount != 0)
      numPorts = inst.routerPortCount;

    auto router = fabric::RouterOp::create(
        builder, loc,
        /*sym_name=*/llvm::StringRef(routerName),
        /*num_ports=*/numPorts,
        /*virtual_channels=*/static_cast<uint64_t>(nocSpec.virtualChannels),
        /*buffer_depth=*/static_cast<uint64_t>(4),
        /*pipeline_stages=*/static_cast<uint64_t>(nocSpec.routerPipelineStages),
        /*flit_width_bits=*/static_cast<uint64_t>(nocSpec.flitWidth),
        /*routing_strategy=*/routingStrategy,
        /*topology_role=*/topologyRole);
    if (inst.routerPortCount != 0)
      router->setAttr("router_kind",
                      builder.getStringAttr(inst.khgType.empty() ? "express"
                                                                  : inst.khgType));
  }
}

void SystemADGMLIRBuilder::emitSharedMemory(
    mlir::OpBuilder &builder, mlir::Location loc,
    const SharedMemorySpec &sharedMemSpec) {

  if (sharedMemSpec.numBanks == 0)
    return;

  uint64_t bankSize = sharedMemSpec.l2SizeBytes / sharedMemSpec.numBanks;

  // Emit L2 cache bank ops
  for (unsigned b = 0; b < sharedMemSpec.numBanks; ++b) {
    std::string bankName = "l2_bank_" + std::to_string(b);
    fabric::SharedMemOp::create(
        builder, loc,
        /*sym_name=*/llvm::StringRef(bankName),
        /*size_bytes=*/bankSize,
        /*width_bytes=*/static_cast<uint64_t>(sharedMemSpec.bankWidthBytes),
        /*num_banks=*/static_cast<uint64_t>(1),
        /*mem_type=*/llvm::StringRef("l2_cache"),
        /*port_count=*/static_cast<uint64_t>(1));
  }

  // Emit external memory interface
  fabric::SharedMemOp::create(
      builder, loc,
      /*sym_name=*/llvm::StringRef("ext_mem_if"),
      /*size_bytes=*/sharedMemSpec.l2SizeBytes,
      /*width_bytes=*/static_cast<uint64_t>(sharedMemSpec.bankWidthBytes),
      /*num_banks=*/static_cast<uint64_t>(sharedMemSpec.numBanks),
      /*mem_type=*/llvm::StringRef("external_dram"),
      /*port_count=*/static_cast<uint64_t>(sharedMemSpec.numBanks));
}

void SystemADGMLIRBuilder::emitMeshLinks(
    mlir::OpBuilder &builder, mlir::Location loc,
    const std::vector<SystemCoreInstance> &instances,
    const NoCSpec &nocSpec) {

  // Build a lookup: (row, col) -> instance index
  auto findInstanceIdx = [&](int row, int col) -> int {
    for (size_t i = 0; i < instances.size(); ++i) {
      const auto &inst = instances[i];
      int instRow = inst.meshRow >= 0 ? inst.meshRow : inst.row;
      int instCol = inst.meshCol >= 0 ? inst.meshCol : inst.col;
      if (instRow == row && instCol == col)
        return static_cast<int>(i);
    }
    return -1;
  };

  // Direction descriptors: egress port, ingress port, dr, dc
  struct Direction {
    int egressPort;
    int ingressPort;
    int dr, dc;
  };
  // N=0, S=1, E=2, W=3, local=4
  Direction dirs[] = {
      {0, 1, -1, 0},  // North: egress port 0, neighbor ingress port 1 (South)
      {1, 0, 1, 0},   // South: egress port 1, neighbor ingress port 0 (North)
      {2, 3, 0, 1},   // East: egress port 2, neighbor ingress port 3 (West)
      {3, 2, 0, -1},  // West: egress port 3, neighbor ingress port 2 (East)
  };

  for (size_t i = 0; i < instances.size(); ++i) {
    const auto &inst = instances[i];
    std::string srcRouter = "router_" + std::to_string(i);
    int instRow = inst.meshRow >= 0 ? inst.meshRow : inst.row;
    int instCol = inst.meshCol >= 0 ? inst.meshCol : inst.col;

    for (const auto &dir : dirs) {
      int nr = instRow + dir.dr;
      int nc = instCol + dir.dc;
      int neighborIdx = findInstanceIdx(nr, nc);
      if (neighborIdx < 0)
        continue;

      std::string dstRouter = "router_" + std::to_string(neighborIdx);

      auto link = fabric::NoCLinkOp::create(
          builder, loc,
          /*source=*/llvm::StringRef(srcRouter),
          /*source_port=*/static_cast<uint64_t>(dir.egressPort),
          /*dest=*/llvm::StringRef(dstRouter),
          /*dest_port=*/static_cast<uint64_t>(dir.ingressPort),
          /*width_bits=*/static_cast<uint64_t>(nocSpec.flitWidth),
          /*latency_cycles=*/
          static_cast<uint64_t>(nocSpec.routerPipelineStages),
          /*bandwidth=*/static_cast<uint64_t>(nocSpec.linkBandwidth));
      setLinkKind(link, builder, "mesh");
    }
  }
}

void SystemADGMLIRBuilder::emitRingLinks(
    mlir::OpBuilder &builder, mlir::Location loc,
    const std::vector<SystemCoreInstance> &instances,
    const NoCSpec &nocSpec) {

  if (instances.empty())
    return;

  // Connect routers in a ring: router_i -> router_(i+1), last -> first
  for (size_t i = 0; i < instances.size(); ++i) {
    size_t next = (i + 1) % instances.size();
    std::string srcRouter = "router_" + std::to_string(i);
    std::string dstRouter = "router_" + std::to_string(next);

    // Forward link
    auto forward = fabric::NoCLinkOp::create(
        builder, loc,
        /*source=*/llvm::StringRef(srcRouter),
        /*source_port=*/static_cast<uint64_t>(0),
        /*dest=*/llvm::StringRef(dstRouter),
        /*dest_port=*/static_cast<uint64_t>(0),
        /*width_bits=*/static_cast<uint64_t>(nocSpec.flitWidth),
        /*latency_cycles=*/
        static_cast<uint64_t>(nocSpec.routerPipelineStages),
        /*bandwidth=*/static_cast<uint64_t>(nocSpec.linkBandwidth));
    setLinkKind(forward, builder, "ring");

    // Reverse link
    auto reverse = fabric::NoCLinkOp::create(
        builder, loc,
        /*source=*/llvm::StringRef(dstRouter),
        /*source_port=*/static_cast<uint64_t>(1),
        /*dest=*/llvm::StringRef(srcRouter),
        /*dest_port=*/static_cast<uint64_t>(1),
        /*width_bits=*/static_cast<uint64_t>(nocSpec.flitWidth),
        /*latency_cycles=*/
        static_cast<uint64_t>(nocSpec.routerPipelineStages),
        /*bandwidth=*/static_cast<uint64_t>(nocSpec.linkBandwidth));
    setLinkKind(reverse, builder, "ring");
  }
}

void SystemADGMLIRBuilder::emitHierarchicalLinks(
    mlir::OpBuilder &builder, mlir::Location loc,
    const std::vector<SystemCoreInstance> &instances,
    const NoCSpec &nocSpec) {

  // For small systems, fall back to mesh
  if (instances.size() <= 4) {
    emitMeshLinks(builder, loc, instances, nocSpec);
    return;
  }

  // Cluster cores into groups of 4
  size_t numClusters = (instances.size() + 3) / 4;

  // Intra-cluster links (full mesh within each cluster)
  for (size_t cluster = 0; cluster < numClusters; ++cluster) {
    size_t base = cluster * 4;
    size_t count = std::min<size_t>(4, instances.size() - base);

    for (size_t i = base; i < base + count; ++i) {
      for (size_t j = i + 1; j < base + count; ++j) {
        std::string routerI = "router_" + std::to_string(i);
        std::string routerJ = "router_" + std::to_string(j);

        // Bidirectional links within cluster
        auto forward = fabric::NoCLinkOp::create(
            builder, loc,
            /*source=*/llvm::StringRef(routerI),
            /*source_port=*/static_cast<uint64_t>(0),
            /*dest=*/llvm::StringRef(routerJ),
            /*dest_port=*/static_cast<uint64_t>(0),
            /*width_bits=*/static_cast<uint64_t>(nocSpec.flitWidth),
            /*latency_cycles=*/
            static_cast<uint64_t>(nocSpec.routerPipelineStages),
            /*bandwidth=*/static_cast<uint64_t>(nocSpec.linkBandwidth));
        setLinkKind(forward, builder, "hierarchical");

        auto reverse = fabric::NoCLinkOp::create(
            builder, loc,
            /*source=*/llvm::StringRef(routerJ),
            /*source_port=*/static_cast<uint64_t>(0),
            /*dest=*/llvm::StringRef(routerI),
            /*dest_port=*/static_cast<uint64_t>(0),
            /*width_bits=*/static_cast<uint64_t>(nocSpec.flitWidth),
            /*latency_cycles=*/
            static_cast<uint64_t>(nocSpec.routerPipelineStages),
            /*bandwidth=*/static_cast<uint64_t>(nocSpec.linkBandwidth));
        setLinkKind(reverse, builder, "hierarchical");
      }
    }
  }

  // Inter-cluster ring links
  for (size_t c = 0; c < numClusters; ++c) {
    size_t nextC = (c + 1) % numClusters;
    size_t srcIdx = c * 4;
    size_t dstIdx = nextC * 4;
    if (srcIdx < instances.size() && dstIdx < instances.size()) {
      std::string srcRouter = "router_" + std::to_string(srcIdx);
      std::string dstRouter = "router_" + std::to_string(dstIdx);

      auto forward = fabric::NoCLinkOp::create(
          builder, loc,
          /*source=*/llvm::StringRef(srcRouter),
          /*source_port=*/static_cast<uint64_t>(0),
          /*dest=*/llvm::StringRef(dstRouter),
          /*dest_port=*/static_cast<uint64_t>(0),
          /*width_bits=*/static_cast<uint64_t>(nocSpec.flitWidth),
          /*latency_cycles=*/
          static_cast<uint64_t>(nocSpec.routerPipelineStages),
          /*bandwidth=*/static_cast<uint64_t>(nocSpec.linkBandwidth));
      setLinkKind(forward, builder, "hierarchical");
    }
  }
}

} // namespace adg
} // namespace loom
