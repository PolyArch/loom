//===-- SystemADGMLIRBuilder.cpp - MLIR builder for system ADG ----*- C++ -*-===//
//
// Part of the loom project.
//
//===----------------------------------------------------------------------===//
//
// Implementation of SystemADGMLIRBuilder which constructs a proper typed MLIR
// module for system-level ADG representation, using fabric.router,
// fabric.shared_mem, and fabric.noc_link ops instead of string concatenation.
//
//===----------------------------------------------------------------------===//

#include "loom/ADG/SystemADGMLIRBuilder.h"
#include "loom/Dialect/Fabric/FabricDialect.h"
#include "loom/Dialect/Fabric/FabricOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <set>

namespace loom {
namespace adg {

mlir::ModuleOp SystemADGMLIRBuilder::build(
    mlir::MLIRContext *ctx, const std::string &systemName,
    const std::vector<CoreType> &coreTypes,
    const std::vector<SystemCoreInstance> &instances, const NoCSpec &nocSpec,
    const SharedMemorySpec &sharedMemSpec) {

  mlir::OpBuilder builder(ctx);
  auto loc = builder.getUnknownLoc();

  // Create the top-level MLIR module
  auto wrapper = mlir::ModuleOp::create(loc);
  builder.setInsertionPointToEnd(wrapper.getBody());

  // Emit per-core fabric.module definitions
  emitCoreTypeDefinitions(builder, wrapper, coreTypes, instances);

  // Emit the system-level fabric.module
  emitSystemModule(builder, wrapper, systemName, coreTypes, instances,
                   nocSpec, sharedMemSpec);

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

  mlir::MLIRContext *ctx = wrapper.getContext();

  // Suppress diagnostic output during core type parsing. Core types are
  // generated via string concatenation by ADGBuilder and may contain
  // constructs that are valid in their original context but fail when
  // parsed into a fresh context (e.g., SSA references across module
  // boundaries). When parsing succeeds we clone the fabric.module; when
  // it fails we silently skip -- the core types are kept as separate
  // files and the system module references them by name.
  mlir::ScopedDiagnosticHandler diagHandler(
      ctx, [](mlir::Diagnostic &) { return mlir::success(); });

  for (unsigned typeId : usedTypeIds) {
    if (typeId >= coreTypes.size())
      continue;

    const auto &coreType = coreTypes[typeId];
    const std::string &mlirText = coreType.mlirText;

    // Parse the core type MLIR text to extract the fabric.module definition.
    auto parsed = mlir::parseSourceString<mlir::ModuleOp>(
        llvm::StringRef(mlirText), ctx);
    if (!parsed) {
      // Parsing failed -- core type will be referenced by name only.
      continue;
    }

    // Find and clone the fabric.module ops from the parsed module
    for (auto &op : parsed->getBody()->getOperations()) {
      if (mlir::isa<fabric::ModuleOp>(op)) {
        builder.setInsertionPointToEnd(wrapper.getBody());
        builder.clone(op);
        break;
      }
    }
  }
}

void SystemADGMLIRBuilder::emitSystemModule(
    mlir::OpBuilder &builder, mlir::ModuleOp wrapper,
    const std::string &systemName,
    const std::vector<CoreType> &coreTypes,
    const std::vector<SystemCoreInstance> &instances,
    const NoCSpec &nocSpec, const SharedMemorySpec &sharedMemSpec) {

  mlir::MLIRContext *ctx = wrapper.getContext();
  auto loc = builder.getUnknownLoc();

  builder.setInsertionPointToEnd(wrapper.getBody());

  // Create system-level fabric.module with () -> () signature
  auto emptyFuncType = mlir::FunctionType::get(ctx, {}, {});
  auto sysModule = fabric::ModuleOp::create(
      builder, loc, systemName, emptyFuncType);

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

    // Use the convenience build overload: (results, module_name, sym_name, operands)
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
}

void SystemADGMLIRBuilder::emitRouters(
    mlir::OpBuilder &builder, mlir::Location loc,
    const std::vector<SystemCoreInstance> &instances,
    const NoCSpec &nocSpec) {

  // Create one router per core instance
  for (size_t i = 0; i < instances.size(); ++i) {
    std::string routerName = "router_" + std::to_string(i);

    // Mesh routers typically have 5 ports (N, S, E, W, local)
    uint64_t numPorts = 5;
    if (nocSpec.topology == NoCSpec::RING)
      numPorts = 3; // fwd, rev, local

    // Use the convenience build overload with StringRef + uint64_t args
    fabric::RouterOp::create(
        builder, loc,
        /*sym_name=*/llvm::StringRef(routerName),
        /*num_ports=*/numPorts,
        /*virtual_channels=*/static_cast<uint64_t>(nocSpec.virtualChannels),
        /*buffer_depth=*/static_cast<uint64_t>(4),
        /*pipeline_stages=*/static_cast<uint64_t>(nocSpec.routerPipelineStages),
        /*flit_width_bits=*/static_cast<uint64_t>(nocSpec.flitWidth));
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
        /*mem_type=*/llvm::StringRef("l2_cache"));
  }

  // Emit external memory interface
  fabric::SharedMemOp::create(
      builder, loc,
      /*sym_name=*/llvm::StringRef("ext_mem_if"),
      /*size_bytes=*/sharedMemSpec.l2SizeBytes,
      /*width_bytes=*/static_cast<uint64_t>(sharedMemSpec.bankWidthBytes),
      /*num_banks=*/static_cast<uint64_t>(sharedMemSpec.numBanks),
      /*mem_type=*/llvm::StringRef("external_dram"));
}

void SystemADGMLIRBuilder::emitMeshLinks(
    mlir::OpBuilder &builder, mlir::Location loc,
    const std::vector<SystemCoreInstance> &instances,
    const NoCSpec &nocSpec) {

  // Build a lookup: (row, col) -> instance index
  auto findInstanceIdx = [&](int row, int col) -> int {
    for (size_t i = 0; i < instances.size(); ++i) {
      if (instances[i].row == row && instances[i].col == col)
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

    for (const auto &dir : dirs) {
      int nr = inst.row + dir.dr;
      int nc = inst.col + dir.dc;
      int neighborIdx = findInstanceIdx(nr, nc);
      if (neighborIdx < 0)
        continue;

      std::string dstRouter = "router_" + std::to_string(neighborIdx);

      fabric::NoCLinkOp::create(
          builder, loc,
          /*source=*/llvm::StringRef(srcRouter),
          /*source_port=*/static_cast<uint64_t>(dir.egressPort),
          /*dest=*/llvm::StringRef(dstRouter),
          /*dest_port=*/static_cast<uint64_t>(dir.ingressPort),
          /*width_bits=*/static_cast<uint64_t>(nocSpec.flitWidth),
          /*latency_cycles=*/
          static_cast<uint64_t>(nocSpec.routerPipelineStages));
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
    fabric::NoCLinkOp::create(
        builder, loc,
        /*source=*/llvm::StringRef(srcRouter),
        /*source_port=*/static_cast<uint64_t>(0),
        /*dest=*/llvm::StringRef(dstRouter),
        /*dest_port=*/static_cast<uint64_t>(0),
        /*width_bits=*/static_cast<uint64_t>(nocSpec.flitWidth),
        /*latency_cycles=*/
        static_cast<uint64_t>(nocSpec.routerPipelineStages));

    // Reverse link
    fabric::NoCLinkOp::create(
        builder, loc,
        /*source=*/llvm::StringRef(dstRouter),
        /*source_port=*/static_cast<uint64_t>(1),
        /*dest=*/llvm::StringRef(srcRouter),
        /*dest_port=*/static_cast<uint64_t>(1),
        /*width_bits=*/static_cast<uint64_t>(nocSpec.flitWidth),
        /*latency_cycles=*/
        static_cast<uint64_t>(nocSpec.routerPipelineStages));
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
        fabric::NoCLinkOp::create(
            builder, loc,
            /*source=*/llvm::StringRef(routerI),
            /*source_port=*/static_cast<uint64_t>(0),
            /*dest=*/llvm::StringRef(routerJ),
            /*dest_port=*/static_cast<uint64_t>(0),
            /*width_bits=*/static_cast<uint64_t>(nocSpec.flitWidth),
            /*latency_cycles=*/
            static_cast<uint64_t>(nocSpec.routerPipelineStages));

        fabric::NoCLinkOp::create(
            builder, loc,
            /*source=*/llvm::StringRef(routerJ),
            /*source_port=*/static_cast<uint64_t>(0),
            /*dest=*/llvm::StringRef(routerI),
            /*dest_port=*/static_cast<uint64_t>(0),
            /*width_bits=*/static_cast<uint64_t>(nocSpec.flitWidth),
            /*latency_cycles=*/
            static_cast<uint64_t>(nocSpec.routerPipelineStages));
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

      fabric::NoCLinkOp::create(
          builder, loc,
          /*source=*/llvm::StringRef(srcRouter),
          /*source_port=*/static_cast<uint64_t>(0),
          /*dest=*/llvm::StringRef(dstRouter),
          /*dest_port=*/static_cast<uint64_t>(0),
          /*width_bits=*/static_cast<uint64_t>(nocSpec.flitWidth),
          /*latency_cycles=*/
          static_cast<uint64_t>(nocSpec.routerPipelineStages));
    }
  }
}

} // namespace adg
} // namespace loom
