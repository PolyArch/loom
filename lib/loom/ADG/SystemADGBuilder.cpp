//===-- SystemADGBuilder.cpp - System-level ADG Builder -----------*- C++ -*-===//
//
// Part of the loom project.
//
//===----------------------------------------------------------------------===//
//
// Implementation of the SystemADGBuilder which composes per-core fabric.module
// definitions into a system-level fabric.module with NoC connectivity and
// shared memory hierarchy.
//
//===----------------------------------------------------------------------===//

#include "loom/ADG/SystemADGBuilder.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <map>
#include <set>
#include <sstream>

namespace loom {
namespace adg {

//===----------------------------------------------------------------------===//
// Internal data structures
//===----------------------------------------------------------------------===//

struct CoreTypeDef {
  std::string typeName;
  std::string mlirText;
  unsigned id;
};

struct SystemADGBuilder::Impl {
  std::string systemName;
  std::vector<CoreTypeDef> coreTypes;
  std::vector<SystemCoreInstance> coreInstances;
  NoCSpec nocSpec;
  SharedMemorySpec sharedMemSpec;
  std::string builtMLIR;
  bool isBuilt = false;

  /// Get the core at a specific grid position, or nullptr.
  const SystemCoreInstance *getCoreAt(int row, int col) const;

  /// Generate the system MLIR text.
  std::string generateSystemMLIR() const;

  /// Generate mesh topology connections.
  void generateMeshConnections(std::ostringstream &os,
                               unsigned &connectionIdx) const;

  /// Generate ring topology connections.
  void generateRingConnections(std::ostringstream &os,
                               unsigned &connectionIdx) const;

  /// Generate hierarchical topology connections.
  void generateHierarchicalConnections(std::ostringstream &os,
                                       unsigned &connectionIdx) const;

  /// Generate L2 memory bank instances.
  void generateL2Banks(std::ostringstream &os) const;

  /// Get instance name for core at grid position.
  std::string getCoreInstanceName(int row, int col) const;
};

const SystemCoreInstance *SystemADGBuilder::Impl::getCoreAt(int row,
                                                            int col) const {
  for (const auto &inst : coreInstances) {
    if (inst.row == row && inst.col == col)
      return &inst;
  }
  return nullptr;
}

std::string SystemADGBuilder::Impl::getCoreInstanceName(int row,
                                                         int col) const {
  const auto *core = getCoreAt(row, col);
  if (core)
    return core->instanceName;
  return "core_" + std::to_string(row) + "_" + std::to_string(col);
}

//===----------------------------------------------------------------------===//
// MLIR generation
//===----------------------------------------------------------------------===//

std::string SystemADGBuilder::Impl::generateSystemMLIR() const {
  std::ostringstream os;

  // Emit a wrapping module containing:
  // 1. Core type definitions (each is a fabric.module)
  // 2. System-level fabric.module with instances and connections

  os << "module {\n\n";

  // Emit core type definitions.
  // Each core type's MLIR is a complete module { fabric.module @Name ... }
  // We need to extract just the fabric.module contents from each.
  std::set<unsigned> emittedTypes;
  for (const auto &inst : coreInstances) {
    if (emittedTypes.count(inst.coreType.id))
      continue;
    emittedTypes.insert(inst.coreType.id);

    const auto &coreType = coreTypes[inst.coreType.id];

    // Extract the fabric.module content from the stored MLIR text.
    // The stored text is: module { fabric.module @Name(...) { ... } }
    // We need to emit just the inner fabric.module definition.
    const std::string &mlir = coreType.mlirText;
    size_t fabModStart = mlir.find("fabric.module");
    size_t moduleEnd = mlir.rfind("}");

    if (fabModStart != std::string::npos && moduleEnd != std::string::npos) {
      // Find the closing brace of the fabric.module (second-to-last })
      size_t searchEnd = moduleEnd;
      size_t fabModEnd = mlir.rfind("}", searchEnd - 1);
      if (fabModEnd != std::string::npos) {
        os << mlir.substr(fabModStart, fabModEnd - fabModStart + 1) << "\n\n";
      }
    }
  }

  // Emit the system-level fabric.module
  os << "fabric.module @" << systemName << "() -> () {\n";

  // Emit core instances
  for (const auto &inst : coreInstances) {
    const auto &coreType = coreTypes[inst.coreType.id];
    os << "  // Core instance: " << inst.instanceName << " at ("
       << inst.row << ", " << inst.col << ")\n";
    os << "  // type: " << coreType.typeName << "\n";
    os << "  // fabric.instance @" << coreType.typeName << " {sym_name = \""
       << inst.instanceName << "\", grid_row = " << inst.row
       << ", grid_col = " << inst.col << "}\n";
  }

  // Emit NoC connections based on topology
  unsigned connectionIdx = 0;
  os << "\n  // NoC connections (topology: ";
  switch (nocSpec.topology) {
  case NoCSpec::MESH:
    os << "MESH";
    break;
  case NoCSpec::RING:
    os << "RING";
    break;
  case NoCSpec::HIERARCHICAL:
    os << "HIERARCHICAL";
    break;
  }
  os << ", flit_width: " << nocSpec.flitWidth
     << ", virtual_channels: " << nocSpec.virtualChannels
     << ", link_bandwidth: " << nocSpec.linkBandwidth
     << ", router_pipeline_stages: " << nocSpec.routerPipelineStages << ")\n";

  switch (nocSpec.topology) {
  case NoCSpec::MESH:
    generateMeshConnections(os, connectionIdx);
    break;
  case NoCSpec::RING:
    generateRingConnections(os, connectionIdx);
    break;
  case NoCSpec::HIERARCHICAL:
    generateHierarchicalConnections(os, connectionIdx);
    break;
  }

  // Emit L2 memory banks
  generateL2Banks(os);

  os << "  fabric.yield\n";
  os << "}\n\n";
  os << "}\n";

  return os.str();
}

void SystemADGBuilder::Impl::generateMeshConnections(
    std::ostringstream &os, unsigned &connectionIdx) const {
  // For each core, connect to adjacent neighbors (N, S, E, W)
  // Determine grid bounds
  int maxRow = 0, maxCol = 0;
  for (const auto &inst : coreInstances) {
    maxRow = std::max(maxRow, inst.row);
    maxCol = std::max(maxCol, inst.col);
  }

  for (const auto &inst : coreInstances) {
    int r = inst.row;
    int c = inst.col;

    // Check each cardinal direction
    struct Direction {
      const char *name;
      const char *egressDir;
      const char *ingressDir;
      int dr, dc;
    };

    Direction dirs[] = {
        {"North", "N", "S", -1, 0},
        {"South", "S", "N", 1, 0},
        {"East", "E", "W", 0, 1},
        {"West", "W", "E", 0, -1},
    };

    for (const auto &dir : dirs) {
      int nr = r + dir.dr;
      int nc = c + dir.dc;
      const auto *neighbor = getCoreAt(nr, nc);
      if (!neighbor)
        continue;

      // Connect: this core's egress -> neighbor's ingress
      os << "  // noc_link_" << connectionIdx++ << ": " << inst.instanceName
         << ".noc_out_" << dir.egressDir << " -> "
         << neighbor->instanceName << ".noc_in_" << dir.ingressDir << "\n";
    }
  }
}

void SystemADGBuilder::Impl::generateRingConnections(
    std::ostringstream &os, unsigned &connectionIdx) const {
  if (coreInstances.empty())
    return;

  // Connect cores in a ring: core_i -> core_(i+1), last -> first
  for (size_t i = 0; i < coreInstances.size(); ++i) {
    size_t next = (i + 1) % coreInstances.size();
    os << "  // noc_link_" << connectionIdx++ << ": "
       << coreInstances[i].instanceName << ".noc_out_fwd -> "
       << coreInstances[next].instanceName << ".noc_in_fwd\n";
    os << "  // noc_link_" << connectionIdx++ << ": "
       << coreInstances[next].instanceName << ".noc_out_rev -> "
       << coreInstances[i].instanceName << ".noc_in_rev\n";
  }
}

void SystemADGBuilder::Impl::generateHierarchicalConnections(
    std::ostringstream &os, unsigned &connectionIdx) const {
  // Group cores into clusters of 4, connect within cluster as mesh,
  // then connect clusters via a higher-level ring
  if (coreInstances.size() <= 4) {
    // Fall back to mesh for small systems
    generateMeshConnections(os, connectionIdx);
    return;
  }

  // Cluster cores by groups of 4
  size_t numClusters = (coreInstances.size() + 3) / 4;
  os << "  // Hierarchical: " << numClusters << " clusters\n";

  for (size_t cluster = 0; cluster < numClusters; ++cluster) {
    size_t base = cluster * 4;
    size_t count = std::min<size_t>(4, coreInstances.size() - base);

    os << "  // Cluster " << cluster << " (intra-cluster mesh)\n";
    for (size_t i = base; i < base + count; ++i) {
      for (size_t j = i + 1; j < base + count; ++j) {
        os << "  // noc_link_" << connectionIdx++ << ": "
           << coreInstances[i].instanceName << " <-> "
           << coreInstances[j].instanceName << "\n";
      }
    }
  }

  // Inter-cluster ring
  os << "  // Inter-cluster ring\n";
  for (size_t c = 0; c < numClusters; ++c) {
    size_t nextC = (c + 1) % numClusters;
    size_t srcIdx = c * 4;
    size_t dstIdx = nextC * 4;
    if (srcIdx < coreInstances.size() && dstIdx < coreInstances.size()) {
      os << "  // noc_link_" << connectionIdx++ << ": cluster_" << c
         << " -> cluster_" << nextC << " via "
         << coreInstances[srcIdx].instanceName << " -> "
         << coreInstances[dstIdx].instanceName << "\n";
    }
  }
}

void SystemADGBuilder::Impl::generateL2Banks(std::ostringstream &os) const {
  if (sharedMemSpec.numBanks == 0)
    return;

  uint64_t bankSize = sharedMemSpec.l2SizeBytes / sharedMemSpec.numBanks;

  os << "\n  // Shared L2 memory banks\n";
  os << "  // total_size: " << sharedMemSpec.l2SizeBytes << " bytes\n";
  os << "  // banks: " << sharedMemSpec.numBanks << "\n";
  os << "  // bank_size: " << bankSize << " bytes\n";
  os << "  // bank_width: " << sharedMemSpec.bankWidthBytes << " bytes\n";

  for (unsigned b = 0; b < sharedMemSpec.numBanks; ++b) {
    os << "  // l2_bank_" << b << ": size=" << bankSize
       << ", width=" << sharedMemSpec.bankWidthBytes << "\n";
  }

  // External memory interface
  os << "\n  // External memory interface\n";
  os << "  // ext_mem_if: connected to NoC and L2 banks\n";
}

//===----------------------------------------------------------------------===//
// SystemADGBuilder public API
//===----------------------------------------------------------------------===//

SystemADGBuilder::SystemADGBuilder(const std::string &systemName)
    : impl_(std::make_unique<Impl>()) {
  impl_->systemName = systemName;
}

SystemADGBuilder::~SystemADGBuilder() = default;

CoreTypeHandle
SystemADGBuilder::registerCoreType(const std::string &typeName,
                                   const std::string &mlirText) {
  unsigned id = impl_->coreTypes.size();
  impl_->coreTypes.push_back({typeName, mlirText, id});
  return CoreTypeHandle{id};
}

SystemCoreInstance SystemADGBuilder::instantiateCore(CoreTypeHandle type,
                                                     const std::string &name,
                                                     int row, int col) {
  if (type.id >= impl_->coreTypes.size()) {
    llvm::report_fatal_error("SystemADGBuilder: invalid CoreTypeHandle");
  }

  SystemCoreInstance inst;
  inst.instanceName = name;
  inst.coreType = type;
  inst.row = row;
  inst.col = col;

  impl_->coreInstances.push_back(inst);
  return inst;
}

void SystemADGBuilder::setNoCSpec(const NoCSpec &spec) {
  impl_->nocSpec = spec;
}

void SystemADGBuilder::setSharedMemorySpec(const SharedMemorySpec &spec) {
  impl_->sharedMemSpec = spec;
}

void SystemADGBuilder::build() {
  if (impl_->coreInstances.empty()) {
    llvm::report_fatal_error(
        "SystemADGBuilder::build(): no core instances registered");
  }

  impl_->builtMLIR = impl_->generateSystemMLIR();
  impl_->isBuilt = true;
}

void SystemADGBuilder::exportSystemMLIR(const std::string &path) {
  if (!impl_->isBuilt) {
    llvm::report_fatal_error(
        "SystemADGBuilder::exportSystemMLIR(): must call build() first");
  }

  std::error_code ec;
  llvm::raw_fd_ostream output(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::report_fatal_error(
        llvm::Twine("SystemADGBuilder: cannot write output file: ") + path +
        "\n" + ec.message());
  }

  output << impl_->builtMLIR;
  output.flush();
}

std::string SystemADGBuilder::getSystemMLIR() const {
  if (!impl_->isBuilt) {
    llvm::report_fatal_error(
        "SystemADGBuilder::getSystemMLIR(): must call build() first");
  }
  return impl_->builtMLIR;
}

const std::vector<SystemCoreInstance> &
SystemADGBuilder::getCoreInstances() const {
  return impl_->coreInstances;
}

const NoCSpec &SystemADGBuilder::getNoCSpec() const {
  return impl_->nocSpec;
}

const SharedMemorySpec &SystemADGBuilder::getSharedMemorySpec() const {
  return impl_->sharedMemSpec;
}

} // namespace adg
} // namespace loom
