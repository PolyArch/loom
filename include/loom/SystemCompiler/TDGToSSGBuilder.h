//===-- TDGToSSGBuilder.h - TDG MLIR -> SSG conversion ------------*- C++ -*-===//
//
// Converts a TDG MLIR module (produced by tapestry::emitTDG) into a
// System Scheduling Graph (SSG) for consumption by the hierarchical compiler.
//
// The SSG is a lightweight directed graph of KernelNode entries connected by
// DataDependency edges. Each KernelNode carries compute profile data and
// variant information extracted from the corresponding DFG modules.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_SYSTEMCOMPILER_TDGTOSSGBUILDER_H
#define LOOM_SYSTEMCOMPILER_TDGTOSSGBUILDER_H

#include "loom/SystemCompiler/L1CoreAssignment.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace loom {

//===----------------------------------------------------------------------===//
// SSG node type: KernelNode
//===----------------------------------------------------------------------===//

/// A node in the System Scheduling Graph representing one kernel.
struct KernelNode {
  /// Kernel name (matches the tdg.kernel sym_name).
  std::string name;

  /// Kernel execution type string ("cgra", "host", "auto").
  std::string kernelType;

  /// Set of variant names available for this kernel.
  /// Each variant typically corresponds to a DFG module key like
  /// "kernelName_v0", "kernelName_v1", etc.
  std::set<std::string> variantSet;

  /// Compute profile extracted from DFG analysis.
  KernelProfile computeProfile;

  /// Whether this node has a valid DFG (false if DFG module was missing).
  bool hasDFG = false;
};

//===----------------------------------------------------------------------===//
// SSG edge type: SSGDataDependency
//===----------------------------------------------------------------------===//

/// An edge in the System Scheduling Graph representing a data dependency
/// between a producer and consumer kernel.
struct SSGDataDependency {
  /// Name of the producer kernel.
  std::string producerName;

  /// Name of the consumer kernel.
  std::string consumerName;

  /// Data volume in bytes transferred on this edge (0 if unknown).
  uint64_t dataVolume = 0;

  /// Ordering semantics.
  std::string ordering = "FIFO";

  /// Data type name (e.g. "f32").
  std::string dataTypeName;

  /// Memory visibility level.
  std::string visibility = "LOCAL_SPM";
};

//===----------------------------------------------------------------------===//
// SystemGraph template
//===----------------------------------------------------------------------===//

/// A lightweight directed graph of nodes and edges.
///
/// NodeT should have a `std::string name` field.
/// EdgeT should have `std::string producerName` and `std::string consumerName`.
template <typename NodeT, typename EdgeT> class SystemGraph {
public:
  /// Add a node to the graph.
  void addNode(NodeT node) { nodes_.push_back(std::move(node)); }

  /// Add an edge to the graph.
  void addEdge(EdgeT edge) { edges_.push_back(std::move(edge)); }

  /// Number of nodes in the graph.
  size_t numNodes() const { return nodes_.size(); }

  /// Number of edges in the graph.
  size_t numEdges() const { return edges_.size(); }

  /// Access all nodes.
  const std::vector<NodeT> &nodes() const { return nodes_; }
  std::vector<NodeT> &nodes() { return nodes_; }

  /// Access all edges.
  const std::vector<EdgeT> &edges() const { return edges_; }
  std::vector<EdgeT> &edges() { return edges_; }

  /// Find a node by name. Returns nullptr if not found.
  const NodeT *findNode(const std::string &name) const {
    for (const auto &n : nodes_) {
      if (n.name == name)
        return &n;
    }
    return nullptr;
  }

  /// Find a node by name (mutable). Returns nullptr if not found.
  NodeT *findNode(const std::string &name) {
    for (auto &n : nodes_) {
      if (n.name == name)
        return &n;
    }
    return nullptr;
  }

  /// Collect all kernel names in the graph.
  std::set<std::string> kernelNames() const {
    std::set<std::string> names;
    for (const auto &n : nodes_)
      names.insert(n.name);
    return names;
  }

  /// Collect all (producer, consumer) edge pairs.
  std::set<std::pair<std::string, std::string>> edgePairs() const {
    std::set<std::pair<std::string, std::string>> pairs;
    for (const auto &e : edges_)
      pairs.insert({e.producerName, e.consumerName});
    return pairs;
  }

private:
  std::vector<NodeT> nodes_;
  std::vector<EdgeT> edges_;
};

/// The concrete SSG type used throughout the pipeline.
using SSG = SystemGraph<KernelNode, SSGDataDependency>;

//===----------------------------------------------------------------------===//
// TDGToSSGBuilder
//===----------------------------------------------------------------------===//

/// Converts a TDG MLIR module and per-kernel DFG modules into an SSG.
///
/// The builder:
///   1. Walks tdg.kernel ops to create KernelNode entries.
///   2. For each kernel, looks up the corresponding DFG module and profiles
///      it using KernelProfiler to populate computeProfile.
///   3. Walks tdg.contract ops to create SSGDataDependency edges.
///   4. Validates the resulting graph (no duplicate kernel names, DAG check).
class TDGToSSGBuilder {
public:
  /// Build an SSG from a TDG MLIR module and per-kernel DFG modules.
  ///
  /// \param tdgModule   The TDG MLIR module containing tdg.graph, tdg.kernel,
  ///                    and tdg.contract ops.
  /// \param dfgModules  Map from kernel name to its DFG (handshake.func) module.
  ///                    Missing entries produce a warning; the kernel still
  ///                    appears in the SSG but with empty profile/variants.
  /// \param ctx         MLIR context for type queries.
  /// \returns           A populated SSG.
  SSG build(mlir::ModuleOp tdgModule,
            const std::map<std::string, mlir::ModuleOp> &dfgModules,
            mlir::MLIRContext &ctx);
};

} // namespace loom

#endif // LOOM_SYSTEMCOMPILER_TDGTOSSGBUILDER_H
