#ifndef LOOM_GRAPH_SYSTEMGRAPH_H
#define LOOM_GRAPH_SYSTEMGRAPH_H

/// Header-only SystemGraph<NodeT, EdgeT> template for system-level graphs.
///
/// Provides adjacency-list storage, mutation, traversal, JSON serialization
/// (via llvm::json), and DOT export. No MLIR dependency.

#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

namespace loom {

/// Lightweight read-only span over a contiguous vector range.
/// Avoids copying on each traversal call.
template <typename T>
class ConstSpan {
public:
  using const_iterator = typename std::vector<T>::const_iterator;

  ConstSpan() = default;
  ConstSpan(const_iterator b, const_iterator e) : begin_(b), end_(e) {}
  explicit ConstSpan(const std::vector<T> &vec)
      : begin_(vec.begin()), end_(vec.end()) {}

  const_iterator begin() const { return begin_; }
  const_iterator end() const { return end_; }
  size_t size() const { return static_cast<size_t>(end_ - begin_); }
  bool empty() const { return begin_ == end_; }

private:
  const_iterator begin_{};
  const_iterator end_{};
};

/// Directed graph template parameterized on node and edge payload types.
///
/// NodeT must provide: toJSON(), static fromJSON(), dotLabel().
/// EdgeT must provide: toJSON(), static fromJSON(), dotLabel().
template <typename NodeT, typename EdgeT>
class SystemGraph {
public:
  using NodeId = unsigned;
  using EdgeId = unsigned;

  //===--------------------------------------------------------------------===//
  // Mutation
  //===--------------------------------------------------------------------===//

  /// Add a node and return its id.
  NodeId addNode(NodeT node) {
    NodeId id = static_cast<NodeId>(nodes_.size());
    nodes_.push_back(std::move(node));
    outAdj_.emplace_back();
    inAdj_.emplace_back();
    return id;
  }

  /// Add a directed edge (src -> dst) and return its id.
  /// Asserts that src and dst are valid node ids.
  EdgeId addEdge(NodeId src, NodeId dst, EdgeT data) {
    assert(src < nodes_.size() && "addEdge: invalid src NodeId");
    assert(dst < nodes_.size() && "addEdge: invalid dst NodeId");
    EdgeId eid = static_cast<EdgeId>(edges_.size());
    edges_.push_back({src, dst, std::move(data)});
    outAdj_[src].push_back(eid);
    inAdj_[dst].push_back(eid);
    return eid;
  }

  //===--------------------------------------------------------------------===//
  // Access
  //===--------------------------------------------------------------------===//

  size_t numNodes() const { return nodes_.size(); }
  size_t numEdges() const { return edges_.size(); }

  const NodeT &node(NodeId id) const {
    assert(id < nodes_.size() && "node: invalid NodeId");
    return nodes_[id];
  }
  NodeT &node(NodeId id) {
    assert(id < nodes_.size() && "node: invalid NodeId");
    return nodes_[id];
  }

  const EdgeT &edge(EdgeId id) const {
    assert(id < edges_.size() && "edge: invalid EdgeId");
    return edges_[id].data;
  }
  EdgeT &edge(EdgeId id) {
    assert(id < edges_.size() && "edge: invalid EdgeId");
    return edges_[id].data;
  }

  /// Get the source node id of an edge.
  NodeId edgeSrc(EdgeId id) const {
    assert(id < edges_.size() && "edgeSrc: invalid EdgeId");
    return edges_[id].src;
  }

  /// Get the destination node id of an edge.
  NodeId edgeDst(EdgeId id) const {
    assert(id < edges_.size() && "edgeDst: invalid EdgeId");
    return edges_[id].dst;
  }

  //===--------------------------------------------------------------------===//
  // Traversal
  //===--------------------------------------------------------------------===//

  /// Edge ids of outgoing edges from a node.
  ConstSpan<EdgeId> outEdges(NodeId id) const {
    assert(id < outAdj_.size() && "outEdges: invalid NodeId");
    return ConstSpan<EdgeId>(outAdj_[id]);
  }

  /// Edge ids of incoming edges to a node.
  ConstSpan<EdgeId> inEdges(NodeId id) const {
    assert(id < inAdj_.size() && "inEdges: invalid NodeId");
    return ConstSpan<EdgeId>(inAdj_[id]);
  }

  /// Node ids reachable via outgoing edges from a node.
  std::vector<NodeId> successors(NodeId id) const {
    std::vector<NodeId> result;
    for (EdgeId eid : outEdges(id))
      result.push_back(edges_[eid].dst);
    return result;
  }

  /// Node ids that have edges leading into the given node.
  std::vector<NodeId> predecessors(NodeId id) const {
    std::vector<NodeId> result;
    for (EdgeId eid : inEdges(id))
      result.push_back(edges_[eid].src);
    return result;
  }

  //===--------------------------------------------------------------------===//
  // JSON Serialization
  //===--------------------------------------------------------------------===//

  llvm::json::Value toJSON() const {
    llvm::json::Object root;

    llvm::json::Array nodesArr;
    for (const auto &n : nodes_)
      nodesArr.push_back(n.toJSON());
    root["nodes"] = std::move(nodesArr);

    llvm::json::Array edgesArr;
    for (const auto &e : edges_) {
      llvm::json::Object eObj;
      eObj["src"] = static_cast<int64_t>(e.src);
      eObj["dst"] = static_cast<int64_t>(e.dst);
      eObj["data"] = e.data.toJSON();
      edgesArr.push_back(std::move(eObj));
    }
    root["edges"] = std::move(edgesArr);

    return llvm::json::Value(std::move(root));
  }

  static SystemGraph fromJSON(const llvm::json::Value &val) {
    SystemGraph g;
    auto *root = val.getAsObject();
    if (!root)
      return g;

    if (auto *nodesArr = root->getArray("nodes")) {
      for (const auto &nVal : *nodesArr)
        g.addNode(NodeT::fromJSON(nVal));
    }

    if (auto *edgesArr = root->getArray("edges")) {
      for (const auto &eVal : *edgesArr) {
        auto *eObj = eVal.getAsObject();
        if (!eObj)
          continue;
        auto src = eObj->getInteger("src");
        auto dst = eObj->getInteger("dst");
        auto *dataVal = eObj->get("data");
        if (!src || !dst || !dataVal)
          continue;
        NodeId srcId = static_cast<NodeId>(*src);
        NodeId dstId = static_cast<NodeId>(*dst);
        if (srcId >= g.numNodes() || dstId >= g.numNodes())
          continue;
        g.addEdge(srcId, dstId, EdgeT::fromJSON(*dataVal));
      }
    }

    return g;
  }

  //===--------------------------------------------------------------------===//
  // DOT Export
  //===--------------------------------------------------------------------===//

  void exportDot(llvm::raw_ostream &os) const {
    os << "digraph {\n";
    os << "  rankdir=TB;\n";

    for (size_t i = 0; i < nodes_.size(); ++i) {
      os << "  \"n" << i << "\" [label=\""
         << escapeDot(nodes_[i].dotLabel()) << "\"];\n";
    }

    for (const auto &e : edges_) {
      os << "  \"n" << e.src << "\" -> \"n" << e.dst << "\"";
      std::string label = e.data.dotLabel();
      if (!label.empty())
        os << " [label=\"" << escapeDot(label) << "\"]";
      os << ";\n";
    }

    os << "}\n";
  }

private:
  struct EdgeEntry {
    NodeId src;
    NodeId dst;
    EdgeT data;
  };

  std::vector<NodeT> nodes_;
  std::vector<EdgeEntry> edges_;
  std::vector<std::vector<EdgeId>> outAdj_;
  std::vector<std::vector<EdgeId>> inAdj_;

  /// Escape characters that are special in DOT label strings.
  static std::string escapeDot(const std::string &s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
      switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '<':
        out += "&lt;";
        break;
      case '>':
        out += "&gt;";
        break;
      case '&':
        out += "&amp;";
        break;
      default:
        out += c;
        break;
      }
    }
    return out;
  }
};

} // namespace loom

#endif // LOOM_GRAPH_SYSTEMGRAPH_H
