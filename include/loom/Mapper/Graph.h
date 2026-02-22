//===-- Graph.h - Mapper graph data model -------------------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Unified Graph container for representing both software dataflow graphs (DFGs)
// and hardware architecture description graphs (ADGs). All entities use the
// ID-as-Index principle: the position of an entity in its owning vector IS its
// ID. Deletion sets slots to nullptr without shifting subsequent entries.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_MAPPER_GRAPH_H
#define LOOM_MAPPER_GRAPH_H

#include "loom/Mapper/Types.h"

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"

#include <memory>
#include <vector>

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace loom {

//===----------------------------------------------------------------------===//
// Port
//===----------------------------------------------------------------------===//

class Port {
public:
  enum Direction { Input, Output };

  IdIndex parentNode = INVALID_ID;
  Direction direction = Input;
  mlir::Type type;
  llvm::SmallVector<IdIndex, 2> connectedEdges;
  llvm::SmallVector<mlir::NamedAttribute, 4> attributes;
};

//===----------------------------------------------------------------------===//
// Edge
//===----------------------------------------------------------------------===//

class Edge {
public:
  IdIndex srcPort = INVALID_ID;
  IdIndex dstPort = INVALID_ID;
  llvm::SmallVector<mlir::NamedAttribute, 4> attributes;
};

//===----------------------------------------------------------------------===//
// Node
//===----------------------------------------------------------------------===//

class Node {
public:
  enum Kind {
    OperationNode,    // Software operation (DFG) or hardware resource (ADG)
    ModuleInputNode,  // Sentinel: represents module input argument
    ModuleOutputNode, // Sentinel: represents module output
  };

  Kind kind = OperationNode;
  llvm::SmallVector<IdIndex, 4> inputPorts;
  llvm::SmallVector<IdIndex, 4> outputPorts;
  llvm::SmallVector<mlir::NamedAttribute, 8> attributes;
};

//===----------------------------------------------------------------------===//
// NonNullRange - iterator adaptor that skips nullptr entries
//===----------------------------------------------------------------------===//

template <typename T>
class NonNullRange {
public:
  class iterator {
  public:
    using difference_type = std::ptrdiff_t;
    using value_type = T *;
    using pointer = T *const *;
    using reference = T *;
    using iterator_category = std::forward_iterator_tag;

    iterator() = default;

    reference operator*() const { return it_->get(); }

    iterator &operator++() {
      ++it_;
      advance();
      return *this;
    }

    iterator operator++(int) {
      auto tmp = *this;
      ++(*this);
      return tmp;
    }

    bool operator==(const iterator &other) const { return it_ == other.it_; }
    bool operator!=(const iterator &other) const { return it_ != other.it_; }

  private:
    friend class NonNullRange;
    using VecIter =
        typename std::vector<std::unique_ptr<T>>::const_iterator;
    VecIter it_;
    VecIter end_;

    iterator(VecIter it, VecIter end) : it_(it), end_(end) { advance(); }

    void advance() {
      while (it_ != end_ && !*it_)
        ++it_;
    }
  };

  explicit NonNullRange(const std::vector<std::unique_ptr<T>> &vec)
      : begin_(vec.begin(), vec.end()), end_(vec.end(), vec.end()) {}

  iterator begin() const { return begin_; }
  iterator end() const { return end_; }

private:
  iterator begin_;
  iterator end_;
};

//===----------------------------------------------------------------------===//
// Graph
//===----------------------------------------------------------------------===//

class Graph {
public:
  explicit Graph(mlir::MLIRContext *ctx = nullptr) : context(ctx) {}

  std::vector<std::unique_ptr<Node>> nodes;
  std::vector<std::unique_ptr<Port>> ports;
  std::vector<std::unique_ptr<Edge>> edges;

  mlir::MLIRContext *context = nullptr;

  /// Append a node, returning its ID (== index in nodes vector).
  IdIndex addNode(std::unique_ptr<Node> node);

  /// Append a port, returning its ID (== index in ports vector).
  IdIndex addPort(std::unique_ptr<Port> port);

  /// Append an edge, returning its ID (== index in edges vector).
  IdIndex addEdge(std::unique_ptr<Edge> edge);

  /// Remove a node. Cascade-deletes owned ports (and their connected edges).
  /// Ports not owned by this node (parentNode != id) are left intact.
  void removeNode(IdIndex id);

  /// Remove a port. Cascade-deletes all connected edges. Removes port ID
  /// from its parent node's port lists.
  void removePort(IdIndex id);

  /// Remove an edge. Removes edge ID from both endpoint ports'
  /// connectedEdges lists.
  void removeEdge(IdIndex id);

  /// Return pointer to node at id, or nullptr if deleted/out-of-range.
  Node *getNode(IdIndex id) const;

  /// Return pointer to port at id, or nullptr if deleted/out-of-range.
  Port *getPort(IdIndex id) const;

  /// Return pointer to edge at id, or nullptr if deleted/out-of-range.
  Edge *getEdge(IdIndex id) const;

  /// Check whether an entity at the given id is valid (non-null, in-range).
  bool isValid(IdIndex id, EntityKind kind) const;

  /// Count non-null entries.
  size_t countNodes() const;
  size_t countPorts() const;
  size_t countEdges() const;

  /// Range-based iteration over non-null entries.
  NonNullRange<Node> nodeRange() const { return NonNullRange<Node>(nodes); }
  NonNullRange<Port> portRange() const { return NonNullRange<Port>(ports); }
  NonNullRange<Edge> edgeRange() const { return NonNullRange<Edge>(edges); }
};

} // namespace loom

#endif // LOOM_MAPPER_GRAPH_H
