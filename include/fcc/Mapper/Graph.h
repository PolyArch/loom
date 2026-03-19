#ifndef FCC_MAPPER_GRAPH_H
#define FCC_MAPPER_GRAPH_H

#include "fcc/Mapper/Types.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"

#include <memory>
#include <vector>

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace fcc {

class Port {
public:
  enum Direction { Input, Output };

  IdIndex parentNode = INVALID_ID;
  Direction direction = Input;
  mlir::Type type;
  llvm::SmallVector<IdIndex, 2> connectedEdges;
  llvm::SmallVector<mlir::NamedAttribute, 4> attributes;
};

class Edge {
public:
  IdIndex srcPort = INVALID_ID;
  IdIndex dstPort = INVALID_ID;
  llvm::SmallVector<mlir::NamedAttribute, 4> attributes;
};

class Node {
public:
  enum Kind {
    OperationNode,
    ModuleInputNode,
    ModuleOutputNode,
  };

  Kind kind = OperationNode;
  llvm::SmallVector<IdIndex, 4> inputPorts;
  llvm::SmallVector<IdIndex, 4> outputPorts;
  llvm::SmallVector<mlir::NamedAttribute, 8> attributes;

  // Cached attribute values for hot-path lookups. Populated by
  // Graph::buildAttributeCache() after graph construction.
  llvm::StringRef cachedResourceClass;
  llvm::StringRef cachedPeName;
  llvm::StringRef cachedOpName;
  llvm::StringRef cachedPeKind;
  llvm::StringRef cachedOpKind;
  bool attributeCacheValid = false;
};

/// Iterator adaptor that skips nullptr entries in a unique_ptr vector.
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
    using VecIter = typename std::vector<std::unique_ptr<T>>::const_iterator;
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

class Graph {
public:
  explicit Graph(mlir::MLIRContext *ctx = nullptr) : context(ctx) {}

  std::vector<std::unique_ptr<Node>> nodes;
  std::vector<std::unique_ptr<Port>> ports;
  std::vector<std::unique_ptr<Edge>> edges;

  mlir::MLIRContext *context = nullptr;

  Graph clone() const;

  void reserve(size_t nodeHint, size_t portHint, size_t edgeHint) {
    nodes.reserve(nodeHint);
    ports.reserve(portHint);
    edges.reserve(edgeHint);
  }

  IdIndex addNode(std::unique_ptr<Node> node);
  IdIndex addPort(std::unique_ptr<Port> port);
  IdIndex addEdge(std::unique_ptr<Edge> edge);

  void removeNode(IdIndex id);
  void removePort(IdIndex id);
  void removeEdge(IdIndex id);

  Node *getNode(IdIndex id) const;
  Port *getPort(IdIndex id) const;
  Edge *getEdge(IdIndex id) const;

  bool isValid(IdIndex id, EntityKind kind) const;

  size_t countNodes() const;
  size_t countPorts() const;
  size_t countEdges() const;

  NonNullRange<Node> nodeRange() const { return NonNullRange<Node>(nodes); }
  NonNullRange<Port> portRange() const { return NonNullRange<Port>(ports); }
  NonNullRange<Edge> edgeRange() const { return NonNullRange<Edge>(edges); }

  /// Populate cached StringRef fields on every Node for hot-path attribute
  /// lookups. Call after graph construction is complete.
  void buildAttributeCache();
};

// Attribute helper utilities for Node.
llvm::StringRef getNodeAttrStr(const Node *node, llvm::StringRef key);
int64_t getNodeAttrInt(const Node *node, llvm::StringRef key,
                       int64_t defaultVal = 0);
bool nodeHasAttr(const Node *node, llvm::StringRef key);

} // namespace fcc

#endif // FCC_MAPPER_GRAPH_H
