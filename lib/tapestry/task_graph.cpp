#include "tapestry/task_graph.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <sstream>
#include <unordered_map>

namespace tapestry {

// ============================================================================
// Enum <-> string helpers
// ============================================================================

const char *orderingToString(Ordering o) {
  switch (o) {
  case Ordering::FIFO:
    return "FIFO";
  case Ordering::UNORDERED:
    return "UNORDERED";
  }
  return "FIFO";
}

Ordering orderingFromString(const std::string &s) {
  if (s == "UNORDERED")
    return Ordering::UNORDERED;
  return Ordering::FIFO;
}

const char *visibilityToString(Visibility v) {
  switch (v) {
  case Visibility::LOCAL_SPM:
    return "LOCAL_SPM";
  case Visibility::SHARED_L2:
    return "SHARED_L2";
  case Visibility::EXTERNAL_DRAM:
    return "EXTERNAL_DRAM";
  }
  return "LOCAL_SPM";
}

Visibility visibilityFromString(const std::string &s) {
  if (s == "SHARED_L2")
    return Visibility::SHARED_L2;
  if (s == "EXTERNAL_DRAM")
    return Visibility::EXTERNAL_DRAM;
  return Visibility::LOCAL_SPM;
}

const char *backpressureToString(Backpressure bp) {
  switch (bp) {
  case Backpressure::BLOCK:
    return "BLOCK";
  case Backpressure::DROP:
    return "DROP";
  case Backpressure::OVERWRITE:
    return "OVERWRITE";
  }
  return "BLOCK";
}

Backpressure backpressureFromString(const std::string &s) {
  if (s == "DROP")
    return Backpressure::DROP;
  if (s == "OVERWRITE")
    return Backpressure::OVERWRITE;
  return Backpressure::BLOCK;
}

const char *executionTargetToString(ExecutionTarget t) {
  switch (t) {
  case ExecutionTarget::CGRA:
    return "CGRA";
  case ExecutionTarget::HOST:
    return "HOST";
  case ExecutionTarget::AUTO_DETECT:
    return "AUTO_DETECT";
  }
  return "AUTO_DETECT";
}

ExecutionTarget executionTargetFromString(const std::string &s) {
  if (s == "CGRA")
    return ExecutionTarget::CGRA;
  if (s == "HOST")
    return ExecutionTarget::HOST;
  return ExecutionTarget::AUTO_DETECT;
}

// ============================================================================
// TaskGraph::Impl -- pimpl holding adjacency list and metadata
// ============================================================================

struct TaskGraph::Impl {
  std::string graphName;
  KernelMap kernels;
  EdgeMap edges;

  // Adjacency lists: successors and predecessors by kernel index.
  std::vector<std::vector<unsigned>> adj;   // adj[i] = successors of kernel i
  std::vector<std::vector<unsigned>> preds; // preds[i] = predecessors of i

  // Name -> index lookup.
  std::unordered_map<std::string, unsigned> nameIndex;

  explicit Impl(const std::string &name) : graphName(name) {}
};

// ============================================================================
// TaskGraph construction / destruction / move
// ============================================================================

TaskGraph::TaskGraph(const std::string &name)
    : impl_(std::make_unique<Impl>(name)) {}

TaskGraph::~TaskGraph() = default;

TaskGraph::TaskGraph(TaskGraph &&) noexcept = default;
TaskGraph &TaskGraph::operator=(TaskGraph &&) noexcept = default;

// ============================================================================
// Kernel definition
// ============================================================================

KernelHandle TaskGraph::addKernelImpl(KernelInfo info) {
  unsigned idx = static_cast<unsigned>(impl_->kernels.size());
  impl_->nameIndex[info.name] = idx;
  impl_->kernels.push_back(std::move(info));
  impl_->adj.emplace_back();   // empty successors list
  impl_->preds.emplace_back(); // empty predecessors list

  return KernelHandle(this, idx);
}

KernelHandle TaskGraph::kernel(const std::string &kernelName) {
  KernelInfo info;
  info.name = kernelName;
  info.provenance.functionName = kernelName;
  return addKernelImpl(std::move(info));
}

// ============================================================================
// Edge definition
// ============================================================================

EdgeHandle TaskGraph::connect(KernelHandle producer, KernelHandle consumer) {
  assert(producer.graph_ == this && "producer does not belong to this graph");
  assert(consumer.graph_ == this && "consumer does not belong to this graph");
  assert(producer.index_ < impl_->kernels.size() && "invalid producer index");
  assert(consumer.index_ < impl_->kernels.size() && "invalid consumer index");

  // Establish topology in adjacency list.
  impl_->adj[producer.index_].push_back(consumer.index_);
  impl_->preds[consumer.index_].push_back(producer.index_);

  // Insert an empty contract into the EdgeMap.
  EdgeKey key{producer.index_, consumer.index_};
  impl_->edges.emplace(key, Contract{});

  return EdgeHandle(this, key);
}

EdgeHandle TaskGraph::edge(const std::string &producerName,
                           const std::string &consumerName) {
  unsigned pidx = kernelIndex(producerName);
  unsigned cidx = kernelIndex(consumerName);
  assert(pidx != static_cast<unsigned>(-1) && "unknown producer kernel name");
  assert(cidx != static_cast<unsigned>(-1) && "unknown consumer kernel name");
  EdgeKey key{pidx, cidx};
  assert(impl_->edges.count(key) && "no edge between the named kernels");
  return EdgeHandle(this, key);
}

// ============================================================================
// Inspection
// ============================================================================

void TaskGraph::dump() const {
  std::cout << "TaskGraph: " << impl_->graphName << "\n";
  std::cout << "  Kernels (" << impl_->kernels.size() << "):\n";
  for (unsigned i = 0; i < impl_->kernels.size(); ++i) {
    const auto &k = impl_->kernels[i];
    std::cout << "    [" << i << "] " << k.name
              << "  target=" << executionTargetToString(k.target) << "\n";
  }

  std::cout << "  Edges (" << impl_->edges.size() << "):\n";
  for (const auto &[key, contract] : impl_->edges) {
    const auto &pname = impl_->kernels[key.first].name;
    const auto &cname = impl_->kernels[key.second].name;
    std::cout << "    " << pname << " -> " << cname;
    if (contract.ordering)
      std::cout << "  ordering=" << orderingToString(*contract.ordering);
    if (contract.dataTypeName)
      std::cout << "  data_type=" << *contract.dataTypeName;
    if (contract.rate)
      std::cout << "  rate=" << *contract.rate;
    if (contract.tileShape) {
      std::cout << "  tile_shape=[";
      for (size_t j = 0; j < contract.tileShape->size(); ++j) {
        if (j > 0)
          std::cout << ",";
        std::cout << (*contract.tileShape)[j];
      }
      std::cout << "]";
    }
    if (contract.visibility)
      std::cout << "  visibility=" << visibilityToString(*contract.visibility);
    std::cout << "\n";
  }
}

std::string TaskGraph::dumpDot() const {
  std::ostringstream os;
  os << "digraph \"" << impl_->graphName << "\" {\n";
  os << "  rankdir=TB;\n";
  for (unsigned i = 0; i < impl_->kernels.size(); ++i) {
    os << "  n" << i << " [label=\"" << impl_->kernels[i].name << "\"];\n";
  }
  for (const auto &[key, contract] : impl_->edges) {
    os << "  n" << key.first << " -> n" << key.second;
    // Annotate with key contract info.
    bool hasLabel = false;
    std::ostringstream label;
    if (contract.ordering) {
      label << orderingToString(*contract.ordering);
      hasLabel = true;
    }
    if (contract.dataTypeName) {
      if (hasLabel) label << "\\n";
      label << *contract.dataTypeName;
      hasLabel = true;
    }
    if (hasLabel)
      os << " [label=\"" << label.str() << "\"]";
    os << ";\n";
  }
  os << "}\n";
  return os.str();
}

size_t TaskGraph::numKernels() const { return impl_->kernels.size(); }

size_t TaskGraph::numEdges() const { return impl_->edges.size(); }

void TaskGraph::forEachKernel(
    std::function<void(const KernelInfo &)> visitor) const {
  for (const auto &k : impl_->kernels)
    visitor(k);
}

void TaskGraph::forEachEdge(
    std::function<void(const std::string &, const std::string &,
                       const Contract &)> visitor) const {
  for (const auto &[key, contract] : impl_->edges) {
    visitor(impl_->kernels[key.first].name,
            impl_->kernels[key.second].name, contract);
  }
}

// ============================================================================
// Internal access
// ============================================================================

const EdgeMap &TaskGraph::edges() const { return impl_->edges; }

const KernelMap &TaskGraph::kernels() const { return impl_->kernels; }

const std::string &TaskGraph::name() const { return impl_->graphName; }

unsigned TaskGraph::kernelIndex(const std::string &kernelName) const {
  auto it = impl_->nameIndex.find(kernelName);
  if (it != impl_->nameIndex.end())
    return it->second;
  return static_cast<unsigned>(-1);
}

const std::vector<std::vector<unsigned>> &TaskGraph::adjacency() const {
  return impl_->adj;
}

const std::vector<std::vector<unsigned>> &TaskGraph::predecessors() const {
  return impl_->preds;
}

Contract &TaskGraph::contractRef(EdgeKey key) { return impl_->edges.at(key); }

const Contract &TaskGraph::contractRef(EdgeKey key) const {
  return impl_->edges.at(key);
}

const KernelInfo &TaskGraph::kernelRef(unsigned idx) const {
  assert(idx < impl_->kernels.size());
  return impl_->kernels[idx];
}

KernelInfo &TaskGraph::kernelRef(unsigned idx) {
  assert(idx < impl_->kernels.size());
  return impl_->kernels[idx];
}

// ============================================================================
// EdgeHandle implementation
// ============================================================================

EdgeHandle &EdgeHandle::ordering(Ordering o) {
  graph_->contractRef(key_).ordering = o;
  return *this;
}

EdgeHandle &EdgeHandle::data_type(const std::string &typeName) {
  graph_->contractRef(key_).dataTypeName = typeName;
  return *this;
}

EdgeHandle &EdgeHandle::rate(int64_t r) {
  graph_->contractRef(key_).rate = r;
  return *this;
}

EdgeHandle &EdgeHandle::tile_shape(std::vector<int64_t> shape) {
  graph_->contractRef(key_).tileShape = std::move(shape);
  return *this;
}

EdgeHandle &EdgeHandle::visibility(Visibility v) {
  graph_->contractRef(key_).visibility = v;
  return *this;
}

EdgeHandle &EdgeHandle::double_buffering(bool enable) {
  graph_->contractRef(key_).doubleBuffering = enable;
  return *this;
}

EdgeHandle &EdgeHandle::backpressure(Backpressure bp) {
  graph_->contractRef(key_).backpressure = bp;
  return *this;
}

EdgeHandle &EdgeHandle::may_fuse(bool b) {
  graph_->contractRef(key_).mayFuse = b;
  return *this;
}

EdgeHandle &EdgeHandle::may_replicate(bool b) {
  graph_->contractRef(key_).mayReplicate = b;
  return *this;
}

EdgeHandle &EdgeHandle::may_pipeline(bool b) {
  graph_->contractRef(key_).mayPipeline = b;
  return *this;
}

EdgeHandle &EdgeHandle::may_reorder(bool b) {
  graph_->contractRef(key_).mayReorder = b;
  return *this;
}

EdgeHandle &EdgeHandle::may_retile(bool b) {
  graph_->contractRef(key_).mayRetile = b;
  return *this;
}

const Contract &EdgeHandle::contract() const {
  return graph_->contractRef(key_);
}

const std::string &EdgeHandle::producerName() const {
  return graph_->kernelRef(key_.first).name;
}

const std::string &EdgeHandle::consumerName() const {
  return graph_->kernelRef(key_.second).name;
}

// ============================================================================
// KernelHandle implementation
// ============================================================================

KernelHandle &KernelHandle::target(ExecutionTarget t) {
  graph_->kernelRef(index_).target = t;
  return *this;
}

const std::string &KernelHandle::name() const {
  return graph_->kernelRef(index_).name;
}

ExecutionTarget KernelHandle::executionTarget() const {
  return graph_->kernelRef(index_).target;
}

const KernelProvenance &KernelHandle::provenance() const {
  return graph_->kernelRef(index_).provenance;
}

} // namespace tapestry
