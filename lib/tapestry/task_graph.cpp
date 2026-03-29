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

const char *placementToString(Placement p) {
  switch (p) {
  case Placement::LOCAL_SPM:
    return "LOCAL_SPM";
  case Placement::SHARED_L2:
    return "SHARED_L2";
  case Placement::EXTERNAL:
    return "EXTERNAL";
  case Placement::AUTO:
    return "AUTO";
  }
  return "AUTO";
}

Placement placementFromString(const std::string &s) {
  if (s == "LOCAL_SPM")
    return Placement::LOCAL_SPM;
  if (s == "SHARED_L2")
    return Placement::SHARED_L2;
  if (s == "EXTERNAL")
    return Placement::EXTERNAL;
  return Placement::AUTO;
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

  // Variant storage: per-kernel list of registered variants.
  std::vector<std::vector<VariantEntry>> variantMap;

  // Path contracts (latency bounds).
  std::vector<PathContract> pathContracts;

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
  std::string kname = info.name;
  impl_->nameIndex[kname] = idx;
  impl_->kernels.push_back(std::move(info));
  impl_->adj.emplace_back();        // empty successors list
  impl_->preds.emplace_back();      // empty predecessors list

  // Seed a default variant entry for this kernel.
  VariantEntry defaultVariant;
  defaultVariant.variantName = kname + "_default";
  defaultVariant.options = VariantOptions{/*unrollFactor=*/1, /*domainRank=*/0};
  impl_->variantMap.push_back({std::move(defaultVariant)});

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
    if (contract.dataVolume)
      std::cout << "  data_volume=" << *contract.dataVolume;
    if (contract.shape)
      std::cout << "  shape=" << *contract.shape;
    if (contract.placement)
      std::cout << "  placement=" << placementToString(*contract.placement);
    if (contract.throughput)
      std::cout << "  throughput=" << *contract.throughput;
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

EdgeHandle &EdgeHandle::data_volume(uint64_t vol) {
  graph_->contractRef(key_).dataVolume = vol;
  return *this;
}

EdgeHandle &EdgeHandle::shape(const std::string &shapeExpr) {
  graph_->contractRef(key_).shape = shapeExpr;
  return *this;
}

EdgeHandle &EdgeHandle::placement(Placement p) {
  graph_->contractRef(key_).placement = p;
  return *this;
}

EdgeHandle &EdgeHandle::throughput(const std::string &expr) {
  graph_->contractRef(key_).throughput = expr;
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
// Variant and path contract implementation
// ============================================================================

KernelHandle TaskGraph::addVariant(KernelHandle baseKernel,
                                   const std::string &variantName,
                                   VariantOptions opts) {
  // Return default (invalid) handle for null or foreign handles.
  if (!baseKernel.graph_ || baseKernel.graph_ != this)
    return KernelHandle();
  if (baseKernel.index_ >= impl_->kernels.size())
    return KernelHandle();

  // Reject duplicate variant names.
  for (const auto &existing : impl_->variantMap[baseKernel.index_]) {
    if (existing.variantName == variantName)
      return KernelHandle();
  }

  VariantEntry entry;
  entry.variantName = variantName;
  entry.options = opts;
  impl_->variantMap[baseKernel.index_].push_back(std::move(entry));

  // Create a new kernel node for the variant.
  KernelInfo info;
  info.name = variantName;
  info.provenance = impl_->kernels[baseKernel.index_].provenance;
  info.target = impl_->kernels[baseKernel.index_].target;
  return addKernelImpl(std::move(info));
}

static const std::vector<VariantEntry> emptyVariants;

const std::vector<VariantEntry> &
TaskGraph::variants(KernelHandle kernel) const {
  if (!kernel.graph_ || kernel.graph_ != this)
    return emptyVariants;
  if (kernel.index_ >= impl_->variantMap.size())
    return emptyVariants;
  return impl_->variantMap[kernel.index_];
}

const std::vector<VariantEntry> &
TaskGraph::variants(unsigned kernelIndex) const {
  if (kernelIndex >= impl_->variantMap.size())
    return emptyVariants;
  return impl_->variantMap[kernelIndex];
}

void TaskGraph::latencyBound(KernelHandle startKernel,
                             KernelHandle endKernel,
                             const std::string &latencyExpr) {
  assert(startKernel.graph_ == this &&
         "startKernel does not belong to this graph");
  assert(endKernel.graph_ == this &&
         "endKernel does not belong to this graph");

  PathContract pc;
  pc.startIdx = startKernel.index_;
  pc.endIdx = endKernel.index_;
  pc.latencyExpr = latencyExpr;
  impl_->pathContracts.push_back(std::move(pc));
}

const std::vector<PathContract> &TaskGraph::pathContracts() const {
  return impl_->pathContracts;
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
