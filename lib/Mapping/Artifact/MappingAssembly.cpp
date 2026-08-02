#include "Mapping/Artifact/MappingArtifact.h"

#include "Mapping/IR/MappingOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

using namespace mlir;

namespace loom::mapping {
namespace {

std::vector<std::uint8_t> bytes(DenseI8ArrayAttr attribute) {
  std::vector<std::uint8_t> result;
  result.reserve(attribute.size());
  for (std::int8_t byte : attribute.asArrayRef())
    result.push_back(static_cast<std::uint8_t>(byte));
  return result;
}

template <typename Attr>
void appendAttrBytes(std::vector<std::uint8_t> &result, Attr attribute) {
  std::vector<std::uint8_t> value = bytes(attribute.getRecord());
  result.insert(result.end(), value.begin(), value.end());
}

void appendAttributeKey(std::vector<std::uint8_t> &result,
                        Attribute attribute) {
  if (auto producer =
          dyn_cast<::mapping::GraphProducerEndpointRefAttr>(attribute)) {
    result.push_back(0);
    appendAttrBytes(result, producer);
    return;
  }
  if (auto consumer =
          dyn_cast<::mapping::GraphConsumerEndpointRefAttr>(attribute)) {
    result.push_back(1);
    appendAttrBytes(result, consumer);
    return;
  }
  appendAttrBytes(
      result,
      cast<::mapping::FabricMemoryEngineTemplateEndpointRefAttr>(attribute));
}

void appendU64(std::vector<std::uint8_t> &result, std::uint64_t value) {
  for (unsigned byte = 0; byte < 8; ++byte)
    result.push_back(static_cast<std::uint8_t>(value >> (8 * (7 - byte))));
}

std::vector<std::uint8_t> childKey(Operation &operation) {
  std::vector<std::uint8_t> key;
  if (auto actor = dyn_cast<::mapping::ComputeActorOp>(operation)) {
    key.push_back(0);
    appendAttrBytes(key, actor.getActor());
    appendAttrBytes(key, actor.getFabricOp());
    appendU64(key, actor.getOperandPorts().size());
    for (std::int64_t port : actor.getOperandPorts())
      appendU64(key, static_cast<std::uint64_t>(port));
    appendU64(key, actor.getResultPorts().size());
    for (std::int64_t port : actor.getResultPorts())
      appendU64(key, static_cast<std::uint64_t>(port));
    return key;
  }

  if (auto boundary = dyn_cast<::mapping::ComputeBoundaryOp>(operation)) {
    key.push_back(1);
    appendAttrBytes(key, boundary.getActor());
    key.push_back(static_cast<std::uint8_t>(boundary.getDirection()));
    appendU64(key, boundary.getPortOrdinal());
    appendAttrBytes(key, boundary.getFuPort());
    return key;
  }
  if (auto actor = dyn_cast<::mapping::MemoryActorOp>(operation)) {
    key.push_back(2);
    appendAttrBytes(key, actor.getActor());
    appendAttrBytes(key, actor.getOperationPort());
    appendAttrBytes(key, actor.getCapability());
    appendU64(key, actor.getOperandPorts().size());
    for (Attribute endpoint : actor.getOperandPorts())
      appendAttributeKey(key, endpoint);
    appendU64(key, actor.getResultPorts().size());
    for (Attribute endpoint : actor.getResultPorts())
      appendAttributeKey(key, endpoint);
    return key;
  }
  if (auto boundary = dyn_cast<::mapping::MemoryGraphBoundaryOp>(operation)) {
    key.push_back(3);
    appendAttributeKey(key, boundary.getTerminal());
    appendAttrBytes(key, boundary.getEndpoint());
    return key;
  }
  auto edge = cast<::mapping::MemoryInternalEdgeOp>(operation);
  key.push_back(4);
  appendAttrBytes(key, edge.getProducer());
  appendAttrBytes(key, edge.getConsumer());
  appendAttrBytes(key, edge.getConnection());
  return key;
}

void canonicalizeChildren(::mapping::ComputeRealizationOp realization) {
  Block &block = realization.getBody().front();
  std::vector<Operation *> children;
  for (Operation &operation : block)
    children.push_back(&operation);
  llvm::sort(children, [](Operation *left, Operation *right) {
    return childKey(*left) < childKey(*right);
  });
  for (Operation *operation : children)
    operation->moveBefore(&block, block.end());
}

void canonicalizeChildren(::mapping::MemoryRealizationOp realization) {
  Block &block = realization.getBody().front();
  std::vector<Operation *> children;
  for (Operation &operation : block)
    children.push_back(&operation);
  llvm::sort(children, [](Operation *left, Operation *right) {
    return childKey(*left) < childKey(*right);
  });
  for (Operation *operation : children)
    operation->moveBefore(&block, block.end());
}

std::vector<std::uint8_t>
realizationKey(::mapping::ComputeRealizationOp realization) {
  std::vector<std::uint8_t> key;
  appendAttrBytes(key, realization.getCapabilityTemplate());
  for (Operation &child : realization.getBody().front()) {
    std::vector<std::uint8_t> childBytes = childKey(child);
    appendU64(key, childBytes.size());
    key.insert(key.end(), childBytes.begin(), childBytes.end());
  }
  return key;
}

std::vector<std::uint8_t>
realizationKey(::mapping::MemoryRealizationOp realization) {
  std::vector<std::uint8_t> key;
  appendAttrBytes(key, realization.getEngine());
  for (Operation &child : realization.getBody().front()) {
    std::vector<std::uint8_t> childBytes = childKey(child);
    appendU64(key, childBytes.size());
    key.insert(key.end(), childBytes.begin(), childBytes.end());
  }
  return key;
}

void canonicalizeTech(::mapping::TechOp root) {
  SmallVector<Attribute> covers(root.getCovers().begin(),
                                root.getCovers().end());
  llvm::sort(covers, [](Attribute left, Attribute right) {
    return bytes(cast<::mapping::GraphRefAttr>(left).getRecord()) <
           bytes(cast<::mapping::GraphRefAttr>(right).getRecord());
  });
  root.setCoversAttr(ArrayAttr::get(root.getContext(), covers));

  Block &body = root.getBody().front();
  std::vector<::mapping::ComputeRealizationOp> computeRealizations;
  for (auto realization : body.getOps<::mapping::ComputeRealizationOp>()) {
    canonicalizeChildren(realization);
    computeRealizations.push_back(realization);
  }
  llvm::sort(computeRealizations, [](auto left, auto right) {
    return realizationKey(left) < realizationKey(right);
  });

  std::vector<::mapping::MemoryRealizationOp> memoryRealizations;
  for (auto realization : body.getOps<::mapping::MemoryRealizationOp>()) {
    canonicalizeChildren(realization);
    memoryRealizations.push_back(realization);
  }
  llvm::sort(memoryRealizations, [](auto left, auto right) {
    return realizationKey(left) < realizationKey(right);
  });

  Builder builder(root.getContext());
  std::uint64_t entityId = 0;
  for (auto realization : computeRealizations) {
    realization.setEntityIdAttr(builder.getI64IntegerAttr(entityId++));
    realization->moveBefore(&body, body.end());
  }
  for (auto realization : memoryRealizations) {
    realization.setEntityIdAttr(builder.getI64IntegerAttr(entityId++));
    realization->moveBefore(&body, body.end());
  }
}

} // namespace

llvm::Expected<CanonicalSemanticBytes>
writeCanonicalMappingAssembly(::mapping::TechOp root) {
  OwningOpRef<Operation *> clone(root->clone());
  auto canonical = cast<::mapping::TechOp>(clone.get());
  canonicalizeTech(canonical);
  if (failed(verify(canonical)))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "mapping artifact is structurally invalid");

  std::string text;
  llvm::raw_string_ostream stream(text);
  canonical.print(stream, OpPrintingFlags().enableDebugInfo(false));
  stream << '\n';
  stream.flush();
  return CanonicalSemanticBytes(
      std::vector<std::uint8_t>(text.begin(), text.end()));
}

} // namespace loom::mapping
