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
    for (std::int32_t port : actor.getOperandPorts())
      appendU64(key, static_cast<std::uint32_t>(port));
    key.push_back(0xff);
    for (std::int32_t port : actor.getResultPorts())
      appendU64(key, static_cast<std::uint32_t>(port));
    return key;
  }

  auto boundary = cast<::mapping::ComputeBoundaryOp>(operation);
  key.push_back(1);
  appendAttrBytes(key, boundary.getActor());
  key.push_back(static_cast<std::uint8_t>(boundary.getDirection()));
  appendU64(key, boundary.getPortOrdinal());
  appendAttrBytes(key, boundary.getFuPort());
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

void canonicalizeTech(::mapping::TechOp root) {
  SmallVector<Attribute> covers(root.getCovers().begin(),
                                root.getCovers().end());
  llvm::sort(covers, [](Attribute left, Attribute right) {
    return bytes(cast<::mapping::GraphRefAttr>(left).getRecord()) <
           bytes(cast<::mapping::GraphRefAttr>(right).getRecord());
  });
  root.setCoversAttr(ArrayAttr::get(root.getContext(), covers));

  Block &body = root.getBody().front();
  std::vector<::mapping::ComputeRealizationOp> realizations;
  for (auto realization : body.getOps<::mapping::ComputeRealizationOp>()) {
    canonicalizeChildren(realization);
    realizations.push_back(realization);
  }
  llvm::sort(realizations, [](auto left, auto right) {
    return realizationKey(left) < realizationKey(right);
  });
  Builder builder(root.getContext());
  for (auto [index, realization] : llvm::enumerate(realizations)) {
    realization.setEntityIdAttr(builder.getI64IntegerAttr(index));
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
