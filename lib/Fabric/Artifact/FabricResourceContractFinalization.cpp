#include "FabricResourceContractFinalization.h"

#include "FabricOperationTransport.h"

#include "Fabric/IR/BoundaryTransfer.h"
#include "Fabric/IR/FabricTypes.h"
#include "Fabric/IR/FifoResourceContract.h"
#include "Fabric/IR/MemoryCapabilityFinalization.h"
#include "Fabric/IR/PhysicalTagResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "Fabric/IR/SwitchResourceContract.h"
#include "Fabric/IR/TemporalPeResourceContract.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Visitors.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

using namespace mlir;

namespace loom::fabric::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_artifact_invalid: " + message);
}

llvm::Expected<std::optional<::fabric::ResourceContract>>
deriveBaseResourceContract(Operation *operation,
                           const FabricCanonicalLabeling &labeling) {
  if (auto fifo = dyn_cast<::fabric::FifoOp>(operation)) {
    const ::fabric::FifoQueueDiscipline discipline =
        fifo.getQueueDiscipline().value_or(
            ::fabric::FifoQueueDiscipline::StrictFifo);
    std::uint32_t tagWidthBits = 0;
    if (auto tagged =
            dyn_cast<::fabric::BitsTagType>(fifo.getOutput().getType()))
      tagWidthBits = tagged.getTagWidth();
    auto contract = ::fabric::createFifoResourceContract(
        static_cast<std::uint32_t>(fifo.getMaxDepth()), fifo.getBypassable(),
        discipline, tagWidthBits);
    if (!contract)
      return contract.takeError();
    return std::optional<::fabric::ResourceContract>(std::move(*contract));
  }
  if (isa<::fabric::BoundaryOp>(operation)) {
    auto contract = ::fabric::ResourceContract::create(
        ::fabric::declareBoundaryTransferContract());
    if (!contract)
      return contract.takeError();
    return std::optional<::fabric::ResourceContract>(std::move(*contract));
  }
  if (isa<::fabric::OpOp>(operation)) {
    auto record = operation->getAttrOfType<DenseI8ArrayAttr>(
        ::fabric::kResourceContractRecordAttrName);
    if (!record)
      return invalid("fabric.op is missing its complete resource contract");
    std::vector<std::uint8_t> bytes;
    bytes.reserve(record.size());
    for (std::int8_t byte : record.asArrayRef())
      bytes.push_back(static_cast<std::uint8_t>(byte));
    auto contract = ::fabric::decodeResourceContractRecord(bytes);
    if (!contract)
      return contract.takeError();
    if (contract->usePatternCount() == 0)
      return invalid("fabric.op resource contract has no use pattern");
    return std::optional<::fabric::ResourceContract>(std::move(*contract));
  }
  if (auto memory = dyn_cast<::fabric::MemOp>(operation)) {
    if (llvm::Error error = ::fabric::validateMemoryCapabilityFinalization(
            memory.getMemoryContract(), memory.getMemoryOperationPortsAttr()))
      return std::move(error);
    return std::optional<::fabric::ResourceContract>();
  }
  if (auto pe = dyn_cast<::fabric::PeOp>(operation);
      pe && pe.getSchedule() == ::fabric::Schedule::Temporal) {
    std::optional<std::uint64_t> peId;
    for (const FabricEntityCarrier &carrier : labeling.carriers)
      if (carrier.op == operation) {
        if (carrier.kind != FabricEntityKind::FabricPeOccurrence)
          return invalid(
              "temporal fabric.pe has the wrong canonical entity kind");
        peId = carrier.id;
        break;
      }
    if (!peId)
      return invalid("temporal fabric.pe has no canonical occurrence");

    llvm::SmallVector<std::uint32_t, 8> fuInputCounts;
    for (Operation *candidate : labeling.canonicalOperationOrder) {
      auto fu = dyn_cast_or_null<::fabric::FuOp>(candidate);
      if (!fu || fu->getParentOp() != operation)
        continue;
      if (fu.getInputs().size() > std::numeric_limits<std::uint32_t>::max())
        return invalid("temporal fabric.pe FU input domain exceeds u32");
      fuInputCounts.push_back(
          static_cast<std::uint32_t>(fu.getInputs().size()));
    }

    auto contextCount = pe.getNumInstruction();
    auto mode = pe.getOperandBufferMode();
    auto entries = pe.getOperandBufferSize();
    if (!contextCount || !mode || !entries)
      return invalid("temporal fabric.pe lacks its verified buffer parameters");
    auto derived = ::fabric::TemporalPeResourceContract::create(
        ::fabric::TemporalPeResourceDeclaration{
            FabricPeOccurrenceRef(*peId), *contextCount, fuInputCounts, *mode,
            *entries, pe.getNumRegFifo().value_or(0),
            pe.getRegFifoDepth().value_or(0),
            pe.getRegFifoPorts().value_or(1)});
    if (!derived)
      return derived.takeError();
    return std::optional<::fabric::ResourceContract>(
        derived->resourceContract());
  }
  if (auto sw = dyn_cast<::fabric::SwitchOp>(operation)) {
    auto derived = ::fabric::deriveSwitchResourceContract(sw);
    if (!derived)
      return derived.takeError();
    return std::optional<::fabric::ResourceContract>(
        derived->resourceContract());
  }
  return std::optional<::fabric::ResourceContract>();
}

llvm::Expected<std::vector<std::uint32_t>>
derivePhysicalTagAssignmentWidths(Operation *operation) {
  const bool isIngressOwner =
      isa<::fabric::PeOp, ::fabric::MemOp, ::fabric::SwitchOp, ::fabric::FifoOp,
          ::fabric::BoundaryOp>(operation);
  if (!isIngressOwner)
    return std::vector<std::uint32_t>();

  auto types = resolveFabricOperationTransportTypes(operation);
  if (!types)
    return types.takeError();
  std::vector<std::uint32_t> widths;
  for (Type input : types->inputs)
    if (auto tagged = dyn_cast<::fabric::BitsTagType>(input))
      widths.push_back(tagged.getTagWidth());

  bool writesTag = isa<::fabric::BoundaryOp>(operation);
  if (auto pe = dyn_cast<::fabric::PeOp>(operation))
    writesTag = pe.getSchedule() == ::fabric::Schedule::Temporal;
  if (auto memory = dyn_cast<::fabric::MemOp>(operation))
    if (const ::fabric::MemoryEngineAttr engine =
            memory.getMemoryContract().getEngine())
      writesTag = engine.getSchedule() == ::fabric::Schedule::Temporal;
  if (writesTag)
    for (Type output : types->outputs)
      if (auto tagged = dyn_cast<::fabric::BitsTagType>(output))
        widths.push_back(tagged.getTagWidth());
  return widths;
}

llvm::Expected<std::optional<::fabric::ResourceContract>>
deriveResourceContract(Operation *operation,
                       const FabricCanonicalLabeling &labeling) {
  auto base = deriveBaseResourceContract(operation, labeling);
  if (!base)
    return base.takeError();
  auto tagWidths = derivePhysicalTagAssignmentWidths(operation);
  if (!tagWidths)
    return tagWidths.takeError();
  if (tagWidths->empty())
    return base;

  const ::fabric::ResourceContract *baseContract = *base ? &**base : nullptr;
  auto extended =
      ::fabric::appendPhysicalTagAssignmentPatterns(baseContract, *tagWidths);
  if (!extended)
    return extended.takeError();
  return std::optional<::fabric::ResourceContract>(std::move(*extended));
}

std::vector<std::int8_t> signedBytes(llvm::ArrayRef<std::uint8_t> bytes) {
  std::vector<std::int8_t> result;
  result.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    result.push_back(static_cast<std::int8_t>(byte));
  return result;
}

std::vector<std::uint8_t> unsignedBytes(llvm::ArrayRef<std::int8_t> bytes) {
  std::vector<std::uint8_t> result;
  result.reserve(bytes.size());
  for (std::int8_t byte : bytes)
    result.push_back(static_cast<std::uint8_t>(byte));
  return result;
}

} // namespace

llvm::Expected<std::optional<::fabric::ResourceContract>>
validateFabricResourceContract(Operation *operation,
                               const FabricCanonicalLabeling &labeling) {
  auto expected = deriveResourceContract(operation, labeling);
  if (!expected)
    return expected.takeError();

  auto record = operation->getAttrOfType<DenseI8ArrayAttr>(
      ::fabric::kResourceContractRecordAttrName);
  if (!*expected) {
    if (record)
      return invalid("an owner without a resource contract carries a record");
    return std::optional<::fabric::ResourceContract>();
  }
  if (!record)
    return invalid("a resource owner is missing its complete contract record");

  std::vector<std::uint8_t> bytes = unsignedBytes(record.asArrayRef());
  auto decoded = ::fabric::decodeResourceContractRecord(bytes);
  if (!decoded)
    return decoded.takeError();
  auto canonical = ::fabric::encodeResourceContractRecord(*decoded);
  if (!canonical)
    return canonical.takeError();
  if (*canonical != bytes)
    return invalid("a resource contract record is not canonical");
  auto expectedBytes = ::fabric::encodeResourceContractRecord(**expected);
  if (!expectedBytes)
    return expectedBytes.takeError();
  if (*expectedBytes != bytes)
    return invalid("a resource contract record disagrees with its owner");
  return std::optional<::fabric::ResourceContract>(std::move(*decoded));
}

llvm::Error
materializeFabricResourceContracts(::fabric::ModuleOp root,
                                   const FabricCanonicalLabeling &labeling) {
  llvm::Error result = llvm::Error::success();
  root->walk([&](Operation *operation) {
    if (result)
      return WalkResult::interrupt();
    auto contract = deriveResourceContract(operation, labeling);
    if (!contract) {
      result = contract.takeError();
      return WalkResult::interrupt();
    }
    operation->removeAttr(::fabric::kResourceContractRecordAttrName);
    if (!*contract)
      return WalkResult::advance();
    auto bytes = ::fabric::encodeResourceContractRecord(**contract);
    if (!bytes) {
      result = bytes.takeError();
      return WalkResult::interrupt();
    }
    operation->setAttr(
        ::fabric::kResourceContractRecordAttrName,
        DenseI8ArrayAttr::get(root.getContext(), signedBytes(*bytes)));
    return WalkResult::advance();
  });
  return result;
}

llvm::Error
validateFabricResourceContracts(::fabric::ModuleOp root,
                                const FabricCanonicalLabeling &labeling) {
  llvm::Error result = llvm::Error::success();
  root->walk([&](Operation *operation) {
    if (result)
      return WalkResult::interrupt();
    auto contract = validateFabricResourceContract(operation, labeling);
    if (!contract) {
      result = contract.takeError();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result;
}

} // namespace loom::fabric::detail
