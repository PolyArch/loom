#include "Mapping/IR/MappingAttrs.h"

#include "Common/Artifact.h"
#include "Dataflow/IR/DataflowCanonicalEntity.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>

using namespace mlir;

namespace {

std::vector<std::uint8_t> unsignedBytes(DenseI8ArrayAttr record) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(record.size());
  for (std::int8_t byte : record.asArrayRef())
    bytes.push_back(static_cast<std::uint8_t>(byte));
  return bytes;
}

::loom::ArtifactIdentity dummyIdentity() {
  std::array<std::uint8_t, ::loom::ArtifactIdentity::byteSize> bytes{};
  auto identity = ::loom::ArtifactIdentity::fromBytes(bytes);
  if (!identity) {
    llvm::consumeError(identity.takeError());
    llvm_unreachable("a 32-byte ArtifactIdentity must decode");
  }
  return std::move(*identity);
}

template <typename Ref>
LogicalResult verifyDataflowRef(function_ref<InFlightDiagnostic()> emitError,
                                DenseI8ArrayAttr record) {
  std::vector<std::uint8_t> bytes = unsignedBytes(record);
  auto decoded =
      ::dataflow::decodeDataflowReference<Ref>(bytes, dummyIdentity());
  if (!decoded) {
    emitError() << llvm::toString(decoded.takeError());
    return failure();
  }
  auto canonical =
      ::dataflow::encodeDataflowReference(dummyIdentity(), *decoded);
  if (!canonical) {
    emitError() << llvm::toString(canonical.takeError());
    return failure();
  }
  if (*canonical != bytes) {
    emitError() << "reference payload is not canonical";
    return failure();
  }
  return success();
}

template <typename Ref>
LogicalResult verifyFabricRef(function_ref<InFlightDiagnostic()> emitError,
                              DenseI8ArrayAttr record) {
  std::vector<std::uint8_t> bytes = unsignedBytes(record);
  auto decoded = ::loom::fabric::decodeFabricRef<Ref>(bytes);
  if (!decoded) {
    emitError() << llvm::toString(decoded.takeError());
    return failure();
  }
  if (::loom::fabric::canonicalFabricBytes(*decoded) != bytes) {
    emitError() << "reference payload is not canonical";
    return failure();
  }
  return success();
}

} // namespace

LogicalResult mapping::ArtifactIdentityAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, DenseI8ArrayAttr record) {
  if (record.size() != ::loom::ArtifactIdentity::byteSize) {
    emitError() << "ArtifactIdentity must contain exactly 32 bytes";
    return failure();
  }
  return success();
}

LogicalResult
mapping::GraphRefAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                              DenseI8ArrayAttr record) {
  return verifyDataflowRef<::dataflow::GraphRef>(emitError, record);
}

LogicalResult
mapping::ActorRefAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                              DenseI8ArrayAttr record) {
  return verifyDataflowRef<::dataflow::ActorRef>(emitError, record);
}

LogicalResult mapping::GraphProducerEndpointRefAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, DenseI8ArrayAttr record) {
  return verifyDataflowRef<::dataflow::CanonicalGraphProducerEndpointRef>(
      emitError, record);
}

LogicalResult mapping::GraphConsumerEndpointRefAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, DenseI8ArrayAttr record) {
  return verifyDataflowRef<::dataflow::CanonicalGraphConsumerEndpointRef>(
      emitError, record);
}

LogicalResult mapping::FabricFuCapabilityTemplateRefAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, DenseI8ArrayAttr record) {
  return verifyFabricRef<::loom::fabric::FabricFuCapabilityTemplateRef>(
      emitError, record);
}

LogicalResult mapping::FabricFuTemplateNodeRefAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, DenseI8ArrayAttr record) {
  return verifyFabricRef<::loom::fabric::FabricFuTemplateNodeRef>(emitError,
                                                                  record);
}

LogicalResult mapping::FabricFuTemplatePortRefAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, DenseI8ArrayAttr record) {
  return verifyFabricRef<::loom::fabric::FabricFuTemplatePortRef>(emitError,
                                                                  record);
}

LogicalResult mapping::FabricMemoryEngineTemplateRefAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, DenseI8ArrayAttr record) {
  return verifyFabricRef<::loom::fabric::FabricMemoryEngineTemplateRef>(
      emitError, record);
}

LogicalResult mapping::FabricMemoryEngineTemplateOperationPortRefAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, DenseI8ArrayAttr record) {
  return verifyFabricRef<
      ::loom::fabric::FabricMemoryEngineTemplateOperationPortRef>(emitError,
                                                                  record);
}

LogicalResult
mapping::FabricMemoryEngineTemplateCapabilityAlternativeRefAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, DenseI8ArrayAttr record) {
  return verifyFabricRef<
      ::loom::fabric::FabricMemoryEngineTemplateCapabilityAlternativeRef>(
      emitError, record);
}

LogicalResult mapping::FabricMemoryEngineTemplateEndpointRefAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, DenseI8ArrayAttr record) {
  return verifyFabricRef<::loom::fabric::FabricMemoryEngineTemplateEndpointRef>(
      emitError, record);
}

LogicalResult
mapping::FabricMemoryEngineTemplateInternalConnectionRefAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, DenseI8ArrayAttr record) {
  return verifyFabricRef<
      ::loom::fabric::FabricMemoryEngineTemplateInternalConnectionRef>(
      emitError, record);
}
