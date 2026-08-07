#include "Mapping/IR/MappingAttrs.h"

#include "Common/Artifact.h"
#include "Common/ArtifactLocalReference.h"
#include "Dataflow/IR/DataflowCanonicalEntity.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/SystemMappingIdentity.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <limits>

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

LogicalResult mapping::SpatialConstraintProjectionKeyAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, std::uint32_t value) {
  if (!mapping::symbolizeSpatialConstraintProjection(value)) {
    emitError() << "unknown Spatial constraint projection ordinal " << value;
    return failure();
  }
  return success();
}

LogicalResult mapping::SystemConstraintProjectionKeyAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, std::uint32_t value) {
  if (!mapping::symbolizeSystemConstraintProjection(value)) {
    emitError() << "unknown System constraint projection ordinal " << value;
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

LogicalResult mapping::RootedGraphLaunchRefAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, DenseI8ArrayAttr record) {
  return verifyDataflowRef<::dataflow::RootedGraphLaunchRef>(emitError, record);
}

LogicalResult mapping::RootThreadLaunchRefAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, DenseI8ArrayAttr record) {
  return verifyDataflowRef<::dataflow::RootThreadLaunchRef>(emitError, record);
}

LogicalResult mapping::SystemServiceObligationKeyAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, DenseI8ArrayAttr record) {
  auto decoded = ::loom::mapping::decodeSystemServiceObligationKey(
      unsignedBytes(record), dummyIdentity());
  if (!decoded) {
    emitError() << llvm::toString(decoded.takeError());
    return failure();
  }
  return success();
}

LogicalResult mapping::ServicePlanSelectionKeyAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, DenseI8ArrayAttr record) {
  auto decoded = ::loom::mapping::decodeServicePlanSelectionKey(
      unsignedBytes(record), dummyIdentity());
  if (!decoded) {
    emitError() << llvm::toString(decoded.takeError());
    return failure();
  }
  return success();
}

LogicalResult mapping::CanonicalServiceLegKeyAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, DenseI8ArrayAttr record) {
  auto decoded = ::loom::mapping::decodeCanonicalServiceLegKey(
      unsignedBytes(record), dummyIdentity());
  if (!decoded) {
    emitError() << llvm::toString(decoded.takeError());
    return failure();
  }
  return success();
}

LogicalResult mapping::SystemTransferTerminalKeyAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, DenseI8ArrayAttr record) {
  auto decoded = ::loom::mapping::decodeSystemTransferTerminalKey(
      unsignedBytes(record), dummyIdentity());
  if (!decoded) {
    emitError() << llvm::toString(decoded.takeError());
    return failure();
  }
  return success();
}

LogicalResult mapping::SystemPresburgerCellAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, std::uint32_t dimensionCount,
    std::uint32_t symbolCount, std::uint32_t localCount, ArrayAttr equalities,
    ArrayAttr inequalities) {
  const std::uint64_t width =
      static_cast<std::uint64_t>(dimensionCount) + symbolCount + localCount + 1;
  if (width > std::numeric_limits<std::size_t>::max()) {
    emitError() << "Presburger row width exceeds native size";
    return failure();
  }
  const auto verifyRows = [&](ArrayAttr rows,
                              llvm::StringRef kind) -> LogicalResult {
    for (Attribute attribute : rows) {
      auto row = dyn_cast<DenseI64ArrayAttr>(attribute);
      if (!row) {
        emitError() << kind << " rows must be dense i64 arrays";
        return failure();
      }
      if (static_cast<std::uint64_t>(row.size()) != width) {
        emitError() << kind << " row has width " << row.size()
                    << " but expected " << width;
        return failure();
      }
    }
    return success();
  };
  if (failed(verifyRows(equalities, "equality")) ||
      failed(verifyRows(inequalities, "inequality")))
    return failure();
  return success();
}

LogicalResult mapping::ArtifactRootReferenceAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, DenseI8ArrayAttr record) {
  std::vector<std::uint8_t> bytes = unsignedBytes(record);
  auto decoded = ::loom::decodeArtifactRootReferencePrefix(bytes);
  if (!decoded) {
    emitError() << llvm::toString(decoded.takeError());
    return failure();
  }
  if (decoded->byteCount != bytes.size() ||
      ::loom::encodeArtifactRootReference(decoded->reference) != bytes) {
    emitError() << "ArtifactRootReference payload is not canonical";
    return failure();
  }
  return success();
}

LogicalResult mapping::LogicalMemoryRootRefAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, DenseI8ArrayAttr record) {
  return verifyDataflowRef<::dataflow::LogicalMemoryRootRef>(emitError, record);
}

LogicalResult mapping::LogicalMemoryRootOrViewRefAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, DenseI8ArrayAttr record) {
  return verifyDataflowRef<::dataflow::LogicalMemoryRootOrViewRef>(emitError,
                                                                   record);
}

LogicalResult mapping::MemoryExposureRefAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, DenseI8ArrayAttr record) {
  return verifyDataflowRef<::dataflow::MemoryExposureRef>(emitError, record);
}

LogicalResult mapping::FenceActorFamilyRefAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, DenseI8ArrayAttr record) {
  return verifyDataflowRef<::dataflow::FenceActorFamilyRef>(emitError, record);
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

#define LOOM_VERIFY_FABRIC_CONSTRAINT_REF(Name, Ref)                           \
  LogicalResult mapping::Name##Attr::verify(                                   \
      function_ref<InFlightDiagnostic()> emitError, DenseI8ArrayAttr record) { \
    return verifyFabricRef<::loom::fabric::Ref>(emitError, record);            \
  }

LOOM_VERIFY_FABRIC_CONSTRAINT_REF(FabricFuOccurrenceRef, FabricFuOccurrenceRef)
LOOM_VERIFY_FABRIC_CONSTRAINT_REF(FabricAccCoreOccurrenceRef,
                                  AccCoreOccurrenceRef)
LOOM_VERIFY_FABRIC_CONSTRAINT_REF(FabricSpatialCoreOccurrenceRef,
                                  SpatialCoreOccurrenceRef)
LOOM_VERIFY_FABRIC_CONSTRAINT_REF(FabricPeOccurrenceRef, FabricPeOccurrenceRef)
LOOM_VERIFY_FABRIC_CONSTRAINT_REF(InstructionContextRef, InstructionContextRef)
LOOM_VERIFY_FABRIC_CONSTRAINT_REF(FabricUsePatternRef, FabricUsePatternRef)
LOOM_VERIFY_FABRIC_CONSTRAINT_REF(FabricPhysicalRefinementDomainRef,
                                  FabricPhysicalRefinementDomainRef)
LOOM_VERIFY_FABRIC_CONSTRAINT_REF(FabricMemoryOccurrenceRef,
                                  FabricMemoryOccurrenceRef)
LOOM_VERIFY_FABRIC_CONSTRAINT_REF(FabricPhysicalTraversalRef,
                                  FabricPhysicalTraversalRef)
LOOM_VERIFY_FABRIC_CONSTRAINT_REF(FabricResourceStateRef,
                                  FabricResourceStateRef)
LOOM_VERIFY_FABRIC_CONSTRAINT_REF(FabricTransportEndpointRef,
                                  FabricTransportEndpointRef)
LOOM_VERIFY_FABRIC_CONSTRAINT_REF(FabricMemoryOperationPortRef,
                                  FabricMemoryOperationPortRef)
LOOM_VERIFY_FABRIC_CONSTRAINT_REF(FabricMemoryOperationContextRef,
                                  FabricMemoryOperationContextRef)
LOOM_VERIFY_FABRIC_CONSTRAINT_REF(FabricMemoryServiceRef,
                                  FabricMemoryServiceRef)
LOOM_VERIFY_FABRIC_CONSTRAINT_REF(FabricMemoryServiceRegionRef,
                                  FabricMemoryServiceRegionRef)
LOOM_VERIFY_FABRIC_CONSTRAINT_REF(LocalMemoryServiceRef, LocalMemoryServiceRef)
LOOM_VERIFY_FABRIC_CONSTRAINT_REF(ManagerEndpointRef, ManagerEndpointRef)
LOOM_VERIFY_FABRIC_CONSTRAINT_REF(SubordinateEndpointRef,
                                  SubordinateEndpointRef)
LOOM_VERIFY_FABRIC_CONSTRAINT_REF(MemoryConsistencyDomainRef,
                                  MemoryConsistencyDomainRef)
LOOM_VERIFY_FABRIC_CONSTRAINT_REF(SystemServiceTransformRef,
                                  SystemServiceTransformRef)

#undef LOOM_VERIFY_FABRIC_CONSTRAINT_REF

LogicalResult mapping::MemoryByteRangeAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, std::uint64_t offsetBytes,
    std::uint64_t sizeBytes) {
  if (sizeBytes == 0) {
    emitError() << "memory byte range size must be positive";
    return failure();
  }
  if (offsetBytes > std::numeric_limits<std::uint64_t>::max() - sizeBytes) {
    emitError() << "memory byte range end overflows u64";
    return failure();
  }
  return success();
}

LogicalResult mapping::OwnerTypedValueAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, DenseI8ArrayAttr record) {
  // The exact referenced owner supplies the only codec and canonicality rule.
  // This carrier has no context-free semantic invariant of its own.
  (void)emitError;
  (void)record;
  return success();
}

LogicalResult mapping::SpatialEventPointAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, Attribute event,
    mapping::OwnerTypedValueAttr guaranteedOffset) {
  if (!isa<mapping::ActorTransitionEventAttr,
           mapping::GraphProducerEndpointRefAttr,
           mapping::GraphConsumerEndpointRefAttr>(event)) {
    emitError() << "event must be a closed Spatial activity-event reference";
    return failure();
  }
  (void)guaranteedOffset;
  return success();
}

LogicalResult mapping::SpatialTransferTerminalAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    GraphProducerEndpointRefAttr producer,
    GraphConsumerEndpointRefAttr consumer) {
  if (!producer) {
    emitError() << "spatial transfer terminal requires a producer";
    return failure();
  }
  return success();
}

LogicalResult mapping::ConstraintUnsignedIntervalAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, IntegerAttr lower,
    IntegerAttr upper) {
  auto lowerType = dyn_cast<IntegerType>(lower.getType());
  auto upperType = dyn_cast<IntegerType>(upper.getType());
  if (!lowerType || !upperType || !lowerType.isUnsigned() ||
      !upperType.isUnsigned()) {
    emitError() << "constraint interval bounds must use unsigned integer types";
    return failure();
  }
  const unsigned width = std::max(lowerType.getWidth(), upperType.getWidth());
  if (!lower.getValue().zext(width).ult(upper.getValue().zext(width))) {
    emitError() << "constraint interval must be non-empty";
    return failure();
  }
  return success();
}

LogicalResult mapping::ConstraintAddressRegionAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    FabricMemoryServiceRefAttr service, ArrayAttr intervals) {
  if (!service) {
    emitError() << "constraint address region requires a memory service";
    return failure();
  }
  if (intervals.empty()) {
    emitError()
        << "constraint address region requires a non-empty interval set";
    return failure();
  }
  for (Attribute interval : intervals) {
    if (!isa<ConstraintUnsignedIntervalAttr>(interval)) {
      emitError() << "constraint address region contains a non-interval value";
      return failure();
    }
  }
  return success();
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
