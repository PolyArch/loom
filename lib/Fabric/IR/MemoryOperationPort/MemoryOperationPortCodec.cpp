#include "Fabric/IR/MemoryOperationPort.h"

#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "MemoryOperationPortInternal.h"

#include <limits>
#include <system_error>

namespace fabric {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(std::errc::invalid_argument, "%s",
                                 message.str().c_str());
}

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

llvm::Error appendCount(std::vector<std::uint8_t> &bytes, std::size_t count,
                        llvm::StringRef field) {
  if (count > std::numeric_limits<std::uint64_t>::max())
    return invalid(field + " count exceeds u64");
  appendU64(bytes, static_cast<std::uint64_t>(count));
  return llvm::Error::success();
}

llvm::Error appendFrame(std::vector<std::uint8_t> &bytes,
                        llvm::ArrayRef<std::uint8_t> field,
                        llvm::StringRef name) {
  if (llvm::Error error = appendCount(bytes, field.size(), name))
    return error;
  bytes.insert(bytes.end(), field.begin(), field.end());
  return llvm::Error::success();
}

class Reader {
public:
  explicit Reader(llvm::ArrayRef<std::uint8_t> bytes) : bytes_(bytes) {}

  llvm::Expected<std::uint32_t> readU32(const llvm::Twine &field) {
    if (remaining() < 4)
      return invalid(field + " is truncated");
    std::uint32_t value = 0;
    for (unsigned index = 0; index < 4; ++index)
      value = (value << 8) | bytes_[offset_++];
    return value;
  }

  llvm::Expected<std::uint64_t> readU64(const llvm::Twine &field) {
    if (remaining() < 8)
      return invalid(field + " is truncated");
    std::uint64_t value = 0;
    for (unsigned index = 0; index < 8; ++index)
      value = (value << 8) | bytes_[offset_++];
    return value;
  }

  llvm::Expected<llvm::ArrayRef<std::uint8_t>>
  readFrame(const llvm::Twine &field) {
    auto size = readU64(field + " length");
    if (!size)
      return size.takeError();
    if (*size > remaining())
      return invalid(field + " is truncated");
    llvm::ArrayRef<std::uint8_t> result = bytes_.slice(offset_, *size);
    offset_ += *size;
    return result;
  }

  llvm::Error finish(const llvm::Twine &record) const {
    if (remaining() != 0)
      return invalid(record + " has trailing bytes");
    return llvm::Error::success();
  }

  std::size_t remaining() const { return bytes_.size() - offset_; }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
  std::size_t offset_ = 0;
};

llvm::Expected<MemoryCapabilityAlternativeRecord>
decodeAlternative(llvm::ArrayRef<std::uint8_t> bytes,
                  mlir::MLIRContext *context) {
  Reader reader(bytes);
  auto actorBytes = reader.readFrame("memory actor contract domain");
  if (!actorBytes)
    return actorBytes.takeError();
  auto actors = decodeMemoryActorContractDomain(*actorBytes, context);
  if (!actors)
    return actors.takeError();

  auto bindingCount = reader.readU64("memory role binding count");
  if (!bindingCount)
    return bindingCount.takeError();
  if (*bindingCount > reader.remaining() / 12)
    return invalid("memory role binding count exceeds its framing");
  std::vector<MemoryRoleEndpointBindingRecord> bindings;
  bindings.reserve(*bindingCount);
  for (std::uint64_t index = 0; index < *bindingCount; ++index) {
    auto roleBytes = reader.readFrame("memory service role");
    if (!roleBytes)
      return roleBytes.takeError();
    auto role = dataflow::decodeServiceValueRole(*roleBytes);
    if (!role)
      return role.takeError();
    auto endpoint = reader.readU64("memory role endpoint ordinal");
    if (!endpoint)
      return endpoint.takeError();
    bindings.push_back({*role, *endpoint});
  }

  auto accessTag = reader.readU32("memory access-domain presence");
  if (!accessTag)
    return accessTag.takeError();
  std::optional<ParameterizedMemoryAccessDomain> accesses;
  if (*accessTag == 1) {
    auto accessBytes = reader.readFrame("memory access domain");
    if (!accessBytes)
      return accessBytes.takeError();
    auto decoded = decodeParameterizedMemoryAccessDomain(*accessBytes);
    if (!decoded)
      return decoded.takeError();
    accesses = std::move(*decoded);
  } else if (*accessTag != 0) {
    return invalid("memory access-domain presence has an unknown tag");
  }

  auto patternCount = reader.readU64("memory admissible pattern count");
  if (!patternCount)
    return patternCount.takeError();
  if (*patternCount == 0 || *patternCount > reader.remaining() / 4)
    return invalid("memory admissible pattern count exceeds its framing");
  std::vector<UsePatternKey> patterns;
  patterns.reserve(*patternCount);
  for (std::uint64_t index = 0; index < *patternCount; ++index) {
    auto ordinal = reader.readU32("memory admissible pattern ordinal");
    if (!ordinal)
      return ordinal.takeError();
    patterns.emplace_back(*ordinal);
  }
  if (llvm::Error error = reader.finish("memory capability alternative"))
    return std::move(error);
  return MemoryCapabilityAlternativeRecord{
      std::move(*actors), std::move(bindings), std::move(accesses),
      std::move(patterns)};
}

llvm::Expected<MemoryOperationPortDeclaration>
decodeDeclaration(llvm::ArrayRef<std::uint8_t> bytes,
                  mlir::MLIRContext *context) {
  Reader reader(bytes);
  std::vector<std::uint64_t> endpointInventory;

  auto endpointCount = reader.readU64("memory endpoint inventory count");
  if (!endpointCount)
    return endpointCount.takeError();
  if (*endpointCount == 0 || *endpointCount > reader.remaining() / 8)
    return invalid("memory endpoint inventory count exceeds its framing");
  endpointInventory.reserve(*endpointCount);
  for (std::uint64_t index = 0; index < *endpointCount; ++index) {
    auto endpoint = reader.readU64("memory endpoint ordinal");
    if (!endpoint)
      return endpoint.takeError();
    endpointInventory.push_back(*endpoint);
  }

  auto contractBytes = reader.readFrame("memory resource contract");
  if (!contractBytes)
    return contractBytes.takeError();
  auto contract = decodeResourceContractRecord(*contractBytes);
  if (!contract)
    return contract.takeError();
  std::vector<MemoryOperationPatternRecord> operationPatterns;

  auto patternCount = reader.readU64("memory operation pattern count");
  if (!patternCount)
    return patternCount.takeError();
  if (*patternCount > reader.remaining() / 4)
    return invalid("memory operation pattern count exceeds its framing");
  operationPatterns.reserve(*patternCount);
  for (std::uint64_t index = 0; index < *patternCount; ++index) {
    auto tag = reader.readU32("memory transaction projection");
    if (!tag || *tag > std::numeric_limits<std::uint8_t>::max())
      return tag ? invalid("memory transaction projection tag exceeds u8")
                 : tag.takeError();
    auto projection =
        decodeMemoryPortTransactionProjection(static_cast<std::uint8_t>(*tag));
    if (!projection)
      return projection.takeError();
    operationPatterns.push_back({*projection});
  }

  auto alternativeCount = reader.readU64("memory capability count");
  if (!alternativeCount)
    return alternativeCount.takeError();
  if (*alternativeCount == 0 || *alternativeCount > reader.remaining() / 8)
    return invalid("memory capability count exceeds its framing");
  std::vector<MemoryCapabilityAlternativeRecord> alternatives;
  alternatives.reserve(*alternativeCount);
  for (std::uint64_t index = 0; index < *alternativeCount; ++index) {
    auto alternativeBytes = reader.readFrame("memory capability alternative");
    if (!alternativeBytes)
      return alternativeBytes.takeError();
    auto alternative = decodeAlternative(*alternativeBytes, context);
    if (!alternative)
      return alternative.takeError();
    alternatives.push_back(std::move(*alternative));
  }
  if (llvm::Error error = reader.finish("memory operation port record"))
    return std::move(error);
  return MemoryOperationPortDeclaration{
      std::move(endpointInventory), std::move(*contract),
      std::move(operationPatterns), std::move(alternatives)};
}

} // namespace

llvm::Expected<std::vector<std::uint8_t>>
detail::encodeMemoryCapabilityAlternativeRecord(
    const MemoryCapabilityAlternativeRecord &alternative) {
  std::vector<std::uint8_t> bytes;
  auto actors =
      encodeMemoryActorContractDomain(alternative.actorContractDomain);
  if (!actors)
    return actors.takeError();
  if (llvm::Error error = appendFrame(bytes, *actors, "actor domain"))
    return std::move(error);

  if (llvm::Error error = appendCount(bytes, alternative.roleToEndpoint.size(),
                                      "role bindings"))
    return std::move(error);
  for (const MemoryRoleEndpointBindingRecord &binding :
       alternative.roleToEndpoint) {
    auto role = dataflow::encodeServiceValueRole(binding.role);
    if (!role)
      return role.takeError();
    if (llvm::Error error = appendFrame(bytes, role->bytes(), "service role"))
      return std::move(error);
    appendU64(bytes, binding.endpointOrdinal);
  }

  appendU32(bytes, alternative.accessDomain ? 1 : 0);
  if (alternative.accessDomain) {
    auto access =
        encodeParameterizedMemoryAccessDomain(*alternative.accessDomain);
    if (!access)
      return access.takeError();
    if (llvm::Error error = appendFrame(bytes, *access, "access domain"))
      return std::move(error);
  }

  if (llvm::Error error = appendCount(
          bytes, alternative.admissibleUsePatterns.size(), "use patterns"))
    return std::move(error);
  for (UsePatternKey pattern : alternative.admissibleUsePatterns)
    appendU32(bytes, pattern.ordinal());
  return bytes;
}

llvm::Expected<std::vector<std::uint8_t>>
detail::encodeMemoryOperationPortDeclaration(
    const MemoryOperationPortDeclaration &declaration) {
  std::vector<std::uint8_t> bytes;
  if (llvm::Error error = appendCount(
          bytes, declaration.endpointInventory.size(), "endpoint inventory"))
    return std::move(error);
  for (std::uint64_t endpoint : declaration.endpointInventory)
    appendU64(bytes, endpoint);

  auto contract = encodeResourceContractRecord(declaration.resourceContract);
  if (!contract)
    return contract.takeError();
  if (llvm::Error error = appendFrame(bytes, *contract, "resource contract"))
    return std::move(error);

  if (llvm::Error error =
          appendCount(bytes, declaration.operationPatternSemantics.size(),
                      "operation patterns"))
    return std::move(error);
  for (const MemoryOperationPatternRecord &pattern :
       declaration.operationPatternSemantics)
    appendU32(bytes, getCanonicalTag(pattern.transactionProjection));

  if (llvm::Error error =
          appendCount(bytes, declaration.capabilityAlternatives.size(),
                      "capability alternatives"))
    return std::move(error);
  for (const MemoryCapabilityAlternativeRecord &alternative :
       declaration.capabilityAlternatives) {
    auto encoded = encodeMemoryCapabilityAlternativeRecord(alternative);
    if (!encoded)
      return encoded.takeError();
    if (llvm::Error error =
            appendFrame(bytes, *encoded, "capability alternative"))
      return std::move(error);
  }
  return bytes;
}

llvm::Expected<std::vector<std::uint8_t>>
encodeMemoryOperationPortRecord(const MemoryOperationPortRecord &record) {
  MemoryOperationPortDeclaration declaration{
      std::vector<std::uint64_t>(record.endpointInventory().begin(),
                                 record.endpointInventory().end()),
      record.resourceContract(),
      std::vector<MemoryOperationPatternRecord>(
          record.operationPatterns().begin(), record.operationPatterns().end()),
      std::vector<MemoryCapabilityAlternativeRecord>(
          record.capabilityAlternatives().begin(),
          record.capabilityAlternatives().end())};
  return detail::encodeMemoryOperationPortDeclaration(declaration);
}

llvm::Expected<MemoryOperationPortRecord> decodeMemoryOperationPortRecord(
    llvm::ArrayRef<std::uint8_t> bytes, mlir::MLIRContext *context,
    Schedule schedule,
    llvm::ArrayRef<MemoryTransportEndpointDescriptor> endpoints) {
  auto declaration = decodeDeclaration(bytes, context);
  if (!declaration)
    return declaration.takeError();
  auto record = MemoryOperationPortRecord::fromCanonical(
      context, schedule, endpoints, std::move(*declaration));
  if (!record)
    return record.takeError();
  auto canonical = encodeMemoryOperationPortRecord(*record);
  if (!canonical)
    return canonical.takeError();
  if (llvm::ArrayRef<std::uint8_t>(*canonical) != bytes)
    return invalid("memory operation port bytes are not canonical");
  return record;
}

} // namespace fabric
