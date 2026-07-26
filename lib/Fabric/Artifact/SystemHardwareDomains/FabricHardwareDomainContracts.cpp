#include "Fabric/Artifact/FabricHardwareDomainContracts.h"

#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <system_error>
#include <utility>

using namespace loom::fabric;

namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "hardware_domain_contract_invalid: " + message);
}

llvm::Error malformed(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "hardware_domain_contract_malformed: " + message);
}

class Writer {
public:
  void u32(std::uint32_t value) {
    for (int shift = 24; shift >= 0; shift -= 8)
      bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
  }

  void u64(std::uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8)
      bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
  }

  void bytes(llvm::ArrayRef<std::uint8_t> value) {
    bytes_.insert(bytes_.end(), value.begin(), value.end());
  }

  void blob(llvm::ArrayRef<std::uint8_t> value) {
    u64(value.size());
    bytes(value);
  }

  std::vector<std::uint8_t> take() { return std::move(bytes_); }

private:
  std::vector<std::uint8_t> bytes_;
};

class Reader {
public:
  explicit Reader(llvm::ArrayRef<std::uint8_t> bytes) : bytes_(bytes) {}

  llvm::Expected<std::uint32_t> u32(llvm::StringRef field) {
    if (bytes_.size() < 4)
      return malformed(field + " is a truncated uint32");
    std::uint32_t value = 0;
    for (unsigned index = 0; index < 4; ++index)
      value = (value << 8) | bytes_[index];
    bytes_ = bytes_.drop_front(4);
    return value;
  }

  llvm::Expected<std::uint64_t> u64(llvm::StringRef field) {
    if (bytes_.size() < 8)
      return malformed(field + " is a truncated uint64");
    std::uint64_t value = 0;
    for (unsigned index = 0; index < 8; ++index)
      value = (value << 8) | bytes_[index];
    bytes_ = bytes_.drop_front(8);
    return value;
  }

  llvm::Expected<std::uint32_t> tag(std::uint32_t count,
                                    llvm::StringRef field) {
    auto value = u32(field);
    if (!value)
      return value.takeError();
    if (*value >= count)
      return malformed(field + " has an unknown discriminant");
    return *value;
  }

  llvm::Expected<std::uint64_t> count(std::size_t minimumEntryBytes,
                                      llvm::StringRef field) {
    auto value = u64(field);
    if (!value)
      return value.takeError();
    if (minimumEntryBytes != 0 && *value > bytes_.size() / minimumEntryBytes)
      return malformed(field + " exceeds remaining framing");
    return *value;
  }

  llvm::Expected<llvm::ArrayRef<std::uint8_t>> take(std::uint64_t size,
                                                    llvm::StringRef field) {
    if (size > bytes_.size())
      return malformed(field + " is truncated");
    llvm::ArrayRef<std::uint8_t> result =
        bytes_.take_front(static_cast<std::size_t>(size));
    bytes_ = bytes_.drop_front(static_cast<std::size_t>(size));
    return result;
  }

  llvm::Expected<llvm::ArrayRef<std::uint8_t>> blob(llvm::StringRef field) {
    auto size = u64((field + " length").str());
    if (!size)
      return size.takeError();
    return take(*size, field);
  }

  llvm::Error finish() const {
    if (!bytes_.empty())
      return malformed("record has trailing bytes");
    return llvm::Error::success();
  }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
};

void writeAPInt(Writer &writer, const llvm::APInt &value) {
  const unsigned width = value.getBitWidth();
  const unsigned byteCount = (width + 7) / 8;
  for (unsigned index = byteCount; index != 0; --index) {
    const unsigned bit = (index - 1) * 8;
    const unsigned count = std::min(8u, width - bit);
    const std::uint8_t byte =
        static_cast<std::uint8_t>(value.extractBitsAsZExtValue(count, bit));
    writer.bytes({byte});
  }
}

llvm::Expected<llvm::APInt> readAPInt(Reader &reader, unsigned width,
                                      llvm::StringRef field) {
  const unsigned byteCount = (width + 7) / 8;
  auto bytes = reader.take(byteCount, field);
  if (!bytes)
    return bytes.takeError();
  const unsigned unusedBits = byteCount * 8 - width;
  if (unusedBits != 0 && ((*bytes)[0] >> (8 - unusedBits)) != 0)
    return malformed(field + " has nonzero unused high bits");
  llvm::APInt value(width, 0);
  for (std::uint8_t byte : *bytes) {
    value <<= 8;
    value |= byte;
  }
  return value;
}

llvm::Expected<std::vector<std::uint8_t>>
encodeAddressDomain(const AddressDomainContractRecord &contract) {
  Writer writer;
  writer.u32(contract.addressWidth());
  writer.u64(contract.ranges().size());
  for (const AddressDomainRange &range : contract.ranges()) {
    writeAPInt(writer, range.lower);
    writeAPInt(writer, range.upperExclusive);
  }
  return writer.take();
}

llvm::Expected<AddressDomainContractRecord>
decodeAddressDomain(llvm::ArrayRef<std::uint8_t> bytes) {
  Reader reader(bytes);
  auto addressWidth = reader.u32("address width");
  if (!addressWidth)
    return addressWidth.takeError();
  if (*addressWidth == 0 || *addressWidth > 64)
    return malformed("address width is outside [1, 64]");
  const unsigned boundWidth = *addressWidth + 1;
  const std::size_t bytesPerBound = (boundWidth + 7) / 8;
  auto count = reader.count(bytesPerBound * 2, "address range count");
  if (!count)
    return count.takeError();
  std::vector<AddressDomainRange> ranges;
  ranges.reserve(static_cast<std::size_t>(*count));
  for (std::uint64_t index = 0; index < *count; ++index) {
    auto lower = readAPInt(reader, boundWidth, "address range lower bound");
    if (!lower)
      return lower.takeError();
    auto upper = readAPInt(reader, boundWidth, "address range upper bound");
    if (!upper)
      return upper.takeError();
    ranges.push_back({std::move(*lower), std::move(*upper)});
  }
  if (llvm::Error error = reader.finish())
    return std::move(error);
  auto contract =
      AddressDomainContractRecord::create(*addressWidth, std::move(ranges));
  if (!contract)
    return contract.takeError();
  auto canonical = encodeAddressDomain(*contract);
  if (!canonical)
    return canonical.takeError();
  if (llvm::ArrayRef<std::uint8_t>(*canonical) != bytes)
    return malformed("address domain is not canonical");
  return contract;
}

llvm::Expected<std::vector<std::uint8_t>>
encodePowerDomain(const PowerDomainContractRecord &contract) {
  Writer writer;
  writer.u64(contract.nominalVoltageUv());
  return writer.take();
}

llvm::Expected<PowerDomainContractRecord>
decodePowerDomain(llvm::ArrayRef<std::uint8_t> bytes) {
  Reader reader(bytes);
  auto voltage = reader.u64("nominal voltage");
  if (!voltage)
    return voltage.takeError();
  if (llvm::Error error = reader.finish())
    return std::move(error);
  return PowerDomainContractRecord::create(*voltage);
}

llvm::Expected<std::vector<std::uint8_t>>
encodeContractPayload(const HardwareDomainContractRecord &record) {
  switch (record.kind()) {
  case FabricHardwareDomainKind::Clock:
    return encodeClockDomainContractRecord(
        std::get<ClockDomainContractRecord>(record.contract()));
  case FabricHardwareDomainKind::Reset:
    return encodeResetDomainContractRecord(
        std::get<ResetDomainContractRecord>(record.contract()));
  case FabricHardwareDomainKind::Power:
    return encodePowerDomain(
        std::get<PowerDomainContractRecord>(record.contract()));
  case FabricHardwareDomainKind::Address:
    return encodeAddressDomain(
        std::get<AddressDomainContractRecord>(record.contract()));
  case FabricHardwareDomainKind::MemoryConsistency:
    return ::fabric::encodeMemoryConsistencyContractRecord(
        std::get<::fabric::MemoryConsistencyContract>(record.contract()));
  }
  llvm_unreachable("closed hardware-domain kind has no payload codec");
}

} // namespace

llvm::Expected<PowerDomainContractRecord>
PowerDomainContractRecord::create(std::uint64_t nominalVoltageUv) {
  if (nominalVoltageUv == 0)
    return invalid("power domain nominal voltage must be positive");
  return PowerDomainContractRecord(nominalVoltageUv);
}

llvm::Expected<AddressDomainContractRecord>
AddressDomainContractRecord::create(std::uint32_t addressWidth,
                                    std::vector<AddressDomainRange> ranges) {
  if (addressWidth == 0 || addressWidth > 64)
    return invalid("address width must be in [1, 64]");
  if (ranges.empty())
    return invalid("address domain range set must not be empty");

  const unsigned boundWidth = addressWidth + 1;
  const llvm::APInt limit = llvm::APInt(boundWidth, 1).shl(addressWidth);
  for (AddressDomainRange &range : ranges) {
    if (range.lower.getActiveBits() > boundWidth ||
        range.upperExclusive.getActiveBits() > boundWidth)
      return invalid("address range exceeds its address width");
    range.lower = range.lower.zextOrTrunc(boundWidth);
    range.upperExclusive = range.upperExclusive.zextOrTrunc(boundWidth);
    if (!range.lower.ult(range.upperExclusive) ||
        range.upperExclusive.ugt(limit))
      return invalid("address range is empty or exceeds the address space");
  }

  llvm::sort(ranges, [](const AddressDomainRange &left,
                        const AddressDomainRange &right) {
    if (left.lower != right.lower)
      return left.lower.ult(right.lower);
    return left.upperExclusive.ult(right.upperExclusive);
  });

  std::vector<AddressDomainRange> normalized;
  normalized.reserve(ranges.size());
  for (AddressDomainRange &range : ranges) {
    if (normalized.empty() ||
        normalized.back().upperExclusive.ult(range.lower)) {
      normalized.push_back(std::move(range));
      continue;
    }
    if (normalized.back().upperExclusive.ult(range.upperExclusive))
      normalized.back().upperExclusive = std::move(range.upperExclusive);
  }
  return AddressDomainContractRecord(addressWidth, std::move(normalized));
}

llvm::Expected<HardwareDomainContractRecord>
HardwareDomainContractRecord::create(
    std::vector<FabricInventoryOwnerRef> members,
    HardwareDomainContract contract) {
  if (members.empty())
    return invalid("hardware domain member set must not be empty");
  std::vector<std::pair<std::vector<std::uint8_t>, FabricInventoryOwnerRef>>
      ordered;
  ordered.reserve(members.size());
  for (FabricInventoryOwnerRef &member : members)
    ordered.emplace_back(canonicalFabricBytes(member), std::move(member));
  llvm::sort(ordered, [](const auto &left, const auto &right) {
    return left.first < right.first;
  });
  for (std::size_t index = 1; index < ordered.size(); ++index)
    if (ordered[index - 1].first == ordered[index].first)
      return invalid("hardware domain contains a duplicate member");
  members.clear();
  members.reserve(ordered.size());
  for (auto &entry : ordered)
    members.push_back(std::move(entry.second));
  return HardwareDomainContractRecord(std::move(members), std::move(contract));
}

FabricHardwareDomainKind HardwareDomainContractRecord::kind() const {
  if (std::holds_alternative<ClockDomainContractRecord>(contract_))
    return FabricHardwareDomainKind::Clock;
  if (std::holds_alternative<ResetDomainContractRecord>(contract_))
    return FabricHardwareDomainKind::Reset;
  if (std::holds_alternative<PowerDomainContractRecord>(contract_))
    return FabricHardwareDomainKind::Power;
  if (std::holds_alternative<AddressDomainContractRecord>(contract_))
    return FabricHardwareDomainKind::Address;
  return FabricHardwareDomainKind::MemoryConsistency;
}

llvm::Expected<std::vector<std::uint8_t>>
loom::fabric::encodeHardwareDomainContractRecord(
    const HardwareDomainContractRecord &record) {
  Writer writer;
  writer.u32(static_cast<std::uint32_t>(record.kind()));
  writer.u64(record.members().size());
  for (const FabricInventoryOwnerRef &member : record.members())
    writer.blob(canonicalFabricBytes(member));
  auto payload = encodeContractPayload(record);
  if (!payload)
    return payload.takeError();
  writer.blob(*payload);
  return writer.take();
}

llvm::Expected<HardwareDomainContractRecord>
loom::fabric::decodeHardwareDomainContractRecord(
    llvm::ArrayRef<std::uint8_t> bytes) {
  Reader reader(bytes);
  auto kind = reader.tag(fabricClosedBound(FabricHardwareDomainKind{}),
                         "hardware domain kind");
  if (!kind)
    return kind.takeError();
  auto memberCount = reader.count(8, "hardware domain member count");
  if (!memberCount)
    return memberCount.takeError();
  std::vector<FabricInventoryOwnerRef> members;
  members.reserve(static_cast<std::size_t>(*memberCount));
  for (std::uint64_t index = 0; index < *memberCount; ++index) {
    auto memberBytes = reader.blob("hardware domain member");
    if (!memberBytes)
      return memberBytes.takeError();
    auto member = decodeFabricRef<FabricInventoryOwnerRef>(*memberBytes);
    if (!member)
      return member.takeError();
    members.push_back(std::move(*member));
  }
  auto payload = reader.blob("hardware domain contract");
  if (!payload)
    return payload.takeError();
  if (llvm::Error error = reader.finish())
    return std::move(error);

  std::optional<HardwareDomainContract> contract;
  switch (static_cast<FabricHardwareDomainKind>(*kind)) {
  case FabricHardwareDomainKind::Clock: {
    auto value = decodeClockDomainContractRecord(*payload);
    if (!value)
      return value.takeError();
    contract.emplace(std::move(*value));
    break;
  }
  case FabricHardwareDomainKind::Reset: {
    auto value = decodeResetDomainContractRecord(*payload);
    if (!value)
      return value.takeError();
    contract.emplace(std::move(*value));
    break;
  }
  case FabricHardwareDomainKind::Power: {
    auto value = decodePowerDomain(*payload);
    if (!value)
      return value.takeError();
    contract.emplace(std::move(*value));
    break;
  }
  case FabricHardwareDomainKind::Address: {
    auto value = decodeAddressDomain(*payload);
    if (!value)
      return value.takeError();
    contract.emplace(std::move(*value));
    break;
  }
  case FabricHardwareDomainKind::MemoryConsistency: {
    auto value = ::fabric::decodeMemoryConsistencyContractRecord(*payload);
    if (!value)
      return value.takeError();
    contract.emplace(std::move(*value));
    break;
  }
  }

  auto record = HardwareDomainContractRecord::create(std::move(members),
                                                     std::move(*contract));
  if (!record)
    return record.takeError();
  auto canonical = encodeHardwareDomainContractRecord(*record);
  if (!canonical)
    return canonical.takeError();
  if (llvm::ArrayRef<std::uint8_t>(*canonical) != bytes)
    return malformed("hardware domain record is not canonical");
  return record;
}
