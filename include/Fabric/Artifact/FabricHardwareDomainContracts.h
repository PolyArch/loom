#ifndef LOOM_FABRIC_ARTIFACT_FABRICHARDWAREDOMAINCONTRACTS_H
#define LOOM_FABRIC_ARTIFACT_FABRICHARDWAREDOMAINCONTRACTS_H

#include "Fabric/Artifact/FabricSystemContracts.h"
#include "Fabric/IR/MemoryConsistencyContract.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

namespace loom::fabric {

class PowerDomainContractRecord {
public:
  static llvm::Expected<PowerDomainContractRecord>
  create(std::uint64_t nominalVoltageUv);

  std::uint64_t nominalVoltageUv() const { return nominalVoltageUv_; }

private:
  explicit PowerDomainContractRecord(std::uint64_t nominalVoltageUv)
      : nominalVoltageUv_(nominalVoltageUv) {}

  std::uint64_t nominalVoltageUv_;
};

struct AddressDomainRange {
  llvm::APInt lower;
  llvm::APInt upperExclusive;
};

class AddressDomainContractRecord {
public:
  static llvm::Expected<AddressDomainContractRecord>
  create(std::uint32_t addressWidth, std::vector<AddressDomainRange> ranges);

  std::uint32_t addressWidth() const { return addressWidth_; }
  llvm::ArrayRef<AddressDomainRange> ranges() const { return ranges_; }

private:
  AddressDomainContractRecord(std::uint32_t addressWidth,
                              std::vector<AddressDomainRange> ranges)
      : addressWidth_(addressWidth), ranges_(std::move(ranges)) {}

  std::uint32_t addressWidth_;
  std::vector<AddressDomainRange> ranges_;
};

using HardwareDomainContract =
    std::variant<ClockDomainContractRecord, ResetDomainContractRecord,
                 PowerDomainContractRecord, AddressDomainContractRecord,
                 ::fabric::MemoryConsistencyContract>;

/// One complete hardware-domain declaration. The variant is the sole owner of
/// the domain kind; members retain their exact existing Fabric references.
class HardwareDomainContractRecord {
public:
  static llvm::Expected<HardwareDomainContractRecord>
  create(std::vector<FabricInventoryOwnerRef> members,
         HardwareDomainContract contract);

  FabricHardwareDomainKind kind() const;
  llvm::ArrayRef<FabricInventoryOwnerRef> members() const { return members_; }
  const HardwareDomainContract &contract() const { return contract_; }

private:
  HardwareDomainContractRecord(std::vector<FabricInventoryOwnerRef> members,
                               HardwareDomainContract contract)
      : members_(std::move(members)), contract_(std::move(contract)) {}

  std::vector<FabricInventoryOwnerRef> members_;
  HardwareDomainContract contract_;
};

llvm::Expected<std::vector<std::uint8_t>>
encodeHardwareDomainContractRecord(const HardwareDomainContractRecord &record);

llvm::Expected<HardwareDomainContractRecord>
decodeHardwareDomainContractRecord(llvm::ArrayRef<std::uint8_t> bytes);

} // namespace loom::fabric

#endif // LOOM_FABRIC_ARTIFACT_FABRICHARDWAREDOMAINCONTRACTS_H
