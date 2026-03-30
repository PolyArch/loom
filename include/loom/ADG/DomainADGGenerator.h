//===-- DomainADGGenerator.h - Domain-specific core type ADG gen ---*- C++ -*-===//
//
// Part of the loom project.
//
//===----------------------------------------------------------------------===//
//
// Generates ADG MLIR for the 6 domain-specific core types (D1-D6).
//
// Each type targets a specific workload domain with hand-tuned parameters:
//   D1: LLM      (FP-heavy, 6x6 spatial, SPM=64KB)
//   D2: CV       (mixed, 4x4 spatial, SPM=32KB)
//   D3: Signal   (temporal, multiply-heavy, SPM=16KB)
//   D4: Crypto   (INT-heavy, 4x4 spatial, 64-bit datapath, SPM=8KB)
//   D5: Sensor   (temporal, control-heavy, SPM=8KB)
//   D6: Control  (spatial, balanced INT, SPM=4KB)
//
// Uses the ADGBuilder API following the same pattern as KHGGenerator.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_ADG_DOMAINADGGENERATOR_H
#define LOOM_ADG_DOMAINADGGENERATOR_H

#include <string>
#include <vector>

namespace loom {
namespace adg {

//===----------------------------------------------------------------------===//
// Domain Type Parameters
//===----------------------------------------------------------------------===//

/// Concrete parameters for a single domain-specific core type (D1-D6).
struct DomainTypeParams {
  std::string typeId;    // "D1".."D6"
  std::string name;      // Human-readable domain name

  unsigned arrayRows;
  unsigned arrayCols;
  unsigned fuAluCount;
  unsigned fuMulCount;
  unsigned fuFpCount;
  unsigned fuMemCount;
  unsigned spmSizeKB;
  bool isTemporal;
  unsigned instructionSlots;
  unsigned numRegisters;
  unsigned dataWidth;

  unsigned totalPEs() const { return arrayRows * arrayCols; }
  bool hasFP() const { return fuFpCount > 0; }
  bool hasSPM() const { return spmSizeKB > 0; }
};

//===----------------------------------------------------------------------===//
// Validation
//===----------------------------------------------------------------------===//

/// Return true if the string is a valid domain-specific type ID (D1-D6).
bool isValidDomainTypeId(const std::string &typeId);

//===----------------------------------------------------------------------===//
// Parameter Construction
//===----------------------------------------------------------------------===//

/// Build DomainTypeParams from a type ID string like "D1".
/// Returns a default-initialized struct with empty typeId if invalid.
DomainTypeParams domainParamsFromTypeId(const std::string &typeId);

//===----------------------------------------------------------------------===//
// ADG Generation
//===----------------------------------------------------------------------===//

/// Generate a complete Fabric MLIR ADG string for a domain-specific type.
/// Uses the ADGBuilder to construct the full ADG and exports as MLIR text.
std::string generateDomainADG(const DomainTypeParams &params);

/// Generate and export a domain-specific ADG to a file.
void exportDomainADG(const DomainTypeParams &params,
                     const std::string &outputPath);

//===----------------------------------------------------------------------===//
// Enumeration
//===----------------------------------------------------------------------===//

/// Return all 6 domain-specific type ID strings in canonical order.
std::vector<std::string> allDomainTypeIds();

/// Return DomainTypeParams for all 6 types in canonical order.
std::vector<DomainTypeParams> allDomainTypes();

} // namespace adg
} // namespace loom

#endif // LOOM_ADG_DOMAINADGGENERATOR_H
