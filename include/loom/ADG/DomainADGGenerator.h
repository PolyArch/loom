//===-- DomainADGGenerator.h - Domain-specific core type ADG gen ---*- C++ -*-===//
//
// Part of the loom project.
//
//===----------------------------------------------------------------------===//
//
// Generates ADG MLIR for the 6 domain-specific core types (D1-D6) and the
// SciComp-specific SC-FP / SC-SPM / SC-CTRL core types.
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

/// Concrete parameters for a scientific-computing KHG type.
struct SciCompTypeParams {
  std::string typeId; // "SC-FP", "SC-SPM", "SC-CTRL"
  std::string name;

  unsigned arrayRows = 0;
  unsigned arrayCols = 0;
  unsigned dataWidth = 32;
  bool isTemporal = false;

  unsigned fpAddCount = 0;
  unsigned fpMulCount = 0;
  unsigned fpDivCount = 0;
  unsigned intAluCount = 0;
  unsigned intMulCount = 0;

  bool decomposable = false;
  unsigned subLaneBits = 0;

  unsigned spmSizeKB = 0;
  unsigned spmLdPorts = 0;
  unsigned spmStPorts = 0;
  unsigned extMemLdPorts = 0;
  unsigned extMemStPorts = 0;

  unsigned instructionSlots = 0;
  unsigned numRegisters = 0;
  unsigned operandBufferSize = 0;

  unsigned scalarInputs = 4;
  unsigned scalarOutputs = 2;

  std::string routingTopology;

  bool hasFMA = false;
  bool hasRSQRT = false;
  bool hasFPMin = false;
  bool hasIndirectLoad = false;
  bool hasScatterStore = false;
  bool hasBranch = false;

  unsigned totalPEs() const { return arrayRows * arrayCols; }
  unsigned totalFPUnits() const { return fpAddCount + fpMulCount + fpDivCount; }
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

//===----------------------------------------------------------------------===//
// SciComp-specific KHG generation
//===----------------------------------------------------------------------===//

/// Return true if the string is a valid scientific-computing KHG type ID.
bool isValidSciCompTypeId(const std::string &typeId);

/// Build SciComp KHG parameters from a type ID string.
SciCompTypeParams sciCompParamsFromTypeId(const std::string &typeId);

/// Generate a complete Fabric MLIR ADG string for a scientific-computing KHG.
std::string generateSciCompADG(const SciCompTypeParams &params);

/// Generate and export a scientific-computing KHG to a file.
void exportSciCompADG(const SciCompTypeParams &params,
                      const std::string &outputPath);

/// Return all 3 scientific-computing type IDs in canonical order.
std::vector<std::string> allSciCompTypeIds();

/// Return parameters for all 3 scientific-computing KHG types.
std::vector<SciCompTypeParams> allSciCompTypes();

} // namespace adg
} // namespace loom

#endif // LOOM_ADG_DOMAINADGGENERATOR_H
