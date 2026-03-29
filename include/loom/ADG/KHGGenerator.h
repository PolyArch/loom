//===-- KHGGenerator.h - Combinatorial KHG type ADG generator -----*- C++ -*-===//
//
// Part of the loom project.
//
//===----------------------------------------------------------------------===//
//
// Generates ADG MLIR for the 24 combinatorial KHG types
// (3 compute-mix x 2 PE-type x 2 SPM x 2 array-size).
//
// Each type is identified by a canonical string ID: C{I|F|M}{S|T}{Y|N}{8|12}
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_ADG_KHGGENERATOR_H
#define LOOM_ADG_KHGGENERATOR_H

#include <string>
#include <vector>

namespace loom {
namespace adg {

//===----------------------------------------------------------------------===//
// KHG Dimension Enumerations
//===----------------------------------------------------------------------===//

enum class KHGComputeMix {
  INT_HEAVY,  // I: alu=6, mul=4, fp=1
  FP_HEAVY,   // F: alu=2, mul=2, fp=6
  MIXED,      // M: alu=4, mul=3, fp=3
};

enum class KHGPEKind {
  SPATIAL,   // S
  TEMPORAL,  // T (instructionSlots=8, numRegisters=8)
};

enum class KHGSPMPresence {
  WITH_SPM,     // Y: spmSizeKB=16, ldPorts=2, stPorts=2
  WITHOUT_SPM,  // N: spmSizeKB=0, ldPorts=0, stPorts=0
};

enum class KHGArraySize {
  SIZE_8,   // 8x8
  SIZE_12,  // 12x12
};

//===----------------------------------------------------------------------===//
// KHG Type Parameters
//===----------------------------------------------------------------------===//

/// Concrete parameters for a single KHG combinatorial core type.
struct KHGTypeParams {
  std::string typeId;
  KHGComputeMix computeMix;
  KHGPEKind peKind;
  KHGSPMPresence spmPresence;
  KHGArraySize arraySize;

  unsigned arrayRows;
  unsigned arrayCols;
  unsigned fuAluCount;
  unsigned fuMulCount;
  unsigned fuFpCount;
  unsigned spmSizeKB;
  unsigned spmLdPorts;
  unsigned spmStPorts;
  unsigned instructionSlots;
  unsigned numRegisters;
  unsigned dataWidth = 32;

  unsigned totalPEs() const { return arrayRows * arrayCols; }
  bool isTemporal() const { return peKind == KHGPEKind::TEMPORAL; }
  bool hasSPM() const { return spmPresence == KHGSPMPresence::WITH_SPM; }
};

//===----------------------------------------------------------------------===//
// Naming Convention
//===----------------------------------------------------------------------===//

/// Encode a 4D parameter tuple into a KHG type ID string.
/// Returns e.g. "CISY8", "CFTY12".
std::string encodeKHGTypeId(KHGComputeMix compute, KHGPEKind pe,
                            KHGSPMPresence spm, KHGArraySize size);

/// Decode a KHG type ID string into the 4D parameter tuple.
/// Returns false if the string is not a valid KHG type ID.
bool decodeKHGTypeId(const std::string &typeId, KHGComputeMix &compute,
                     KHGPEKind &pe, KHGSPMPresence &spm, KHGArraySize &size);

/// Return true if the string is a valid KHG type ID.
bool isValidKHGTypeId(const std::string &typeId);

//===----------------------------------------------------------------------===//
// Parameter Construction
//===----------------------------------------------------------------------===//

/// Build the full KHGTypeParams for a given 4D parameter combination.
KHGTypeParams makeKHGParams(KHGComputeMix compute, KHGPEKind pe,
                            KHGSPMPresence spm, KHGArraySize size);

/// Build KHGTypeParams from a type ID string like "CISY8".
/// Returns a default-initialized struct with empty typeId if invalid.
KHGTypeParams paramsFromTypeId(const std::string &typeId);

//===----------------------------------------------------------------------===//
// ADG Generation
//===----------------------------------------------------------------------===//

/// Generate a complete Fabric MLIR ADG string for a KHG type.
/// Uses the ADGBuilder to construct the full ADG and exports as MLIR text.
std::string generateKHGADG(const KHGTypeParams &params);

/// Generate and export a KHG ADG to a file.
void exportKHGADG(const KHGTypeParams &params, const std::string &outputPath);

//===----------------------------------------------------------------------===//
// Enumeration
//===----------------------------------------------------------------------===//

/// Return all 24 KHG type ID strings in canonical order.
std::vector<std::string> allKHGTypeIds();

/// Return KHGTypeParams for all 24 types in canonical order.
std::vector<KHGTypeParams> allKHGTypes();

} // namespace adg
} // namespace loom

#endif // LOOM_ADG_KHGGENERATOR_H
