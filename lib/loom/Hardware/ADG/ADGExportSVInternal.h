//===-- ADGExportSVInternal.h - Internal declarations for SV export -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Internal header shared by ADGExportSV*.cpp files.  Not part of the public
// API; lives next to the implementation files rather than in include/.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_HARDWARE_ADG_ADGEXPORTSVINTERNAL_H
#define LOOM_HARDWARE_ADG_ADGEXPORTSVINTERNAL_H

#include "loom/Hardware/ADG/ADGBuilderImpl.h"

#include <string>
#include <utility>
#include <vector>

namespace loom {
namespace adg {

//===----------------------------------------------------------------------===//
// MLIRStmt struct -- parsed representation of a single MLIR SSA line
//===----------------------------------------------------------------------===//

/// Parse a simple MLIR SSA line: "%res = dialect.op %a, %b : type"
/// Returns {result, opName, {operands}, typeAnnotation} or empty result on
/// failure.
struct MLIRStmt {
  std::string result; // e.g., "%t0"
  std::string opName; // e.g., "arith.muli"
  std::vector<std::string> operands; // e.g., {"%arg0", "%arg1"}
  std::string typeAnnotation; // e.g., "i16 to i32" (text after ':')
  std::string predicate; // e.g., "sgt" for compare ops
};

//===----------------------------------------------------------------------===//
// Utility functions (ADGExportSVUtil.cpp)
//===----------------------------------------------------------------------===//

unsigned getDataWidthBits(const Type &t);
unsigned getTagWidthBits(const Type &t);

void writeFile(const std::string &path, const std::string &content);
void copyTemplateFile(const std::string &srcDir,
                      const std::string &relPath,
                      const std::string &dstDir);
std::string readFile(const std::string &path);

unsigned getOpLatency(const std::string &opName);
std::string opToSVModule(const std::string &opName);
bool isConversionOp(const std::string &opName);
bool isCompareOp(const std::string &opName);

int cmpiPredicateToInt(const std::string &pred);
int cmpfPredicateToInt(const std::string &pred);
int resolveComparePredicate(const std::string &opName,
                            const std::string &pred);

MLIRStmt parseMLIRLine(const std::string &line);
unsigned parseMLIRTypeWidth(const std::string &typeStr);
std::pair<unsigned, unsigned>
parseConversionWidths(const std::string &typeAnnotation);

std::vector<std::string> extractBodyMLIROps(const std::string &bodyMLIR);
unsigned computeBodyMLIRLatency(const std::string &bodyMLIR);
unsigned computeBodyMLIRMaxWidth(const std::string &bodyMLIR);

bool isValidSVIdentifier(const std::string &name);

//===----------------------------------------------------------------------===//
// Generation functions (ADGExportSVGen.cpp)
//===----------------------------------------------------------------------===//

const char *svModuleName(ModuleKind kind);

unsigned getNumConnected(const SwitchDef &def);
unsigned getNumConnected(const TemporalSwitchDef &def);

std::string genFullPESV(const PEDef &def);
std::string genFullTemporalPESV(const std::string &templateDir,
                                 const TemporalPEDef &def,
                                 const std::vector<PEDef> &peDefs,
                                 const std::string &instName);
std::string fillPETemplate(const std::string &templateDir,
                            const PEDef &def);

std::string genConstantPEParams(const ConstantPEDef &def);
unsigned ceilLog2(unsigned v);
std::string genLoadPEParams(const LoadPEDef &def);
std::string genStorePEParams(const StorePEDef &def);
std::string genTemporalSwitchParams(const TemporalSwitchDef &def);
std::string genMemoryParams(const MemoryDef &def);
std::string genExtMemoryParams(const ExtMemoryDef &def);
std::string genSwitchParams(const SwitchDef &def);
std::string genAddTagParams(const AddTagDef &def);
std::string genDelTagParams(const DelTagDef &def);
std::string genMapTagParams(const MapTagDef &def);
std::string genFifoParams(const FifoDef &def);

bool hasSVTemplate(ModuleKind kind);
bool hasErrorOutput(ModuleKind kind);

} // namespace adg
} // namespace loom

#endif // LOOM_HARDWARE_ADG_ADGEXPORTSVINTERNAL_H
