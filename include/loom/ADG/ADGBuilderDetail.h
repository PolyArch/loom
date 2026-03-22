//===-- ADGBuilderDetail.h - ADG Builder internal details -------*- C++ -*-===//
//
// Part of the loom project.
//
//===----------------------------------------------------------------------===//
//
// Internal data structures and helpers shared by ADGBuilder implementation
// units. This header is not part of the public builder API.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_ADG_ADGBUILDERDETAIL_H
#define LOOM_ADG_ADGBUILDERDETAIL_H

#include "loom/ADG/ADGBuilder.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace loom {
namespace adg {

struct FUDef {
  std::string name;
  std::vector<std::string> inputTypes;
  std::vector<std::string> outputTypes;
  std::vector<std::string> ops;
  std::string rawBody;
  std::int64_t latency = 1;
  std::int64_t interval = 1;
};

struct PEDef {
  std::string name;
  std::vector<std::string> inputTypes;
  std::vector<std::string> outputTypes;
  std::vector<unsigned> fuIndices;
  bool temporal = false;
  unsigned numRegister = 0;
  unsigned numInstruction = 1;
  unsigned regFifoDepth = 0;
  bool enableShareOperandBuffer = false;
  std::optional<unsigned> operandBufferSize;
};

struct SWDef {
  std::string name;
  std::vector<std::string> inputTypes;
  std::vector<std::string> outputTypes;
  std::vector<std::vector<bool>> connectivity;
  bool temporal = false;
  int decomposableBits = -1;
  unsigned numRouteTable = 1;
};

struct MemoryDef {
  std::string name;
  unsigned ldPorts = 1;
  unsigned stPorts = 1;
  unsigned lsqDepth = 0;
  std::string memrefType = "memref<256xi32>";
  unsigned numRegion = 1;
  bool isPrivate = true;
};

struct ExtMemDef {
  std::string name;
  unsigned ldPorts = 1;
  unsigned stPorts = 1;
  unsigned lsqDepth = 0;
  std::string memrefType = "memref<?xi32>";
  unsigned numRegion = 1;
};

struct FIFODef {
  std::string name;
  unsigned depth = 2;
  unsigned bitsWidth = 32;
};

struct AddTagNodeDef {
  std::string inputType;
  std::string outputType;
  std::uint64_t tag = 0;
};

struct MapTagNodeDef {
  std::string inputType;
  std::string outputType;
  std::vector<MapTagEntrySpec> table;
};

struct DelTagNodeDef {
  std::string inputType;
  std::string outputType;
};

enum class InstanceKind { PE, SW, Memory, ExtMem, FIFO, AddTag, MapTag, DelTag };

struct InstanceDef {
  InstanceKind kind;
  unsigned defIdx;
  std::string name;
};

struct Connection {
  unsigned srcInst;
  unsigned srcPort;
  unsigned dstInst;
  unsigned dstPort;
};

struct MemrefInput {
  std::string name;
  std::string typeStr;
};

struct MemrefConnection {
  unsigned memrefIdx;
  unsigned instIdx;
};

struct ScalarInput {
  std::string name;
  std::string typeStr;
};

struct ScalarOutput {
  std::string name;
  std::string typeStr;
};

struct ScalarToInstanceConn {
  unsigned scalarIdx;
  unsigned dstInst;
  unsigned dstPort;
};

struct InstanceToScalarConn {
  unsigned srcInst;
  unsigned srcPort;
  unsigned scalarOutputIdx;
};

struct VizPlacement {
  double centerX = 0.0;
  double centerY = 0.0;
  int gridRow = -1;
  int gridCol = -1;
};

struct ADGBuilder::Impl {
  std::string moduleName;

  std::vector<FUDef> fuDefs;
  std::vector<PEDef> peDefs;
  std::vector<SWDef> swDefs;
  std::vector<MemoryDef> memoryDefs;
  std::vector<ExtMemDef> extMemDefs;
  std::vector<FIFODef> fifoDefs;
  std::vector<AddTagNodeDef> addTagDefs;
  std::vector<MapTagNodeDef> mapTagDefs;
  std::vector<DelTagNodeDef> delTagDefs;

  std::vector<InstanceDef> instances;
  std::vector<Connection> connections;

  std::vector<MemrefInput> memrefInputs;
  std::vector<MemrefConnection> memrefConnections;

  std::vector<ScalarInput> scalarInputs;
  std::vector<ScalarOutput> scalarOutputs;

  std::vector<ScalarToInstanceConn> scalarToInstConns;
  std::vector<InstanceToScalarConn> instToScalarConns;
  std::map<unsigned, VizPlacement> vizPlacements;

  std::string generateMLIR(llvm::StringRef vizFileName) const;
  std::string generateVizJson() const;
  bool validate(std::string &errMsg) const;
};

namespace detail {

std::string bitsType(unsigned width);

std::string taggedType(llvm::StringRef valueType, unsigned tagWidth);

std::optional<unsigned> tryParseBitsWidth(llvm::StringRef typeStr);

std::optional<unsigned> tryParseScalarWidth(llvm::StringRef typeStr);

std::optional<std::string> tryParseMemrefElementType(llvm::StringRef typeStr);

unsigned getMinMemoryTagWidth(unsigned laneCount);

unsigned getMemoryInputCount(unsigned ldPorts, unsigned stPorts, bool isExtMem);

unsigned getMemoryOutputCount(unsigned ldPorts, unsigned stPorts,
                              bool hasPublicMemrefResult);

std::string getDefaultMemoryInputType(unsigned ldPorts, unsigned stPorts,
                                      llvm::StringRef memrefType, bool isExtMem,
                                      unsigned portIdx);

std::string getDefaultMemoryOutputType(unsigned ldPorts, unsigned stPorts,
                                       llvm::StringRef memrefType,
                                       bool hasPublicMemrefResult,
                                       unsigned portIdx);

std::optional<unsigned> inferUniformBitsWidth(
    const std::vector<std::string> &types, unsigned prefixCount);

void emitFUBody(std::ostringstream &os, const FUDef &fu,
                const std::string &indent);

unsigned getInstanceOutputCount(const std::vector<InstanceDef> &instances,
                                const std::vector<PEDef> &peDefs,
                                const std::vector<SWDef> &swDefs,
                                const std::vector<MemoryDef> &memoryDefs,
                                const std::vector<ExtMemDef> &extMemDefs,
                                unsigned instIdx);

std::string getInstanceInputType(const std::vector<InstanceDef> &instances,
                                 const std::vector<PEDef> &peDefs,
                                 const std::vector<SWDef> &swDefs,
                                 const std::vector<MemoryDef> &memoryDefs,
                                 const std::vector<ExtMemDef> &extMemDefs,
                                 const std::vector<AddTagNodeDef> &addTagDefs,
                                 const std::vector<MapTagNodeDef> &mapTagDefs,
                                 const std::vector<DelTagNodeDef> &delTagDefs,
                                 const std::vector<FIFODef> &fifoDefs,
                                 unsigned instIdx, unsigned portIdx);

std::string getInstanceOutputType(const std::vector<InstanceDef> &instances,
                                  const std::vector<PEDef> &peDefs,
                                  const std::vector<SWDef> &swDefs,
                                  const std::vector<MemoryDef> &memoryDefs,
                                  const std::vector<ExtMemDef> &extMemDefs,
                                  const std::vector<AddTagNodeDef> &addTagDefs,
                                  const std::vector<MapTagNodeDef> &mapTagDefs,
                                  const std::vector<DelTagNodeDef> &delTagDefs,
                                  const std::vector<FIFODef> &fifoDefs,
                                  unsigned instIdx, unsigned portIdx);

} // namespace detail

} // namespace adg
} // namespace loom

#endif // LOOM_ADG_ADGBUILDERDETAIL_H
