#ifndef LOOM_SIMULATOR_STATICMODELTYPES_H
#define LOOM_SIMULATOR_STATICMODELTYPES_H

#include "loom/Mapper/Types.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace loom {
namespace sim {

enum class StaticModuleKind : uint8_t {
  BoundaryInput = 0,
  BoundaryOutput = 1,
  FunctionUnit = 2,
  SpatialSwitch = 3,
  TemporalSwitch = 4,
  AddTag = 5,
  MapTag = 6,
  DelTag = 7,
  Fifo = 8,
  Memory = 9,
  ExtMemory = 10,
  TemporalPE = 11,
  Unknown = 255,
};

enum class CompletionObligationKind : uint8_t {
  OutputPort = 0,
  MemoryRegion = 1,
};

enum class StaticGraphNodeKind : uint8_t {
  Operation = 0,
  ModuleInput = 1,
  ModuleOutput = 2,
};

enum class StaticPortDirection : uint8_t {
  Input = 0,
  Output = 1,
};

struct StaticIntAttr {
  std::string name;
  int64_t value = 0;
};

struct StaticStringAttr {
  std::string name;
  std::string value;
};

struct StaticByteArrayAttr {
  std::string name;
  std::vector<int8_t> value;
};

struct StaticIntArrayAttr {
  std::string name;
  std::vector<int64_t> value;
};

struct StaticStringArrayAttr {
  std::string name;
  std::vector<std::string> value;
};

struct StaticPortDesc {
  uint32_t portId = 0;
  uint32_t parentNodeId = 0;
  StaticPortDirection direction = StaticPortDirection::Input;
  bool isTagged = false;
  bool isMemRef = false;
  bool isNone = false;
  unsigned valueWidth = 0;
  unsigned tagWidth = 0;
};

struct StaticModuleDesc {
  uint32_t hwNodeId = 0;
  StaticGraphNodeKind nodeKind = StaticGraphNodeKind::Operation;
  StaticModuleKind kind = StaticModuleKind::Unknown;
  std::string name;
  std::string opKind;
  std::string resourceClass;
  std::vector<IdIndex> inputPorts;
  std::vector<IdIndex> outputPorts;
  std::vector<StaticIntAttr> intAttrs;
  std::vector<StaticStringAttr> strAttrs;
  std::vector<StaticByteArrayAttr> byteArrayAttrs;
  std::vector<StaticIntArrayAttr> intArrayAttrs;
  std::vector<StaticStringArrayAttr> stringArrayAttrs;
};

struct StaticChannelDesc {
  uint32_t hwEdgeId = 0;
  IdIndex srcPort = INVALID_ID;
  IdIndex dstPort = INVALID_ID;
  IdIndex srcNode = INVALID_ID;
  IdIndex dstNode = INVALID_ID;
  int peInputIndex = -1;
  int peOutputIndex = -1;
  bool touchesBoundaryInput = false;
  bool touchesBoundaryOutput = false;
};

struct StaticPEDesc {
  std::string peName;
  std::string peKind;
  std::vector<IdIndex> fuNodeIds;
  int row = 0;
  int col = 0;
  unsigned numInputPorts = 0;
  unsigned numOutputPorts = 0;
  unsigned numInstruction = 0;
  unsigned numRegister = 0;
  unsigned regFifoDepth = 0;
  unsigned tagWidth = 0;
  bool enableShareOperandBuffer = false;
  unsigned operandBufferSize = 0;
};

struct StaticInputBinding {
  unsigned boundaryOrdinal = 0;
  IdIndex swNodeId = INVALID_ID;
};

struct StaticOutputBinding {
  unsigned boundaryOrdinal = 0;
  IdIndex swNodeId = INVALID_ID;
};

struct StaticMemoryBinding {
  unsigned regionId = 0;
  unsigned regionIndex = 0;
  IdIndex swNodeId = INVALID_ID;
  IdIndex hwNodeId = INVALID_ID;
  unsigned startLane = 0;
  unsigned endLane = 1;
  unsigned elemSizeLog2 = 0;
  bool supportsLoad = false;
  bool supportsStore = false;
};

struct CompletionObligation {
  CompletionObligationKind kind = CompletionObligationKind::OutputPort;
  unsigned ordinal = 0;
  IdIndex swNodeId = INVALID_ID;
  std::string description;
};

struct StaticConfigSlice {
  std::string name;
  std::string kind;
  IdIndex hwNode = INVALID_ID;
  uint32_t wordOffset = 0;
  uint32_t wordCount = 0;
  bool complete = true;
};

struct StaticConfigImage {
  std::vector<StaticConfigSlice> slices;
  std::vector<uint32_t> words;

  const StaticConfigSlice *findSliceByNameAndKind(const std::string &name,
                                                  const std::string &kind) const;
};

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_STATICMODELTYPES_H
