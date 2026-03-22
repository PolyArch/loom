#include "fcc/Simulator/CycleKernel.h"

#include "fcc/Simulator/SimTemporalPE.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <set>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace fcc {
namespace sim {

namespace {

constexpr int kInvalidModuleIndex = -1;
constexpr uint64_t kIdleCyclesForBoundary = 32;
constexpr unsigned kMaxCombIterations = 4;

bool hasOutstandingCompletionObligations(const StaticMappedModel &staticModel) {
  return !staticModel.getCompletionObligations().empty();
}

unsigned computePortStateSize(const StaticMappedModel &model) {
  unsigned maxPortId = 0;
  for (const auto &port : model.getPorts())
    maxPortId = std::max(maxPortId, port.portId);
  return model.getPorts().empty() ? 0 : (maxPortId + 1);
}

unsigned computeNodeStateSize(const StaticMappedModel &model) {
  unsigned maxNodeId = 0;
  for (const auto &module : model.getModules())
    maxNodeId = std::max(maxNodeId, module.hwNodeId);
  return model.getModules().empty() ? 0 : (maxNodeId + 1);
}

bool channelEquals(const SimChannel &lhs, const SimChannel &rhs) {
  return lhs.valid == rhs.valid && lhs.ready == rhs.ready &&
         lhs.data == rhs.data && lhs.tag == rhs.tag &&
         lhs.hasTag == rhs.hasTag && lhs.generation == rhs.generation;
}

int64_t getIntAttr(const StaticModuleDesc &module, const char *name,
                   int64_t defaultValue = 0) {
  for (const auto &attr : module.intAttrs) {
    if (attr.name == name)
      return attr.value;
  }
  return defaultValue;
}

std::string getStringAttr(const StaticModuleDesc &module, const char *name) {
  for (const auto &attr : module.strAttrs) {
    if (attr.name == name)
      return attr.value;
  }
  return {};
}

bool isTemporalInternalFunctionUnit(const StaticModuleDesc &module) {
  return module.kind == StaticModuleKind::FunctionUnit &&
         getStringAttr(module, "pe_kind") == "temporal_pe";
}

unsigned bitWidthForChoices(unsigned count) {
  if (count <= 1)
    return 0;
  unsigned value = count - 1u;
  unsigned bits = 0;
  while (value != 0) {
    ++bits;
    value >>= 1;
  }
  return bits;
}

unsigned getPortOrdinal(const StaticModuleDesc &module, IdIndex portId,
                        StaticPortDirection direction) {
  const std::vector<IdIndex> &ports =
      (direction == StaticPortDirection::Input) ? module.inputPorts
                                                : module.outputPorts;
  for (unsigned idx = 0; idx < ports.size(); ++idx) {
    if (ports[idx] == portId)
      return idx;
  }
  return std::numeric_limits<unsigned>::max();
}

unsigned getFunctionUnitConfigBitWidth(const StaticModuleDesc &module) {
  return static_cast<unsigned>(
      std::max<int64_t>(0, getIntAttr(module, "fu_config_bits", 0)));
}

uint64_t readBits(const std::vector<uint32_t> &words, unsigned &bitPos,
                  unsigned width) {
  uint64_t value = 0;
  for (unsigned bit = 0; bit < width; ++bit) {
    unsigned wordIdx = bitPos / 32;
    unsigned wordBit = bitPos % 32;
    if (wordIdx < words.size() && ((words[wordIdx] >> wordBit) & 1u) != 0)
      value |= (uint64_t{1} << bit);
    ++bitPos;
  }
  return value;
}

void packBits(std::vector<uint32_t> &words, unsigned &bitPos, uint64_t value,
              unsigned width) {
  if (width == 0)
    return;
  unsigned finalBit = bitPos + width;
  unsigned requiredWords = (finalBit + 31) / 32;
  if (words.size() < requiredWords)
    words.resize(requiredWords, 0);
  for (unsigned bit = 0; bit < width; ++bit) {
    if (((value >> bit) & 1u) == 0)
      continue;
    unsigned absBit = bitPos + bit;
    words[absBit / 32] |= (1u << (absBit % 32));
  }
  bitPos = finalBit;
}

std::vector<uint32_t> extractBitRange(const std::vector<uint32_t> &words,
                                      unsigned startBit, unsigned width) {
  std::vector<uint32_t> out;
  unsigned srcBit = startBit;
  unsigned dstBit = 0;
  for (unsigned remaining = width; remaining > 0;) {
    unsigned chunk = std::min(remaining, 32u);
    packBits(out, dstBit, readBits(words, srcBit, chunk), chunk);
    remaining -= chunk;
  }
  return out;
}

struct DecodedMuxField {
  uint64_t sel = 0;
  bool discard = false;
  bool disconnect = true;
};

struct DecodedSpatialPEConfig {
  bool enabled = false;
  unsigned opcode = 0;
  IdIndex activeFuId = INVALID_ID;
  std::vector<DecodedMuxField> inputFields;
  std::vector<DecodedMuxField> outputFields;
  std::vector<uint32_t> fuConfigWords;
};

struct SpatialPELayoutInfo {
  unsigned maxFuInputs = 0;
  unsigned maxFuOutputs = 0;
  unsigned maxFuConfigBits = 0;
};

SpatialPELayoutInfo computeSpatialPELayout(
    const StaticPEDesc &pe,
    const std::unordered_map<IdIndex, const StaticModuleDesc *> &moduleByNode) {
  SpatialPELayoutInfo layout;
  for (IdIndex fuId : pe.fuNodeIds) {
    auto it = moduleByNode.find(fuId);
    if (it == moduleByNode.end())
      continue;
    const StaticModuleDesc *module = it->second;
    layout.maxFuInputs =
        std::max<unsigned>(layout.maxFuInputs, module->inputPorts.size());
    layout.maxFuOutputs =
        std::max<unsigned>(layout.maxFuOutputs, module->outputPorts.size());
    layout.maxFuConfigBits = std::max<unsigned>(
        layout.maxFuConfigBits, getFunctionUnitConfigBitWidth(*module));
  }
  return layout;
}

DecodedSpatialPEConfig decodeSpatialPEConfig(
    const StaticPEDesc &pe, const SpatialPELayoutInfo &layout,
    const std::vector<uint32_t> &words) {
  DecodedSpatialPEConfig decoded;
  unsigned bitPos = 0;
  decoded.enabled = readBits(words, bitPos, 1) != 0;
  decoded.opcode = static_cast<unsigned>(
      readBits(words, bitPos, bitWidthForChoices(pe.fuNodeIds.size())));
  if (decoded.enabled && decoded.opcode < pe.fuNodeIds.size())
    decoded.activeFuId = pe.fuNodeIds[decoded.opcode];

  unsigned inputSelBits = bitWidthForChoices(pe.numInputPorts);
  decoded.inputFields.reserve(layout.maxFuInputs);
  for (unsigned idx = 0; idx < layout.maxFuInputs; ++idx) {
    DecodedMuxField field;
    field.sel = readBits(words, bitPos, inputSelBits);
    field.discard = readBits(words, bitPos, 1) != 0;
    field.disconnect = readBits(words, bitPos, 1) != 0;
    decoded.inputFields.push_back(field);
  }

  unsigned outputSelBits = bitWidthForChoices(pe.numOutputPorts);
  decoded.outputFields.reserve(layout.maxFuOutputs);
  for (unsigned idx = 0; idx < layout.maxFuOutputs; ++idx) {
    DecodedMuxField field;
    field.sel = readBits(words, bitPos, outputSelBits);
    field.discard = readBits(words, bitPos, 1) != 0;
    field.disconnect = readBits(words, bitPos, 1) != 0;
    decoded.outputFields.push_back(field);
  }

  decoded.fuConfigWords =
      extractBitRange(words, bitPos, layout.maxFuConfigBits);
  return decoded;
}

bool simDebugEnabled() {
  const char *env = std::getenv("FCC_SIM_DEBUG");
  return env && env[0] != '\0' && env[0] != '0';
}

unsigned computeCombIterationBudget(const StaticMappedModel &model) {
  unsigned structuralBound =
      static_cast<unsigned>(std::max<size_t>(model.getChannels().size(),
                                             model.getModules().size()));
  structuralBound = std::max<unsigned>(structuralBound, 1);
  return std::max(kMaxCombIterations, structuralBound);
}

const char *moduleKindName(StaticModuleKind kind) {
  switch (kind) {
  case StaticModuleKind::BoundaryInput:
    return "boundary_input";
  case StaticModuleKind::BoundaryOutput:
    return "boundary_output";
  case StaticModuleKind::FunctionUnit:
    return "function_unit";
  case StaticModuleKind::SpatialSwitch:
    return "spatial_sw";
  case StaticModuleKind::TemporalSwitch:
    return "temporal_sw";
  case StaticModuleKind::AddTag:
    return "add_tag";
  case StaticModuleKind::MapTag:
    return "map_tag";
  case StaticModuleKind::DelTag:
    return "del_tag";
  case StaticModuleKind::Fifo:
    return "fifo";
  case StaticModuleKind::Memory:
    return "memory";
  case StaticModuleKind::ExtMemory:
    return "extmemory";
  case StaticModuleKind::TemporalPE:
    return "temporal_pe";
  case StaticModuleKind::Unknown:
    return "unknown";
  }
  return "unknown";
}

} // namespace

CycleKernel::CycleKernel(const SimConfig &config) : config_(config) {}

bool CycleKernel::build(const StaticMappedModel &staticModel) {
  staticModel_ = staticModel;
  built_ = true;
  configured_ = false;
  externalMemoryMode_ = false;

  portState_.assign(computePortStateSize(staticModel_), {});
  edgeState_.assign(staticModel_.getChannels().size(), {});
  visibleInputEdge_.assign(portState_.size(), -1);
  inputSourcePort_.assign(portState_.size(), {});
  outputDestPorts_.assign(portState_.size(), {});
  inputChannelIndex_.assign(portState_.size(), {});
  outputChannelIndices_.assign(portState_.size(), {});
  outputFanoutState_.assign(portState_.size(), {});
  forcedReadyOutputPort_.assign(portState_.size(), 0);
  completedStoreRegions_.assign(staticModel_.getMemoryBindings().size(), 0);
  modules_.clear();

  unsigned maxNodeId = computeNodeStateSize(staticModel_);
  boundaryInputModuleIndex_.assign(maxNodeId, kInvalidModuleIndex);
  boundaryOutputModuleIndex_.assign(maxNodeId, kInvalidModuleIndex);

  for (size_t channelIdx = 0; channelIdx < staticModel_.getChannels().size();
       ++channelIdx) {
    const auto &channel = staticModel_.getChannels()[channelIdx];
    if (channel.dstPort < static_cast<IdIndex>(inputSourcePort_.size()))
      inputSourcePort_[channel.dstPort].push_back(channel.srcPort);
    if (channel.srcPort < static_cast<IdIndex>(outputDestPorts_.size()))
      outputDestPorts_[channel.srcPort].push_back(channel.dstPort);
    if (channel.dstPort != INVALID_ID &&
        channel.dstPort < static_cast<IdIndex>(inputChannelIndex_.size()))
      inputChannelIndex_[channel.dstPort].push_back(
          static_cast<unsigned>(channelIdx));
    if (channel.srcPort != INVALID_ID &&
        channel.srcPort < static_cast<IdIndex>(outputChannelIndices_.size()))
      outputChannelIndices_[channel.srcPort].push_back(
          static_cast<unsigned>(channelIdx));
  }

  modules_.reserve(staticModel_.getModules().size());
  for (const auto &moduleDesc : staticModel_.getModules()) {
    if (isTemporalInternalFunctionUnit(moduleDesc))
      continue;
    std::unique_ptr<SimModule> module = createSimModule(moduleDesc, staticModel_);
    module->hwNodeId = moduleDesc.hwNodeId;
    module->name = moduleDesc.name;
    module->kind = moduleDesc.kind;
    for (IdIndex portId : moduleDesc.inputPorts) {
      if (portId != INVALID_ID &&
          portId < static_cast<IdIndex>(portState_.size()))
        module->inputs.push_back(&portState_[portId]);
    }
    for (IdIndex portId : moduleDesc.outputPorts) {
      if (portId != INVALID_ID &&
          portId < static_cast<IdIndex>(portState_.size()))
        module->outputs.push_back(&portState_[portId]);
    }
    module->bindRuntimeServices(this);

    int moduleIndex = static_cast<int>(modules_.size());
    if (moduleDesc.kind == StaticModuleKind::BoundaryInput &&
        moduleDesc.hwNodeId < boundaryInputModuleIndex_.size()) {
      boundaryInputModuleIndex_[moduleDesc.hwNodeId] = moduleIndex;
    } else if (moduleDesc.kind == StaticModuleKind::BoundaryOutput &&
               moduleDesc.hwNodeId < boundaryOutputModuleIndex_.size()) {
      boundaryOutputModuleIndex_[moduleDesc.hwNodeId] = moduleIndex;
    }
    modules_.push_back(std::move(module));
  }

  syntheticModuleBegin_ = modules_.size();
  uint32_t nextSyntheticHwNodeId =
      static_cast<uint32_t>(computeNodeStateSize(staticModel_));
  for (const StaticPEDesc &pe : staticModel_.getPEs()) {
    if (pe.peKind != "temporal_pe")
      continue;
    std::unique_ptr<SimModule> module = createTemporalPEModule(pe, staticModel_);
    if (!module)
      continue;
    module->hwNodeId = nextSyntheticHwNodeId++;
    std::vector<IdIndex> inputPorts =
        temporalPEInputRepresentativePorts(pe, staticModel_);
    for (IdIndex portId : inputPorts) {
      if (portId != INVALID_ID &&
          portId < static_cast<IdIndex>(portState_.size()))
        module->inputs.push_back(&portState_[portId]);
    }
    std::vector<IdIndex> outputPorts =
        temporalPEOutputCandidatePorts(pe, staticModel_);
    for (IdIndex portId : outputPorts) {
      if (portId != INVALID_ID &&
          portId < static_cast<IdIndex>(portState_.size()))
        module->outputs.push_back(&portState_[portId]);
    }
    module->bindRuntimeServices(this);
    modules_.push_back(std::move(module));
  }

  resetExecution();
  return true;
}

bool CycleKernel::configure(const StaticConfigImage &configImage) {
  if (!built_)
    return false;
  configImage_ = configImage;
  configured_ = true;
  traceDocument_.version = 1;
  traceDocument_.traceKind = "fcc_cycle_trace";
  traceDocument_.producer = "fcc";
  traceDocument_.coreId = config_.coreId;
  configuredFunctionUnits_.clear();
  configuredModuleNodes_.clear();
  moduleConfigReadyCycle_.clear();
  moduleComponentName_.clear();
  moduleFunctionUnitName_.clear();
  configSliceTimings_.clear();

  std::unordered_map<IdIndex, const StaticModuleDesc *> moduleByNode;
  std::unordered_map<IdIndex, size_t> moduleIndexByNode;
  std::unordered_map<IdIndex, std::string> fuToPEName;
  std::unordered_set<IdIndex> temporalFUNodes;
  moduleByNode.reserve(staticModel_.getModules().size());
  for (size_t moduleIdx = 0; moduleIdx < staticModel_.getModules().size();
       ++moduleIdx) {
    const StaticModuleDesc &moduleDesc = staticModel_.getModules()[moduleIdx];
    moduleByNode[moduleDesc.hwNodeId] = &moduleDesc;
    moduleIndexByNode[moduleDesc.hwNodeId] = moduleIdx;
  }
  for (const StaticPEDesc &pe : staticModel_.getPEs()) {
    for (IdIndex fuNodeId : pe.fuNodeIds) {
      fuToPEName[fuNodeId] = pe.peName;
      if (pe.peKind == "temporal_pe")
        temporalFUNodes.insert(fuNodeId);
    }
  }

  traceDocument_.modules.clear();
  traceDocument_.modules.reserve(staticModel_.getModules().size());
  for (const StaticModuleDesc &module : staticModel_.getModules()) {
    TraceModuleInfo info;
    info.hwNodeId = module.hwNodeId;
    info.kind = moduleKindName(module.kind);
    info.name = module.name;
    if (module.kind == StaticModuleKind::FunctionUnit) {
      auto it = fuToPEName.find(module.hwNodeId);
      info.componentName = (it != fuToPEName.end()) ? it->second : module.name;
      info.functionUnitName = module.name;
    } else if (module.kind == StaticModuleKind::BoundaryInput) {
      info.componentName = "module_in";
      auto ordinal = staticModel_.getBoundaryInputOrdinal(module.hwNodeId);
      info.boundaryOrdinal =
          ordinal ? static_cast<int32_t>(*ordinal) : static_cast<int32_t>(-1);
    } else if (module.kind == StaticModuleKind::BoundaryOutput) {
      info.componentName = "module_out";
      auto ordinal = staticModel_.getBoundaryOutputOrdinal(module.hwNodeId);
      info.boundaryOrdinal =
          ordinal ? static_cast<int32_t>(*ordinal) : static_cast<int32_t>(-1);
    } else {
      info.componentName = module.name;
    }
    traceDocument_.modules.push_back(std::move(info));
  }

  for (const TraceModuleInfo &info : traceDocument_.modules) {
    moduleComponentName_[info.hwNodeId] = info.componentName;
    moduleFunctionUnitName_[info.hwNodeId] = info.functionUnitName;
  }

  configSliceTimings_.reserve(configImage_.slices.size());
  for (const StaticConfigSlice &slice : configImage_.slices) {
    ConfigSliceTiming timing;
    timing.name = slice.name;
    timing.kind = slice.kind;
    timing.hwNodeId = static_cast<uint32_t>(slice.hwNode);
    timing.wordOffset = slice.wordOffset;
    timing.wordCount = slice.wordCount;
    timing.startCycle = config_.configWordsPerCycle == 0
                            ? 0
                            : (slice.wordOffset / config_.configWordsPerCycle);
    timing.endCycle =
        (config_.configWordsPerCycle == 0 || slice.wordCount == 0)
            ? timing.startCycle
            : ((static_cast<uint64_t>(slice.wordOffset) + slice.wordCount +
                config_.configWordsPerCycle - 1) /
               config_.configWordsPerCycle);
    configSliceTimings_.push_back(std::move(timing));
  }

  std::unordered_map<std::string, DecodedSpatialPEConfig> spatialPEConfigs;
  std::unordered_map<std::string, std::vector<uint32_t>> temporalPEConfigWords;
  std::unordered_map<std::string, std::vector<IdIndex>> temporalPEInputReps;
  std::set<std::string> seenSpatialPENames;
  std::unordered_set<IdIndex> activeFunctionUnits;
  debugInterestingNodes_.clear();
  for (const StaticPEDesc &pe : staticModel_.getPEs()) {
    if (pe.peKind != "spatial_pe")
      continue;
    seenSpatialPENames.insert(pe.peName);
    SpatialPELayoutInfo layout = computeSpatialPELayout(pe, moduleByNode);
    std::vector<uint32_t> sliceWords;
    if (const auto *slice =
            configImage_.findSliceByNameAndKind(pe.peName, "spatial_pe")) {
      if (slice->wordOffset + slice->wordCount <= configImage_.words.size()) {
        sliceWords.assign(configImage_.words.begin() + slice->wordOffset,
                          configImage_.words.begin() + slice->wordOffset +
                              slice->wordCount);
      }
    }
    DecodedSpatialPEConfig decoded =
        decodeSpatialPEConfig(pe, layout, sliceWords);
    bool printSpatialConfig = decoded.enabled || pe.peName == "pe_3_1";
    if (simDebugEnabled() && printSpatialConfig) {
      std::cerr << "CycleKernel spatial_pe " << pe.peName
                << " enable=" << decoded.enabled
                << " opcode=" << decoded.opcode
                << " active_fu=" << decoded.activeFuId << "\n";
      for (unsigned idx = 0; idx < decoded.inputFields.size(); ++idx) {
        const auto &field = decoded.inputFields[idx];
        std::cerr << "  in_mux[" << idx << "] sel=" << field.sel
                  << " discard=" << field.discard
                  << " disconnect=" << field.disconnect << "\n";
      }
      for (unsigned idx = 0; idx < decoded.outputFields.size(); ++idx) {
        const auto &field = decoded.outputFields[idx];
        std::cerr << "  out_mux[" << idx << "] sel=" << field.sel
                  << " discard=" << field.discard
                  << " disconnect=" << field.disconnect << "\n";
      }
      std::cerr << "  fu_words=[";
      for (size_t idx = 0; idx < decoded.fuConfigWords.size(); ++idx) {
        if (idx)
          std::cerr << ", ";
        std::cerr << decoded.fuConfigWords[idx];
      }
      std::cerr << "]\n";
    }
    if (decoded.enabled && decoded.activeFuId != INVALID_ID)
      activeFunctionUnits.insert(decoded.activeFuId);
    spatialPEConfigs[pe.peName] = std::move(decoded);
  }
  configuredFunctionUnits_ = activeFunctionUnits;

  for (const StaticPEDesc &pe : staticModel_.getPEs()) {
    if (pe.peKind != "temporal_pe")
      continue;
    temporalPEInputReps[pe.peName] =
        temporalPEInputRepresentativePorts(pe, staticModel_);
    if (const auto *slice =
            configImage_.findSliceByNameAndKind(pe.peName, "temporal_pe")) {
      if (slice->wordOffset + slice->wordCount <= configImage_.words.size()) {
        temporalPEConfigWords[pe.peName] = std::vector<uint32_t>(
            configImage_.words.begin() + slice->wordOffset,
            configImage_.words.begin() + slice->wordOffset + slice->wordCount);
      }
    }
  }

  if (simDebugEnabled()) {
    std::cerr << "CycleKernel spatial_pe inventory:";
    for (const auto &name : seenSpatialPENames)
      std::cerr << " " << name;
    std::cerr << "\n";
  }

  inputSourcePort_.assign(portState_.size(), {});
  outputDestPorts_.assign(portState_.size(), {});
  inputChannelIndex_.assign(portState_.size(), {});
  outputChannelIndices_.assign(portState_.size(), {});
  visibleInputEdge_.assign(portState_.size(), -1);
  forcedReadyOutputPort_.assign(portState_.size(), 0);
  std::fill(edgeState_.begin(), edgeState_.end(), SimChannel());
  for (size_t channelIdx = 0; channelIdx < staticModel_.getChannels().size();
       ++channelIdx) {
    const StaticChannelDesc &channel = staticModel_.getChannels()[channelIdx];
    bool include = true;
    auto srcModuleIt = moduleByNode.find(channel.srcNode);
    auto dstModuleIt = moduleByNode.find(channel.dstNode);
    const StaticModuleDesc *srcModule =
        (srcModuleIt != moduleByNode.end()) ? srcModuleIt->second : nullptr;
    const StaticModuleDesc *dstModule =
        (dstModuleIt != moduleByNode.end()) ? dstModuleIt->second : nullptr;

    if (temporalFUNodes.count(channel.srcNode) != 0 ||
        temporalFUNodes.count(channel.dstNode) != 0) {
      bool keepTemporalIngress = false;
      bool keepTemporalEgress = false;
      if (temporalFUNodes.count(channel.dstNode) != 0 && channel.peInputIndex >= 0) {
        auto peIt = fuToPEName.find(channel.dstNode);
        if (peIt != fuToPEName.end()) {
          const auto repsIt = temporalPEInputReps.find(peIt->second);
          unsigned ingressIdx = static_cast<unsigned>(channel.peInputIndex);
          keepTemporalIngress =
              repsIt != temporalPEInputReps.end() &&
              ingressIdx < repsIt->second.size() &&
              repsIt->second[ingressIdx] == channel.dstPort;
        }
      }
      if (temporalFUNodes.count(channel.srcNode) != 0 && channel.peOutputIndex >= 0)
        keepTemporalEgress = true;
      include = keepTemporalIngress || keepTemporalEgress;
    }

    if (dstModule && dstModule->kind == StaticModuleKind::FunctionUnit) {
      std::string peName = getStringAttr(*dstModule, "pe_name");
      auto peConfigIt = spatialPEConfigs.find(peName);
      if (peConfigIt != spatialPEConfigs.end()) {
        const DecodedSpatialPEConfig &cfg = peConfigIt->second;
        if (!cfg.enabled || cfg.activeFuId != channel.dstNode) {
          include = false;
        } else if (channel.peInputIndex >= 0) {
          unsigned ordinal =
              getPortOrdinal(*dstModule, channel.dstPort,
                             StaticPortDirection::Input);
          if (ordinal >= cfg.inputFields.size()) {
            include = false;
          } else {
            const DecodedMuxField &field = cfg.inputFields[ordinal];
            include = !field.disconnect &&
                      field.sel == static_cast<uint64_t>(channel.peInputIndex);
          }
        }
      }
    }

    if (include && srcModule && srcModule->kind == StaticModuleKind::FunctionUnit) {
      std::string peName = getStringAttr(*srcModule, "pe_name");
      auto peConfigIt = spatialPEConfigs.find(peName);
      if (peConfigIt != spatialPEConfigs.end()) {
        const DecodedSpatialPEConfig &cfg = peConfigIt->second;
        if (!cfg.enabled || cfg.activeFuId != channel.srcNode) {
          include = false;
        } else if (channel.peOutputIndex >= 0) {
          unsigned ordinal =
              getPortOrdinal(*srcModule, channel.srcPort,
                             StaticPortDirection::Output);
          if (ordinal >= cfg.outputFields.size()) {
            include = false;
          } else {
            const DecodedMuxField &field = cfg.outputFields[ordinal];
            if (field.disconnect) {
              include = false;
            } else if (field.discard) {
              if (channel.srcPort != INVALID_ID &&
                  channel.srcPort < static_cast<IdIndex>(forcedReadyOutputPort_.size()))
                forcedReadyOutputPort_[channel.srcPort] = 1;
              include = false;
            } else {
              include = field.sel ==
                        static_cast<uint64_t>(channel.peOutputIndex);
            }
          }
        }
      }
    }

    bool touchesDebugModule =
        activeFunctionUnits.find(channel.srcNode) != activeFunctionUnits.end() ||
        activeFunctionUnits.find(channel.dstNode) != activeFunctionUnits.end() || channel.srcNode == 735 ||
        channel.dstNode == 735 || channel.srcNode == 736 ||
        channel.dstNode == 736;
    if (simDebugEnabled() && touchesDebugModule) {
      std::cerr << "CycleKernel channel srcNode=" << channel.srcNode
                   << " srcPort=" << channel.srcPort << " dstNode=" << channel.dstNode
                   << " dstPort=" << channel.dstPort
                   << " peIn=" << channel.peInputIndex
                   << " peOut=" << channel.peOutputIndex
                   << " include=" << include << "\n";
    }

    if (!include)
      continue;
    if (activeFunctionUnits.find(channel.srcNode) != activeFunctionUnits.end() ||
        activeFunctionUnits.find(channel.dstNode) != activeFunctionUnits.end() ||
        (srcModule && (srcModule->kind == StaticModuleKind::Memory ||
                       srcModule->kind == StaticModuleKind::ExtMemory)) ||
        (dstModule && (dstModule->kind == StaticModuleKind::Memory ||
                       dstModule->kind == StaticModuleKind::ExtMemory))) {
      debugInterestingNodes_.insert(channel.srcNode);
      debugInterestingNodes_.insert(channel.dstNode);
    }
    if (channel.dstPort != INVALID_ID &&
        channel.dstPort < static_cast<IdIndex>(inputSourcePort_.size()))
      inputSourcePort_[channel.dstPort].push_back(channel.srcPort);
    if (channel.srcPort != INVALID_ID &&
        channel.srcPort < static_cast<IdIndex>(outputDestPorts_.size()))
      outputDestPorts_[channel.srcPort].push_back(channel.dstPort);
    if (channel.dstPort != INVALID_ID &&
        channel.dstPort < static_cast<IdIndex>(inputChannelIndex_.size()))
      inputChannelIndex_[channel.dstPort].push_back(
          static_cast<unsigned>(channelIdx));
    if (channel.srcPort != INVALID_ID &&
        channel.srcPort < static_cast<IdIndex>(outputChannelIndices_.size()))
      outputChannelIndices_[channel.srcPort].push_back(
          static_cast<unsigned>(channelIdx));
  }

  for (size_t moduleIdx = 0; moduleIdx < modules_.size(); ++moduleIdx) {
    SimModule *module = modules_[moduleIdx].get();
    std::vector<uint32_t> words;

    auto moduleIt = moduleByNode.find(module->hwNodeId);
    if (moduleIt != moduleByNode.end()) {
      const StaticModuleDesc &moduleDesc = *moduleIt->second;
      if (moduleDesc.kind == StaticModuleKind::FunctionUnit) {
        std::string peName = getStringAttr(moduleDesc, "pe_name");
        auto peConfigIt = spatialPEConfigs.find(peName);
        if (peConfigIt != spatialPEConfigs.end() &&
            peConfigIt->second.enabled &&
            peConfigIt->second.activeFuId == moduleDesc.hwNodeId) {
          words = peConfigIt->second.fuConfigWords;
        } else if (const auto *slice = configImage_.findSliceByNameAndKind(
                       moduleDesc.name, moduleDesc.opKind)) {
          if (slice->wordOffset + slice->wordCount <= configImage_.words.size()) {
            words.assign(configImage_.words.begin() + slice->wordOffset,
                         configImage_.words.begin() + slice->wordOffset +
                             slice->wordCount);
          }
        }
      } else if (const auto *slice = configImage_.findSliceByNameAndKind(
                     moduleDesc.name, moduleDesc.opKind)) {
        if (slice->wordOffset + slice->wordCount <= configImage_.words.size()) {
          words.assign(configImage_.words.begin() + slice->wordOffset,
                       configImage_.words.begin() + slice->wordOffset +
                           slice->wordCount);
        }
      }
    } else {
      for (const auto &entry : temporalPEConfigWords) {
        if (module->name == entry.first) {
          words = entry.second;
          break;
        }
      }
    }

    module->reset();
    module->configure(words);

    bool configured = false;
    uint64_t readyCycle = 0;
    if (moduleIt != moduleByNode.end()) {
      const StaticModuleDesc &moduleDesc = *moduleIt->second;
      if (moduleDesc.kind == StaticModuleKind::BoundaryInput ||
          moduleDesc.kind == StaticModuleKind::BoundaryOutput) {
        configured = true;
      } else if (moduleDesc.kind == StaticModuleKind::FunctionUnit) {
        std::string peKind = getStringAttr(moduleDesc, "pe_kind");
        std::string peName = getStringAttr(moduleDesc, "pe_name");
        if (peKind == "spatial_pe") {
          configured = configuredFunctionUnits_.count(moduleDesc.hwNodeId) != 0;
          if (configured) {
            if (const auto *slice =
                    configImage_.findSliceByNameAndKind(peName, "spatial_pe")) {
              readyCycle = (config_.configWordsPerCycle == 0 ||
                            slice->wordCount == 0)
                               ? 0
                               : ((static_cast<uint64_t>(slice->wordOffset) +
                                   slice->wordCount +
                                   config_.configWordsPerCycle - 1) /
                                  config_.configWordsPerCycle);
            }
          }
        } else if (peKind == "temporal_pe") {
          if (const auto *slice =
                  configImage_.findSliceByNameAndKind(peName, "temporal_pe")) {
            configured = slice->wordCount != 0;
            readyCycle = (config_.configWordsPerCycle == 0 ||
                          slice->wordCount == 0)
                             ? 0
                             : ((static_cast<uint64_t>(slice->wordOffset) +
                                 slice->wordCount +
                                 config_.configWordsPerCycle - 1) /
                                config_.configWordsPerCycle);
          }
        } else if (const auto *slice = configImage_.findSliceByNameAndKind(
                       moduleDesc.name, moduleDesc.opKind)) {
          configured = slice->wordCount != 0;
          readyCycle = (config_.configWordsPerCycle == 0 ||
                        slice->wordCount == 0)
                           ? 0
                           : ((static_cast<uint64_t>(slice->wordOffset) +
                               slice->wordCount +
                               config_.configWordsPerCycle - 1) /
                              config_.configWordsPerCycle);
        }
      } else if (const auto *slice = configImage_.findSliceByNameAndKind(
                     moduleDesc.name, moduleDesc.opKind)) {
        configured = slice->wordCount != 0 || moduleDesc.kind == StaticModuleKind::Memory ||
                     moduleDesc.kind == StaticModuleKind::ExtMemory;
        readyCycle = (config_.configWordsPerCycle == 0 ||
                      slice->wordCount == 0)
                         ? 0
                         : ((static_cast<uint64_t>(slice->wordOffset) +
                             slice->wordCount +
                             config_.configWordsPerCycle - 1) /
                            config_.configWordsPerCycle);
      }
      if (configured) {
        configuredModuleNodes_.insert(moduleDesc.hwNodeId);
        moduleConfigReadyCycle_[moduleDesc.hwNodeId] = readyCycle;
      }
    } else if (!words.empty()) {
      configuredModuleNodes_.insert(module->hwNodeId);
      if (const auto *slice =
              configImage_.findSliceByNameAndKind(module->name, "temporal_pe")) {
        uint64_t readyCycle = (config_.configWordsPerCycle == 0 ||
                               slice->wordCount == 0)
                                  ? 0
                                  : ((static_cast<uint64_t>(slice->wordOffset) +
                                      slice->wordCount +
                                      config_.configWordsPerCycle - 1) /
                                     config_.configWordsPerCycle);
        moduleConfigReadyCycle_[module->hwNodeId] = readyCycle;
      }
    }
  }

  if (simDebugEnabled()) {
    for (const auto &moduleDesc : staticModel_.getModules()) {
      bool printModule =
          debugInterestingNodes_.find(moduleDesc.hwNodeId) !=
              debugInterestingNodes_.end() ||
          moduleDesc.kind == StaticModuleKind::Memory ||
          moduleDesc.kind == StaticModuleKind::ExtMemory;
      if (!printModule)
        continue;
      std::cerr << "CycleKernel connectivity " << moduleDesc.name << " node="
                   << moduleDesc.hwNodeId << "\n";
      for (IdIndex portId : moduleDesc.inputPorts) {
        std::cerr << "  in_port " << portId << " <-";
        if (portId != INVALID_ID &&
            portId < static_cast<IdIndex>(inputSourcePort_.size())) {
          for (IdIndex srcPort : inputSourcePort_[portId])
            std::cerr << " " << srcPort;
        }
        std::cerr << "\n";
      }
      for (IdIndex portId : moduleDesc.outputPorts) {
        std::cerr << "  out_port " << portId << " ->";
        if (portId != INVALID_ID &&
            portId < static_cast<IdIndex>(outputDestPorts_.size())) {
          for (IdIndex dstPort : outputDestPorts_[portId])
            std::cerr << " " << dstPort;
        }
        std::cerr << "\n";
      }
    }
  }

  resetExecution();
  return true;
}

void CycleKernel::resetExecution() {
  done_ = false;
  quiescent_ = false;
  deadlocked_ = false;
  currentCycle_ = 0;
  lastBoundaryReason_ = BoundaryReason::None;
  lastTransferCount_ = 0;
  lastActivityCount_ = 0;
  idleCycleStreak_ = 0;
  nextMemoryRequestId_ = 1;
  std::fill(portState_.begin(), portState_.end(), SimChannel());
  std::fill(edgeState_.begin(), edgeState_.end(), SimChannel());
  outstandingMemoryRequests_.clear();
  completedMemoryRequests_.clear();
  std::fill(completedStoreRegions_.begin(), completedStoreRegions_.end(), 0);
  memoryRegionPerf_.clear();
  loadRequestCount_ = 0;
  storeRequestCount_ = 0;
  loadBytes_ = 0;
  storeBytes_ = 0;
  activeCycles_ = 0;
  idleCycles_ = 0;
  fabricActiveCycles_ = 0;
  needMemIssueCycles_ = 0;
  waitMemRespCycles_ = 0;
  budgetBoundaryCount_ = 0;
  deadlockBoundaryCount_ = 0;
  maxInflightMemoryRequests_ = 0;
  traceDocument_.events.clear();
  traceDocument_.epochId = 0;
  traceDocument_.invocationId = 0;
  for (auto &module : modules_)
    module->reset();
}

void CycleKernel::resetAll() {
  built_ = false;
  configured_ = false;
  staticModel_ = StaticMappedModel();
  configImage_ = StaticConfigImage();
  portState_.clear();
  edgeState_.clear();
  visibleInputEdge_.clear();
  inputSourcePort_.clear();
  outputDestPorts_.clear();
  inputChannelIndex_.clear();
  outputChannelIndices_.clear();
  outputFanoutState_.clear();
  completedStoreRegions_.clear();
  modules_.clear();
  syntheticModuleBegin_ = 0;
  boundaryInputModuleIndex_.clear();
  boundaryOutputModuleIndex_.clear();
  configuredFunctionUnits_.clear();
  configuredModuleNodes_.clear();
  moduleConfigReadyCycle_.clear();
  moduleComponentName_.clear();
  moduleFunctionUnitName_.clear();
  configSliceTimings_.clear();
  resetExecution();
}

void CycleKernel::setInvocationContext(uint32_t epochId, uint64_t invocationId) {
  traceDocument_.epochId = epochId;
  traceDocument_.invocationId = invocationId;
}

void CycleKernel::appendKernelEvent(uint64_t cycle, SimPhase phase,
                                    uint32_t hwNodeId, EventKind kind,
                                    uint32_t arg0, uint32_t arg1) {
  TraceEvent event;
  event.cycle = cycle;
  event.phase = phase;
  event.coreId = config_.coreId;
  event.hwNodeId = hwNodeId;
  event.eventKind = kind;
  event.arg0 = arg0;
  event.arg1 = arg1;
  traceDocument_.events.push_back(event);
}

void CycleKernel::setInputTokens(unsigned boundaryOrdinal,
                                 const std::vector<SimToken> &tokens) {
  for (const auto &moduleDesc : staticModel_.getModules()) {
    if (moduleDesc.kind != StaticModuleKind::BoundaryInput)
      continue;
    auto ordinal = staticModel_.getBoundaryInputOrdinal(moduleDesc.hwNodeId);
    if (!ordinal || *ordinal != boundaryOrdinal)
      continue;
    if (moduleDesc.hwNodeId >= boundaryInputModuleIndex_.size())
      return;
    int moduleIndex = boundaryInputModuleIndex_[moduleDesc.hwNodeId];
    if (moduleIndex >= 0)
      modules_[moduleIndex]->setInputTokens(tokens);
    return;
  }
}

const std::vector<SimToken> &
CycleKernel::getOutputTokens(unsigned boundaryOrdinal) const {
  for (const auto &moduleDesc : staticModel_.getModules()) {
    if (moduleDesc.kind != StaticModuleKind::BoundaryOutput)
      continue;
    auto ordinal = staticModel_.getBoundaryOutputOrdinal(moduleDesc.hwNodeId);
    if (!ordinal || *ordinal != boundaryOrdinal)
      continue;
    if (moduleDesc.hwNodeId >= boundaryOutputModuleIndex_.size())
      break;
    int moduleIndex = boundaryOutputModuleIndex_[moduleDesc.hwNodeId];
    if (moduleIndex >= 0)
      return modules_[moduleIndex]->getCollectedTokens();
    break;
  }
  static const std::vector<SimToken> empty;
  return empty;
}

void CycleKernel::rebuildPortSignalsFromSnapshot(
    const std::vector<SimChannel> &snapshot) {
  for (const auto &port : staticModel_.getPorts()) {
    SimChannel &state = portState_[port.portId];
    if (port.direction == StaticPortDirection::Input) {
      if (port.portId != INVALID_ID &&
          port.portId < static_cast<IdIndex>(visibleInputEdge_.size()))
        visibleInputEdge_[port.portId] = -1;
      state.ready = false;
      state.valid = false;
      state.data = 0;
      state.tag = 0;
      state.hasTag = false;
      state.generation = 0;
      if (port.portId != INVALID_ID &&
          port.portId < static_cast<IdIndex>(inputChannelIndex_.size())) {
        for (unsigned edgeIdx : inputChannelIndex_[port.portId]) {
          if (edgeIdx >= staticModel_.getChannels().size())
            continue;
          const StaticChannelDesc &edge = staticModel_.getChannels()[edgeIdx];
          if (edge.srcPort == INVALID_ID ||
              edge.srcPort >= static_cast<IdIndex>(snapshot.size()))
            continue;
          const SimChannel &src = snapshot[edge.srcPort];
          bool visible = src.valid;
          if (edge.srcPort != INVALID_ID &&
              edge.srcPort < static_cast<IdIndex>(outputFanoutState_.size())) {
            const OutputFanoutState &fanout = outputFanoutState_[edge.srcPort];
            if (edge.srcPort < static_cast<IdIndex>(outputChannelIndices_.size())) {
              const auto &edgeIndices = outputChannelIndices_[edge.srcPort];
              for (size_t localIdx = 0; localIdx < edgeIndices.size(); ++localIdx) {
                if (edgeIndices[localIdx] != edgeIdx)
                  continue;
                if (fanout.generation == src.generation &&
                    localIdx < fanout.captured.size() &&
                    fanout.captured[localIdx] != 0)
                  visible = false;
                break;
              }
            }
          }
          if (!visible)
            continue;
          state.valid = true;
          state.data = src.data;
          state.tag = src.tag;
          state.hasTag = src.hasTag;
          state.generation = src.generation;
          if (port.portId != INVALID_ID &&
              port.portId < static_cast<IdIndex>(visibleInputEdge_.size()))
            visibleInputEdge_[port.portId] = static_cast<int>(edgeIdx);
          break;
        }
      }
    } else {
      state.valid = false;
      state.data = 0;
      state.tag = 0;
      state.hasTag = false;
      state.generation = 0;
      bool forcedReady = forcedReadyOutputPort_[port.portId] != 0;
      bool ready = true;
      if (!forcedReady) {
        uint64_t predictedGeneration =
            (port.portId != INVALID_ID &&
             port.portId < static_cast<IdIndex>(snapshot.size()))
                ? snapshot[port.portId].generation
                : 0;
        syncOutputFanoutState(port.portId, predictedGeneration);
        if (port.portId != INVALID_ID &&
            port.portId < static_cast<IdIndex>(outputChannelIndices_.size())) {
          const auto &edgeIndices = outputChannelIndices_[port.portId];
          OutputFanoutState &fanout = outputFanoutState_[port.portId];
          for (size_t localIdx = 0; localIdx < edgeIndices.size(); ++localIdx) {
            unsigned edgeIdx = edgeIndices[localIdx];
            if (edgeIdx >= edgeState_.size() || localIdx >= fanout.captured.size()) {
              ready = false;
              continue;
            }
            if (fanout.generation != 0 && fanout.captured[localIdx] != 0)
              continue;
            ready = ready && edgeCanAcceptNow(edgeIdx, snapshot);
          }
        }
      } else {
        ready = true;
      }
      state.ready = ready;
    }
  }
}

void CycleKernel::finalizePortSignals() {
  for (const auto &port : staticModel_.getPorts()) {
    if (port.direction == StaticPortDirection::Input) {
      continue;
    }

    bool forcedReady = forcedReadyOutputPort_[port.portId] != 0;
    bool ready = true;
    if (!forcedReady) {
      syncOutputFanoutState(port.portId, portState_[port.portId].generation);
      if (port.portId != INVALID_ID &&
          port.portId < static_cast<IdIndex>(outputChannelIndices_.size())) {
        const auto &edgeIndices = outputChannelIndices_[port.portId];
        OutputFanoutState &fanout = outputFanoutState_[port.portId];
        for (size_t localIdx = 0; localIdx < edgeIndices.size(); ++localIdx) {
          unsigned edgeIdx = edgeIndices[localIdx];
          if (edgeIdx >= edgeState_.size() || localIdx >= fanout.captured.size()) {
            ready = false;
            continue;
          }
          if (fanout.generation != 0 && fanout.captured[localIdx] != 0)
            continue;
          ready = ready && edgeCanAcceptNow(edgeIdx, portState_);
        }
      }
    } else {
      ready = true;
    }
  portState_[port.portId].ready = ready;
  }
}

bool CycleKernel::edgeCanAcceptNow(
    unsigned edgeIdx, const std::vector<SimChannel> &snapshot) const {
  if (edgeIdx >= staticModel_.getChannels().size())
    return false;
  const StaticChannelDesc &channel = staticModel_.getChannels()[edgeIdx];
  if (channel.dstPort == INVALID_ID ||
      channel.dstPort >= static_cast<IdIndex>(snapshot.size()))
    return false;
  if (!snapshot[channel.dstPort].ready)
    return false;
  if (channel.dstPort != INVALID_ID &&
      channel.dstPort < static_cast<IdIndex>(visibleInputEdge_.size())) {
    int selectedEdge = visibleInputEdge_[channel.dstPort];
    if (selectedEdge >= 0 && selectedEdge != static_cast<int>(edgeIdx))
      return false;
  }
  return true;
}

void CycleKernel::syncOutputFanoutState(IdIndex outputPortId,
                                        uint64_t generation) {
  if (outputPortId == INVALID_ID ||
      outputPortId >= static_cast<IdIndex>(outputFanoutState_.size()))
    return;
  OutputFanoutState &fanout = outputFanoutState_[outputPortId];
  size_t edgeCount =
      (outputPortId < static_cast<IdIndex>(outputChannelIndices_.size()))
          ? outputChannelIndices_[outputPortId].size()
          : 0;
  if (fanout.captured.size() != edgeCount)
    fanout.captured.assign(edgeCount, 0);
  if (generation == 0) {
    fanout.generation = 0;
    fanout.completionEmitted = false;
    std::fill(fanout.captured.begin(), fanout.captured.end(), 0);
    return;
  }
  if (fanout.generation != generation) {
    fanout.generation = generation;
    fanout.completionEmitted = false;
    std::fill(fanout.captured.begin(), fanout.captured.end(), 0);
  }
}

bool CycleKernel::completionObligationsSatisfied() const {
  for (const auto &obligation : staticModel_.getCompletionObligations()) {
    if (obligation.kind == CompletionObligationKind::OutputPort) {
      if (obligation.ordinal >= boundaryOutputModuleIndex_.size())
        return false;
      const auto &tokens = getOutputTokens(obligation.ordinal);
      if (tokens.empty())
        return false;
      continue;
    }
    if (obligation.kind == CompletionObligationKind::MemoryRegion &&
        (regionHasOutstandingRequests(obligation.ordinal) ||
         !memoryRegionCompletionObserved(obligation.ordinal)))
      return false;
  }
  return true;
}

bool CycleKernel::hardwareEmpty(std::string *details) const {
  std::vector<std::string> parts;

  if (!completedMemoryRequests_.empty() || !outstandingMemoryRequests_.empty()) {
    parts.push_back("memory(outstanding=" +
                    std::to_string(outstandingMemoryRequests_.size()) +
                    ", completed=" +
                    std::to_string(completedMemoryRequests_.size()) + ")");
  }

  size_t liveEdgeCount = 0;
  std::vector<std::string> edgeParts;
  for (size_t edgeIdx = 0; edgeIdx < edgeState_.size(); ++edgeIdx) {
    const SimChannel &state = edgeState_[edgeIdx];
    if (!state.valid)
      continue;
    ++liveEdgeCount;
    if (edgeParts.size() < 4) {
      const auto &edge = staticModel_.getChannels()[edgeIdx];
      edgeParts.push_back("edge#" + std::to_string(edgeIdx) + "(hw=" +
                          std::to_string(edge.hwEdgeId) + ", gen=" +
                          std::to_string(state.generation) + ")");
    }
  }
  if (liveEdgeCount != 0) {
    std::string text = "live_edges=" + std::to_string(liveEdgeCount);
    if (!edgeParts.empty()) {
      text += " [";
      for (size_t idx = 0; idx < edgeParts.size(); ++idx) {
        if (idx)
          text += ", ";
        text += edgeParts[idx];
      }
      text += "]";
    }
    parts.push_back(std::move(text));
  }

  std::vector<std::string> pendingModules;
  for (const auto &module : modules_) {
    if (!module->hasPendingWork())
      continue;
    if (pendingModules.size() < 6) {
      std::string text = module->name;
      std::string state = module->getDebugStateSummary();
      if (!state.empty())
        text += "{" + state + "}";
      pendingModules.push_back(std::move(text));
    }
  }
  if (!pendingModules.empty()) {
    std::string text = "pending_modules=" + std::to_string(pendingModules.size());
    text += " [";
    for (size_t idx = 0; idx < pendingModules.size(); ++idx) {
      if (idx)
        text += ", ";
      text += pendingModules[idx];
    }
    text += "]";
    parts.push_back(std::move(text));
  }

  if (details) {
    details->clear();
    for (size_t idx = 0; idx < parts.size(); ++idx) {
      if (idx)
        *details += "; ";
      *details += parts[idx];
    }
  }
  return parts.empty();
}

bool CycleKernel::validateSuccessfulTermination(std::string &error) const {
  error.clear();
  if (!completionObligationsSatisfied()) {
    error = "termination audit failed: software-visible completion obligations "
            "are not satisfied";
    return false;
  }
  std::string hardwareDetails;
  if (!hardwareEmpty(&hardwareDetails)) {
    error =
        "termination audit failed: invocation completed but hardware is not "
        "empty";
    if (!hardwareDetails.empty())
      error += " (" + hardwareDetails + ")";
    return false;
  }
  return true;
}

bool CycleKernel::hasPendingInternalWork() const {
  return hasPendingModuleOrMemoryWork();
}

void CycleKernel::rebuildVisibleEdgeSignals() {
  std::fill(edgeState_.begin(), edgeState_.end(), SimChannel());
  for (size_t edgeIdx = 0; edgeIdx < staticModel_.getChannels().size(); ++edgeIdx) {
    const StaticChannelDesc &edge = staticModel_.getChannels()[edgeIdx];
    if (edge.srcPort == INVALID_ID ||
        edge.srcPort >= static_cast<IdIndex>(portState_.size()))
      continue;
    const SimChannel &src = portState_[edge.srcPort];
    if (!src.valid)
      continue;

    bool visible = true;
    if (edge.srcPort < static_cast<IdIndex>(outputFanoutState_.size())) {
      const OutputFanoutState &fanout = outputFanoutState_[edge.srcPort];
      if (fanout.generation == src.generation &&
          edge.srcPort < static_cast<IdIndex>(outputChannelIndices_.size())) {
        const auto &edgeIndices = outputChannelIndices_[edge.srcPort];
        for (size_t localIdx = 0; localIdx < edgeIndices.size(); ++localIdx) {
          if (edgeIndices[localIdx] != static_cast<unsigned>(edgeIdx))
            continue;
          if (localIdx < fanout.captured.size() && fanout.captured[localIdx] != 0)
            visible = false;
          break;
        }
      }
    }
    if (!visible)
      continue;

    SimChannel &dst = edgeState_[edgeIdx];
    dst.valid = true;
    dst.data = src.data;
    dst.tag = src.tag;
    dst.hasTag = src.hasTag;
    dst.generation = src.generation;
  }
}

bool CycleKernel::hasPendingModuleOrMemoryWork() const {
  if (!completedMemoryRequests_.empty() || !outstandingMemoryRequests_.empty())
    return true;
  for (const auto &module : modules_) {
    if (module->hasPendingWork())
      return true;
  }
  return false;
}

bool CycleKernel::memoryRegionCompletionObserved(unsigned regionId) const {
  return regionId < completedStoreRegions_.size() &&
         completedStoreRegions_[regionId] != 0;
}

FinalStateSummary CycleKernel::getFinalStateSummary() const {
  FinalStateSummary summary;
  summary.obligationsSatisfied = completionObligationsSatisfied();
  summary.hardwareEmpty = hardwareEmpty(&summary.terminationAuditError);
  summary.quiescent = quiescent_;
  summary.done = done_;
  summary.deadlocked = deadlocked_;
  summary.idleCycleStreak = idleCycleStreak_;
  summary.outstandingMemoryRequestCount = outstandingMemoryRequests_.size();
  summary.completedMemoryResponseCount = completedMemoryRequests_.size();

  summary.livePorts.reserve(staticModel_.getPorts().size());
  for (const auto &port : staticModel_.getPorts()) {
    if (port.portId >= portState_.size())
      continue;
    const SimChannel &state = portState_[port.portId];
    if (!state.valid)
      continue;
    FinalStatePortSnapshot snap;
    snap.portId = port.portId;
    snap.parentNodeId = port.parentNodeId;
    snap.isInput = (port.direction == StaticPortDirection::Input);
    snap.valid = state.valid;
    snap.ready = state.ready;
    snap.data = state.data;
    snap.tag = state.tag;
    snap.hasTag = state.hasTag;
    snap.generation = state.generation;
    summary.livePorts.push_back(std::move(snap));
  }

  summary.liveEdges.reserve(staticModel_.getChannels().size());
  for (size_t edgeIdx = 0; edgeIdx < staticModel_.getChannels().size(); ++edgeIdx) {
    if (edgeIdx >= edgeState_.size())
      continue;
    const SimChannel &state = edgeState_[edgeIdx];
    if (!state.valid)
      continue;
    const StaticChannelDesc &edge = staticModel_.getChannels()[edgeIdx];
    FinalStateEdgeSnapshot snap;
    snap.edgeIndex = static_cast<uint32_t>(edgeIdx);
    snap.hwEdgeId = edge.hwEdgeId;
    snap.srcPort = edge.srcPort;
    snap.dstPort = edge.dstPort;
    snap.valid = state.valid;
    snap.ready = state.ready;
    snap.data = state.data;
    snap.tag = state.tag;
    snap.hasTag = state.hasTag;
    snap.generation = state.generation;
    summary.liveEdges.push_back(std::move(snap));
  }

  summary.pendingModules.reserve(modules_.size());
  summary.moduleSummaries.reserve(modules_.size());
  for (const auto &module : modules_) {
    bool pending = module->hasPendingWork();
    uint64_t collectedCount = module->getCollectedTokens().size();
    FinalStateModuleSnapshot snap;
    snap.hwNodeId = module->hwNodeId;
    snap.name = module->name;
    snap.kind = moduleKindName(module->kind);
    snap.hasPendingWork = pending;
    snap.collectedTokenCount = collectedCount;
    snap.logicalFireCount = module->getLogicalFireCount();
    snap.inputCaptureCount = module->getInputCaptureCount();
    snap.outputTransferCount = module->getOutputTransferCount();
    snap.debugState = module->getDebugStateSummary();
    snap.counters = module->getDebugCounters();
    bool hasLivePort = false;
    for (const SimChannel *ch : module->inputs)
      hasLivePort = hasLivePort || (ch != nullptr && ch->valid);
    for (const SimChannel *ch : module->outputs)
      hasLivePort = hasLivePort || (ch != nullptr && ch->valid);
    bool interesting =
        pending || collectedCount != 0 || snap.logicalFireCount != 0 ||
        snap.inputCaptureCount != 0 || snap.outputTransferCount != 0 ||
        !snap.debugState.empty() || !snap.counters.empty() || hasLivePort;
    if (interesting)
      summary.moduleSummaries.push_back(snap);
    if (pending)
      summary.pendingModules.push_back(std::move(snap));
  }

  return summary;
}

void CycleKernel::evaluateBoundaryState() {
  if (!built_ || !configured_) {
    quiescent_ = false;
    done_ = false;
    deadlocked_ = false;
    lastBoundaryReason_ = BoundaryReason::None;
    return;
  }

  bool idleThisCycle = lastTransferCount_ == 0 && lastActivityCount_ == 0 &&
                       outstandingMemoryRequests_.empty() &&
                       completedMemoryRequests_.empty();
  if (idleThisCycle)
    ++idleCycleStreak_;
  else
    idleCycleStreak_ = 0;

  bool pendingInternalWork = hasPendingInternalWork();
  if (externalMemoryMode_ && !outgoingMemoryRequests_.empty()) {
    quiescent_ = false;
    done_ = false;
    deadlocked_ = false;
    lastBoundaryReason_ = BoundaryReason::NeedMemIssue;
    return;
  }
  if (externalMemoryMode_ && outgoingMemoryRequests_.empty() &&
      !outstandingMemoryRequests_.empty() && completedMemoryRequests_.empty() &&
      idleThisCycle) {
    quiescent_ = false;
    done_ = false;
    deadlocked_ = false;
    lastBoundaryReason_ = BoundaryReason::WaitMemResp;
    return;
  }
  quiescent_ = idleCycleStreak_ >= kIdleCyclesForBoundary;
  bool obligationsSatisfied = completionObligationsSatisfied();
  done_ = obligationsSatisfied && quiescent_;
  deadlocked_ =
      !obligationsSatisfied && quiescent_ &&
      (pendingInternalWork || hasOutstandingCompletionObligations(staticModel_));

  if (done_)
    lastBoundaryReason_ = BoundaryReason::InvocationDone;
  else if (deadlocked_)
    lastBoundaryReason_ = BoundaryReason::Deadlock;
  else
    lastBoundaryReason_ = BoundaryReason::None;
}

void dumpBudgetHitDebug(const StaticMappedModel &staticModel,
                        const std::vector<std::unique_ptr<SimModule>> &modules,
                        const std::vector<SimChannel> &ports,
                        const std::vector<SimChannel> &edges,
                        size_t outstandingCount,
                        bool obligationsSatisfied, bool quiescent, bool done,
                        bool deadlocked, uint64_t cycle) {
  if (!simDebugEnabled())
    return;
  std::cerr << "CycleKernel budget-hit summary cycle=" << cycle
               << " obligationsSatisfied=" << obligationsSatisfied
               << " quiescent=" << quiescent << " done=" << done
               << " deadlocked=" << deadlocked
               << " outstanding_mem=" << outstandingCount << "\n";

  unsigned validPorts = 0;
  for (const auto &port : staticModel.getPorts()) {
    if (ports[port.portId].valid)
      ++validPorts;
  }
  std::cerr << "  valid_ports=" << validPorts << "\n";
  for (const auto &port : staticModel.getPorts()) {
    if (!ports[port.portId].valid)
      continue;
    std::cerr << "    port " << port.portId << " node=" << port.parentNodeId
              << " dir="
              << (port.direction == StaticPortDirection::Input ? "in" : "out")
              << " g=" << ports[port.portId].generation
              << " d=" << ports[port.portId].data
              << " t=" << ports[port.portId].tag
              << " ht=" << ports[port.portId].hasTag << "\n";
  }

  unsigned validEdges = 0;
  for (size_t edgeIdx = 0; edgeIdx < edges.size(); ++edgeIdx) {
    if (!edges[edgeIdx].valid)
      continue;
    ++validEdges;
    const StaticChannelDesc &channel = staticModel.getChannels()[edgeIdx];
    std::cerr << "    edge[" << edgeIdx << "] hw=" << channel.hwEdgeId << " "
              << channel.srcPort << "->" << channel.dstPort
              << " g=" << edges[edgeIdx].generation
              << " d=" << edges[edgeIdx].data << " t=" << edges[edgeIdx].tag
              << " ht=" << edges[edgeIdx].hasTag << "\n";
  }
  std::cerr << "  valid_edges=" << validEdges << "\n";

  for (const auto &module : modules) {
    bool interesting = module->hasPendingWork() || !module->getCollectedTokens().empty();
    if (!interesting) {
      for (const SimChannel *ch : module->inputs)
        interesting = interesting || ch->valid;
      for (const SimChannel *ch : module->outputs)
        interesting = interesting || ch->valid;
    }
    if (!interesting)
      continue;
    std::cerr << "  module hw=" << module->hwNodeId << " name="
                 << module->name << " kind="
                 << static_cast<unsigned>(module->kind)
                 << " pending=" << module->hasPendingWork()
                 << " collected=" << module->getCollectedTokens().size()
                 << "\n";
    module->debugDump(std::cerr);
    const StaticModuleDesc *moduleDesc =
        staticModel.findModule(static_cast<IdIndex>(module->hwNodeId));
    for (size_t i = 0; i < module->inputs.size(); ++i) {
      const SimChannel *ch = module->inputs[i];
      std::cerr << "    in" << i << " v=" << ch->valid << " r=" << ch->ready
                   << " d=" << ch->data << " t=" << ch->tag
                   << " ht=" << ch->hasTag << " g=" << ch->generation
                   << "\n";
      if (moduleDesc && i < moduleDesc->inputPorts.size()) {
        std::cerr << "      portId=" << moduleDesc->inputPorts[i] << "\n";
      }
    }
    for (size_t i = 0; i < module->outputs.size(); ++i) {
      const SimChannel *ch = module->outputs[i];
      std::cerr << "    out" << i << " v=" << ch->valid << " r=" << ch->ready
                   << " d=" << ch->data << " t=" << ch->tag
                   << " ht=" << ch->hasTag << " g=" << ch->generation
                   << "\n";
      if (moduleDesc && i < moduleDesc->outputPorts.size()) {
        std::cerr << "      portId=" << moduleDesc->outputPorts[i] << "\n";
      }
    }
  }
}

void CycleKernel::stepCycle() {
  if (!built_ || !configured_)
    return;

  for (auto &channel : portState_)
    channel.didTransfer = false;

  if (currentCycle_ == 0)
    appendKernelEvent(currentCycle_, SimPhase::Evaluate, 0,
                      EventKind::InvocationStart);

  retireReadyMemoryRequests();

  std::vector<SimChannel> snapshot = portState_;
  bool converged = false;
  unsigned combIterationBudget = computeCombIterationBudget(staticModel_);
  for (unsigned iter = 0; iter < combIterationBudget; ++iter) {
    rebuildPortSignalsFromSnapshot(snapshot);
    for (auto &module : modules_)
      module->evaluate();
    finalizePortSignals();

    bool stable = true;
    for (size_t idx = 0; idx < portState_.size(); ++idx) {
      if (!channelEquals(snapshot[idx], portState_[idx])) {
        stable = false;
        break;
      }
    }
    if (stable) {
      converged = true;
      break;
    }
    snapshot = portState_;
  }

  if (!converged) {
    if (simDebugEnabled()) {
      std::cerr << "CycleKernel non-convergence cycle=" << currentCycle_
                << " iter_budget=" << combIterationBudget << "\n";
      dumpBudgetHitDebug(staticModel_, modules_, portState_, edgeState_,
                         outstandingMemoryRequests_.size(),
                         completionObligationsSatisfied(), quiescent_, done_,
                         true, currentCycle_);
    }
    appendKernelEvent(currentCycle_, SimPhase::Commit, 0,
                      EventKind::DeviceError);
    quiescent_ = false;
    done_ = false;
    deadlocked_ = true;
    lastBoundaryReason_ = BoundaryReason::Deadlock;
    ++currentCycle_;
    return;
  }

  if (simDebugEnabled()) {
    for (const auto &port : staticModel_.getPorts()) {
      if (port.direction != StaticPortDirection::Output)
        continue;
      if (port.portId == INVALID_ID ||
          port.portId >= static_cast<IdIndex>(outputChannelIndices_.size()))
        continue;
      const auto &channels = outputChannelIndices_[port.portId];
      if (channels.size() != 1)
        continue;
      const StaticChannelDesc &edge = staticModel_.getChannels()[channels.front()];
      IdIndex dstPort = edge.dstPort;
      if (dstPort == INVALID_ID ||
          dstPort >= static_cast<IdIndex>(portState_.size()))
        continue;
      if (portState_[port.portId].ready != portState_[dstPort].ready) {
        std::cerr << "CycleKernel single-dst ready mismatch srcPort="
                     << port.portId << " dstPort=" << dstPort
                     << " srcReady=" << portState_[port.portId].ready
                     << " dstReady=" << portState_[dstPort].ready << "\n";
      }
    }
  }

  if (simDebugEnabled() &&
      (currentCycle_ < 2 || (currentCycle_ >= 20 && currentCycle_ < 27) ||
       (currentCycle_ >= 32 && currentCycle_ < 41) ||
       (currentCycle_ >= 130 && currentCycle_ < 151) ||
       (currentCycle_ >= 224 && currentCycle_ < 258))) {
    std::cerr << "CycleKernel debug cycle=" << currentCycle_ << "\n";
    for (size_t moduleIdx = 0; moduleIdx < modules_.size(); ++moduleIdx) {
      const auto &module = modules_[moduleIdx];
      bool interesting = false;
      for (const SimChannel *ch : module->inputs)
        interesting = interesting || ch->valid || ch->ready;
      for (const SimChannel *ch : module->outputs)
        interesting = interesting || ch->valid || ch->ready;
      interesting = interesting ||
                    debugInterestingNodes_.find(module->hwNodeId) !=
                        debugInterestingNodes_.end();
      if (!interesting)
        continue;
      std::cerr << "  module[" << moduleIdx << "] hw=" << module->hwNodeId
                   << " " << module->name << " kind="
                   << static_cast<unsigned>(module->kind) << "\n";
      module->debugDump(std::cerr);
      const StaticModuleDesc *moduleDesc =
          staticModel_.findModule(static_cast<IdIndex>(module->hwNodeId));
      for (size_t i = 0; i < module->inputs.size(); ++i) {
        const SimChannel *ch = module->inputs[i];
        std::cerr << "    in" << i << " v=" << ch->valid << " r=" << ch->ready
                     << " d=" << ch->data << " t=" << ch->tag
                     << " ht=" << ch->hasTag << " g=" << ch->generation
                     << "\n";
        if (moduleDesc && i < moduleDesc->inputPorts.size()) {
          IdIndex portId = moduleDesc->inputPorts[i];
          std::cerr << "      portId=" << portId;
          if (portId != INVALID_ID &&
              portId < static_cast<IdIndex>(inputSourcePort_.size())) {
            for (size_t srcIdx = 0; srcIdx < inputSourcePort_[portId].size(); ++srcIdx) {
              IdIndex srcPort = inputSourcePort_[portId][srcIdx];
              std::cerr << (srcIdx == 0 ? " srcPort=" : ",srcPort=") << srcPort;
              const StaticPortDesc *srcPortDesc = staticModel_.findPort(srcPort);
              const StaticModuleDesc *srcModule = srcPortDesc
                                                      ? staticModel_.findModule(
                                                            srcPortDesc->parentNodeId)
                                                      : nullptr;
              if (srcModule) {
                std::cerr << " srcNode=" << srcModule->hwNodeId << ":"
                          << srcModule->name << ":"
                          << moduleKindName(srcModule->kind);
              }
              if (portId < static_cast<IdIndex>(inputChannelIndex_.size()) &&
                  srcIdx < inputChannelIndex_[portId].size())
                std::cerr << " edge=" << inputChannelIndex_[portId][srcIdx];
            }
          }
          std::cerr << "\n";
        }
      }
      for (size_t i = 0; i < module->outputs.size(); ++i) {
        const SimChannel *ch = module->outputs[i];
        std::cerr << "    out" << i << " v=" << ch->valid << " r=" << ch->ready
                     << " d=" << ch->data << " t=" << ch->tag
                     << " ht=" << ch->hasTag << " g=" << ch->generation
                     << "\n";
        if (moduleDesc && i < moduleDesc->outputPorts.size()) {
          IdIndex portId = moduleDesc->outputPorts[i];
          std::cerr << "      portId=" << portId;
          if (portId != INVALID_ID &&
              portId < static_cast<IdIndex>(outputChannelIndices_.size())) {
            std::cerr << " dsts=[";
            bool first = true;
            for (unsigned edgeIdx : outputChannelIndices_[portId]) {
              IdIndex dstPort = staticModel_.getChannels()[edgeIdx].dstPort;
              if (!first)
                std::cerr << ", ";
              first = false;
              std::cerr << dstPort;
              const StaticPortDesc *dstPortDesc = staticModel_.findPort(dstPort);
              const StaticModuleDesc *dstModule = dstPortDesc
                                                      ? staticModel_.findModule(
                                                            dstPortDesc->parentNodeId)
                                                      : nullptr;
              if (dstModule) {
                std::cerr << "->" << dstModule->hwNodeId << ":"
                             << dstModule->name << ":"
                             << moduleKindName(dstModule->kind);
              }
              std::cerr << "(edge=" << edgeIdx << ")";
            }
            std::cerr << "]";
          }
          std::cerr << "\n";
        }
      }
    }
  }

  lastTransferCount_ = 0;
  for (const auto &port : staticModel_.getPorts()) {
    if (port.direction != StaticPortDirection::Output)
      continue;
    if (!portState_[port.portId].valid || portState_[port.portId].generation == 0)
      continue;
    syncOutputFanoutState(port.portId, portState_[port.portId].generation);
    bool allCaptured = true;
    if (port.portId != INVALID_ID &&
        port.portId < static_cast<IdIndex>(outputChannelIndices_.size())) {
      const auto &edgeIndices = outputChannelIndices_[port.portId];
      OutputFanoutState &fanout = outputFanoutState_[port.portId];
      for (size_t localIdx = 0; localIdx < edgeIndices.size(); ++localIdx) {
        unsigned edgeIdx = edgeIndices[localIdx];
        if (edgeIdx >= staticModel_.getChannels().size() ||
            localIdx >= fanout.captured.size()) {
          allCaptured = false;
          continue;
        }
        if (fanout.captured[localIdx] != 0)
          continue;
        if (!edgeCanAcceptNow(edgeIdx, portState_)) {
          allCaptured = false;
          continue;
        }
        const StaticChannelDesc &edge = staticModel_.getChannels()[edgeIdx];
        if (edge.dstPort != INVALID_ID &&
            edge.dstPort < static_cast<IdIndex>(portState_.size()))
          portState_[edge.dstPort].didTransfer = true;
        fanout.captured[localIdx] = 1;
        ++lastTransferCount_;
      }
      for (uint8_t captured : fanout.captured)
        allCaptured = allCaptured && (captured != 0);
    }
    if (allCaptured && !outputFanoutState_[port.portId].completionEmitted) {
      portState_[port.portId].didTransfer = true;
      if (port.portId != INVALID_ID &&
          port.portId < static_cast<IdIndex>(outputFanoutState_.size())) {
        outputFanoutState_[port.portId].completionEmitted = true;
      }
    }
  }
  rebuildVisibleEdgeSignals();

  size_t traceCountBeforeModules = traceDocument_.events.size();
  for (auto &module : modules_)
    module->commit();
  for (auto &module : modules_)
    module->collectTraceEvents(traceDocument_.events, currentCycle_);
  lastActivityCount_ =
      static_cast<uint64_t>(traceDocument_.events.size() - traceCountBeforeModules);

  evaluateBoundaryState();

  if (done_) {
    appendKernelEvent(currentCycle_, SimPhase::Commit, 0,
                      EventKind::InvocationDone);
  } else if (deadlocked_) {
    dumpBudgetHitDebug(staticModel_, modules_, portState_, edgeState_,
                       outstandingMemoryRequests_.size(),
                       completionObligationsSatisfied(), quiescent_, done_,
                       deadlocked_, currentCycle_);
    appendKernelEvent(currentCycle_, SimPhase::Commit, 0,
                      EventKind::DeviceError);
  }

  if (lastTransferCount_ != 0 || lastActivityCount_ != 0)
    ++activeCycles_;
  else
    ++idleCycles_;
  if (lastActivityCount_ != 0 || lastTransferCount_ != 0)
    ++fabricActiveCycles_;
  if (lastBoundaryReason_ == BoundaryReason::NeedMemIssue)
    ++needMemIssueCycles_;
  else if (lastBoundaryReason_ == BoundaryReason::WaitMemResp)
    ++waitMemRespCycles_;
  else if (lastBoundaryReason_ == BoundaryReason::Deadlock)
    ++deadlockBoundaryCount_;

  ++currentCycle_;
}

BoundaryReason CycleKernel::runUntilBoundary(uint64_t maxCycles) {
  if (!built_ || !configured_) {
    lastBoundaryReason_ = BoundaryReason::Deadlock;
    return lastBoundaryReason_;
  }
  for (uint64_t iter = 0; iter < maxCycles; ++iter) {
    stepCycle();
    if (lastBoundaryReason_ != BoundaryReason::None)
      return lastBoundaryReason_;
  }
  dumpBudgetHitDebug(staticModel_, modules_, portState_, edgeState_,
                     outstandingMemoryRequests_.size(),
                     completionObligationsSatisfied(), quiescent_, done_,
                     deadlocked_, currentCycle_);
  ++budgetBoundaryCount_;
  lastBoundaryReason_ = BoundaryReason::BudgetHit;
  return lastBoundaryReason_;
}

std::string CycleKernel::setMemoryRegionBacking(unsigned regionId, uint8_t *data,
                                                size_t sizeBytes) {
  std::string error;
  if (!bindMemoryRegion(regionId, data, sizeBytes, error))
    return error;
  return {};
}

std::string CycleKernel::bindExternalMemoryRegion(unsigned regionId,
                                                  uint64_t baseByteAddr,
                                                  size_t sizeBytes) {
  std::string error;
  if (boundMemoryRegions_.size() <= regionId)
    boundMemoryRegions_.resize(regionId + 1);
  boundMemoryRegions_[regionId].data = nullptr;
  boundMemoryRegions_[regionId].baseByteAddr = baseByteAddr;
  boundMemoryRegions_[regionId].sizeBytes = sizeBytes;
  boundMemoryRegions_[regionId].external = true;
  (void)error;
  return {};
}


} // namespace sim
} // namespace fcc
