#include "fcc/Simulator/CycleBackend.h"

#include "fcc/Simulator/FunctionalBackend.h"
#include "fcc/Simulator/SimFunctionUnit.h"
#include "fcc/Simulator/StaticModelBuilder.h"

#include <sstream>

namespace fcc {
namespace sim {

namespace {

std::string getStringAttr(const StaticModuleDesc &module, const char *name) {
  for (const auto &attr : module.strAttrs) {
    if (attr.name == name)
      return attr.value;
  }
  return {};
}

} // namespace

bool CycleSimulationBackend::modelSupportsKernelExecution() const {
  for (const auto &module : staticModel_.getModules()) {
    switch (module.kind) {
    case StaticModuleKind::BoundaryInput:
    case StaticModuleKind::BoundaryOutput:
    case StaticModuleKind::SpatialSwitch:
    case StaticModuleKind::TemporalSwitch:
    case StaticModuleKind::AddTag:
    case StaticModuleKind::MapTag:
    case StaticModuleKind::DelTag:
    case StaticModuleKind::Fifo:
      break;
    case StaticModuleKind::FunctionUnit:
      if (getStringAttr(module, "pe_kind") == "temporal_pe")
        break;
      if (!functionUnitModuleSupportedByCycleKernel(module))
        return false;
      break;
    case StaticModuleKind::Memory:
    case StaticModuleKind::ExtMemory:
    case StaticModuleKind::TemporalPE:
      break;
    case StaticModuleKind::Unknown:
      return false;
    }
  }
  return true;
}

CycleSimulationBackend::CycleSimulationBackend(const SimConfig &config)
    : config_(config), kernel_(config) {}

CycleSimulationBackend::~CycleSimulationBackend() = default;

std::string CycleSimulationBackend::connect() {
  if (fallback_)
    return fallback_->connect();
  return {};
}

std::string CycleSimulationBackend::buildFromMappedState(
    const Graph &dfg, const Graph &adg, const MappingState &mapping) {
  if (!buildStaticMappedModel(dfg, adg, mapping, {}, staticModel_))
    return "failed to build cycle backend static model";
  if (!kernel_.build(staticModel_))
    return "failed to build cycle kernel";
  useKernelExecution_ = modelSupportsKernelExecution();
  hasFallbackGraph_ = !useKernelExecution_;
  pendingInputs_.assign(staticModel_.getInputBindings().size(), {});
  collectedOutputs_.assign(staticModel_.getOutputBindings().size(), {});
  memoryBindings_.clear();
  if (!hasFallbackGraph_) {
    fallback_.reset();
    return {};
  }
  fallback_ = std::make_unique<FunctionalSimulationBackend>(config_);
  std::string err = fallback_->connect();
  if (!err.empty())
    return err;
  return fallback_->buildFromMappedState(dfg, adg, mapping);
}

std::string CycleSimulationBackend::buildFromMappedState(
    const Graph &dfg, const Graph &adg, const MappingState &mapping,
    llvm::ArrayRef<PEContainment> peContainment) {
  if (!buildStaticMappedModel(dfg, adg, mapping, peContainment, staticModel_))
    return "failed to build cycle backend static model";
  if (!kernel_.build(staticModel_))
    return "failed to build cycle kernel";
  useKernelExecution_ = modelSupportsKernelExecution();
  hasFallbackGraph_ = !useKernelExecution_;
  pendingInputs_.assign(staticModel_.getInputBindings().size(), {});
  collectedOutputs_.assign(staticModel_.getOutputBindings().size(), {});
  memoryBindings_.clear();
  if (!hasFallbackGraph_) {
    fallback_.reset();
    return {};
  }
  fallback_ = std::make_unique<FunctionalSimulationBackend>(config_);
  std::string err = fallback_->connect();
  if (!err.empty())
    return err;
  return fallback_->buildFromMappedState(dfg, adg, mapping, peContainment);
}

std::string
CycleSimulationBackend::buildFromStaticModel(const StaticMappedModel &model) {
  staticModel_ = model;
  if (!kernel_.build(staticModel_))
    return "failed to build cycle kernel";
  useKernelExecution_ = modelSupportsKernelExecution();
  hasFallbackGraph_ = false;
  fallback_.reset();
  pendingInputs_.assign(staticModel_.getInputBindings().size(), {});
  collectedOutputs_.assign(staticModel_.getOutputBindings().size(), {});
  memoryBindings_.clear();
  if (!useKernelExecution_)
    return "static-model execution requires cycle-kernel support";
  return {};
}

std::string
CycleSimulationBackend::buildFromStaticModel(StaticMappedModel &&model) {
  staticModel_ = std::move(model);
  if (!kernel_.build(staticModel_))
    return "failed to build cycle kernel";
  useKernelExecution_ = modelSupportsKernelExecution();
  hasFallbackGraph_ = false;
  fallback_.reset();
  pendingInputs_.assign(staticModel_.getInputBindings().size(), {});
  collectedOutputs_.assign(staticModel_.getOutputBindings().size(), {});
  memoryBindings_.clear();
  if (!useKernelExecution_)
    return "static-model execution requires cycle-kernel support";
  return {};
}

std::string
CycleSimulationBackend::loadConfig(const std::vector<uint8_t> &configBlob) {
  configImage_.words.clear();
  configImage_.slices.clear();
  configImage_.words.reserve((configBlob.size() + 3) / 4);
  for (size_t offset = 0; offset < configBlob.size(); offset += 4) {
    uint32_t word = 0;
    for (unsigned byte = 0; byte < 4 && offset + byte < configBlob.size();
         ++byte)
      word |= static_cast<uint32_t>(configBlob[offset + byte]) << (byte * 8);
    configImage_.words.push_back(word);
  }
  if (!kernel_.configure(configImage_))
    return "failed to configure cycle kernel";
  if (hasFallbackGraph_ && fallback_)
    return fallback_->loadConfig(configBlob);
  return {};
}

std::string CycleSimulationBackend::loadConfig(
    const std::vector<uint8_t> &configBlob,
    llvm::ArrayRef<fcc::ConfigGen::ConfigSlice> configSlices) {
  std::string err = loadConfig(configBlob);
  if (!err.empty())
    return err;
  configImage_.slices.clear();
  configImage_.slices.reserve(configSlices.size());
  for (const auto &slice : configSlices) {
    StaticConfigSlice staticSlice;
    staticSlice.name = slice.name;
    staticSlice.kind = slice.kind;
    staticSlice.hwNode = slice.hwNode;
    staticSlice.wordOffset = slice.wordOffset;
    staticSlice.wordCount = slice.wordCount;
    staticSlice.complete = slice.complete;
    configImage_.slices.push_back(std::move(staticSlice));
  }
  if (!kernel_.configure(configImage_))
    return "failed to configure cycle kernel";
  return {};
}

std::string CycleSimulationBackend::setInput(
    unsigned portIdx, const std::vector<uint64_t> &data,
    const std::vector<uint16_t> &tags) {
  if (portIdx < pendingInputs_.size()) {
    pendingInputs_[portIdx].clear();
    pendingInputs_[portIdx].reserve(data.size());
    for (size_t idx = 0; idx < data.size(); ++idx) {
      SimToken token;
      token.data = data[idx];
      token.hasTag = !tags.empty();
      token.tag = tags.empty() ? 0 : tags[idx];
      pendingInputs_[portIdx].push_back(token);
    }
  }
  if (hasFallbackGraph_ && fallback_)
    return fallback_->setInput(portIdx, data, tags);
  return {};
}

std::string CycleSimulationBackend::setExtMemoryBacking(unsigned regionId,
                                                        uint8_t *data,
                                                        size_t sizeBytes) {
  if (memoryBindings_.size() <= regionId)
    memoryBindings_.resize(regionId + 1);
  memoryBindings_[regionId].data = data;
  memoryBindings_[regionId].sizeBytes = sizeBytes;
  std::string kernelErr = kernel_.setMemoryRegionBacking(regionId, data, sizeBytes);
  if (!kernelErr.empty())
    return kernelErr;
  if (hasFallbackGraph_ && fallback_)
    return fallback_->setExtMemoryBacking(regionId, data, sizeBytes);
  return {};
}

bool CycleSimulationBackend::serviceOutgoingMemoryRequests(std::string &error) {
  for (const MemoryRequestRecord &request : kernel_.drainOutgoingMemoryRequests()) {
    if (request.regionId >= memoryBindings_.size() ||
        memoryBindings_[request.regionId].data == nullptr) {
      std::ostringstream oss;
      oss << "cycle backend missing memory backing for region "
          << request.regionId;
      error = oss.str();
      return false;
    }
    MemoryRegionBinding &binding = memoryBindings_[request.regionId];
    if (request.byteAddr + request.byteWidth > binding.sizeBytes) {
      std::ostringstream oss;
      oss << "cycle backend memory OOB for region " << request.regionId
          << " at byte " << request.byteAddr;
      error = oss.str();
      return false;
    }

    MemoryCompletion completion;
    completion.requestId = request.requestId;
    completion.kind = request.kind;
    completion.regionId = request.regionId;
    completion.ownerNodeId = request.ownerNodeId;
    completion.tag = request.tag;
    completion.hasTag = request.hasTag;
    if (request.kind == MemoryRequestKind::Load) {
      uint64_t value = 0;
      for (unsigned byte = 0; byte < request.byteWidth; ++byte) {
        value |= uint64_t(binding.data[request.byteAddr + byte]) << (byte * 8);
      }
      completion.data = value;
    } else {
      for (unsigned byte = 0; byte < request.byteWidth; ++byte) {
        binding.data[request.byteAddr + byte] =
            static_cast<uint8_t>((request.data >> (byte * 8)) & 0xffu);
      }
      completion.data = 0;
    }
    kernel_.pushMemoryCompletion(completion);
  }
  return true;
}

SimResult CycleSimulationBackend::invoke(uint32_t epochId,
                                         uint64_t invocationId) {
  if (useKernelExecution_) {
    kernel_.resetExecution();
    kernel_.setInvocationContext(epochId, invocationId);
    for (unsigned portIdx = 0; portIdx < pendingInputs_.size(); ++portIdx)
      kernel_.setInputTokens(portIdx, pendingInputs_[portIdx]);

    SimResult result;
    result.configCycles =
        config_.configWordsPerCycle == 0
            ? 0
            : ((static_cast<uint64_t>(configImage_.words.size()) +
                config_.configWordsPerCycle - 1) /
               config_.configWordsPerCycle);
    result.totalConfigWrites = configImage_.words.size();

    uint64_t remainingCycles = config_.maxCycles;
    while (true) {
      BoundaryReason reason = kernel_.runUntilBoundary(remainingCycles);
      uint64_t consumedCycles = kernel_.getCurrentCycle();
      remainingCycles =
          (consumedCycles >= config_.maxCycles) ? 0 : (config_.maxCycles - consumedCycles);

      if (reason == BoundaryReason::NeedMemIssue) {
        if (!serviceOutgoingMemoryRequests(result.errorMessage)) {
          result.success = false;
          result.termination = RunTermination::DeviceError;
          break;
        }
        if (remainingCycles == 0) {
          result.success = false;
          result.termination = RunTermination::Timeout;
          result.errorMessage = "cycle kernel budget exceeded";
          break;
        }
        continue;
      }
      if (reason == BoundaryReason::WaitMemResp) {
        result.success = false;
        result.termination = RunTermination::DeviceError;
        result.errorMessage =
            "cycle backend stalled waiting for memory response";
        break;
      }

      switch (reason) {
      case BoundaryReason::InvocationDone:
        result.success = true;
        result.termination = RunTermination::Completed;
        if (!kernel_.validateSuccessfulTermination(result.errorMessage)) {
          result.success = false;
          result.termination = RunTermination::DeviceError;
        }
        break;
      case BoundaryReason::BudgetHit:
        result.success = false;
        result.termination = RunTermination::Timeout;
        result.errorMessage = "cycle kernel budget exceeded";
        break;
      case BoundaryReason::Deadlock:
        result.success = false;
        result.termination = RunTermination::DeviceError;
        result.errorMessage = "cycle kernel deadlock";
        break;
      case BoundaryReason::None:
        result.success = false;
        result.termination = RunTermination::ContractError;
        result.errorMessage = "cycle kernel returned no boundary reason";
        break;
      case BoundaryReason::NeedMemIssue:
      case BoundaryReason::WaitMemResp:
        break;
      }
      break;
    }

    result.totalCycles = result.configCycles + kernel_.getCurrentCycle();
    result.acceleratorStats =
        kernel_.buildAcceleratorStats(/*configLoadStartCycle=*/0,
                                      result.configCycles,
                                      /*kernelLaunchCycle=*/result.configCycles,
                                      /*configDmaRequestCount=*/0,
                                      /*configDmaReadBytes=*/0);
    result.traceDocument = kernel_.getTraceDocument();
    result.traceEvents = result.traceDocument.events;
    result.finalState = kernel_.getFinalStateSummary();
    collectedOutputs_.assign(collectedOutputs_.size(), {});
    for (unsigned portIdx = 0; portIdx < collectedOutputs_.size(); ++portIdx)
      collectedOutputs_[portIdx] = kernel_.getOutputTokens(portIdx);
    return result;
  }
  if (!fallback_) {
    SimResult result;
    result.success = false;
    result.termination = RunTermination::ContractError;
    result.errorMessage = "fallback backend is unavailable";
    return result;
  }
  return fallback_->invoke(epochId, invocationId);
}

std::vector<uint64_t>
CycleSimulationBackend::getOutput(unsigned portIdx) const {
  if (useKernelExecution_ && portIdx < collectedOutputs_.size()) {
    std::vector<uint64_t> data;
    data.reserve(collectedOutputs_[portIdx].size());
    for (const SimToken &token : collectedOutputs_[portIdx])
      data.push_back(token.data);
    return data;
  }
  if (fallback_)
    return fallback_->getOutput(portIdx);
  return {};
}

std::vector<uint16_t>
CycleSimulationBackend::getOutputTags(unsigned portIdx) const {
  if (useKernelExecution_ && portIdx < collectedOutputs_.size()) {
    std::vector<uint16_t> tags;
    tags.reserve(collectedOutputs_[portIdx].size());
    for (const SimToken &token : collectedOutputs_[portIdx])
      tags.push_back(token.hasTag ? token.tag : 0);
    return tags;
  }
  if (fallback_)
    return fallback_->getOutputTags(portIdx);
  return {};
}

void CycleSimulationBackend::resetExecution() {
  kernel_.resetExecution();
  collectedOutputs_.assign(collectedOutputs_.size(), {});
  if (fallback_)
    fallback_->resetExecution();
}

void CycleSimulationBackend::resetAll() {
  configImage_.words.clear();
  configImage_.slices.clear();
  staticModel_ = StaticMappedModel();
  pendingInputs_.clear();
  collectedOutputs_.clear();
  memoryBindings_.clear();
  useKernelExecution_ = false;
  hasFallbackGraph_ = false;
  kernel_.resetAll();
  if (fallback_)
    fallback_->resetAll();
}

unsigned CycleSimulationBackend::getNumInputPorts() const {
  return fallback_->getNumInputPorts();
}

unsigned CycleSimulationBackend::getNumOutputPorts() const {
  return fallback_->getNumOutputPorts();
}

} // namespace sim
} // namespace fcc
