//===-- ADGBuilderPE.cpp - PE builder implementations -------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "ADGBuilderImpl.h"

namespace loom {
namespace adg {

//===----------------------------------------------------------------------===//
// PEBuilder
//===----------------------------------------------------------------------===//

PEBuilder::PEBuilder(ADGBuilder *builder, unsigned peId)
    : builder_(builder), peId_(peId) {}

PEBuilder &PEBuilder::setLatency(int16_t min, int16_t typical, int16_t max) {
  auto &pe = builder_->impl_->peDefs[peId_];
  pe.latMin = min;
  pe.latTyp = typical;
  pe.latMax = max;
  return *this;
}

PEBuilder &PEBuilder::setInterval(int16_t min, int16_t typical, int16_t max) {
  auto &pe = builder_->impl_->peDefs[peId_];
  pe.intMin = min;
  pe.intTyp = typical;
  pe.intMax = max;
  return *this;
}

PEBuilder &PEBuilder::setInputPorts(std::vector<Type> types) {
  builder_->impl_->peDefs[peId_].inputPorts = std::move(types);
  return *this;
}

PEBuilder &PEBuilder::setOutputPorts(std::vector<Type> types) {
  builder_->impl_->peDefs[peId_].outputPorts = std::move(types);
  return *this;
}

PEBuilder &PEBuilder::addOp(const std::string &opName) {
  builder_->impl_->peDefs[peId_].singleOp = opName;
  return *this;
}

PEBuilder &PEBuilder::setBodyMLIR(const std::string &mlirString) {
  builder_->impl_->peDefs[peId_].bodyMLIR = mlirString;
  return *this;
}

PEBuilder &PEBuilder::setInterfaceCategory(InterfaceCategory category) {
  builder_->impl_->peDefs[peId_].interface = category;
  return *this;
}

PEBuilder::operator PEHandle() const { return PEHandle{peId_}; }

//===----------------------------------------------------------------------===//
// ConstantPEBuilder
//===----------------------------------------------------------------------===//

ConstantPEBuilder::ConstantPEBuilder(ADGBuilder *builder, unsigned defId)
    : builder_(builder), defId_(defId) {}

ConstantPEBuilder &ConstantPEBuilder::setLatency(int16_t min, int16_t typical,
                                                  int16_t max) {
  auto &def = builder_->impl_->constantPEDefs[defId_];
  def.latMin = min;
  def.latTyp = typical;
  def.latMax = max;
  return *this;
}

ConstantPEBuilder &ConstantPEBuilder::setInterval(int16_t min, int16_t typical,
                                                   int16_t max) {
  auto &def = builder_->impl_->constantPEDefs[defId_];
  def.intMin = min;
  def.intTyp = typical;
  def.intMax = max;
  return *this;
}

ConstantPEBuilder &ConstantPEBuilder::setOutputType(Type type) {
  builder_->impl_->constantPEDefs[defId_].outputType = type;
  return *this;
}

ConstantPEBuilder::operator ConstantPEHandle() const {
  return ConstantPEHandle{defId_};
}

//===----------------------------------------------------------------------===//
// LoadPEBuilder
//===----------------------------------------------------------------------===//

LoadPEBuilder::LoadPEBuilder(ADGBuilder *builder, unsigned defId)
    : builder_(builder), defId_(defId) {}

LoadPEBuilder &LoadPEBuilder::setDataType(Type type) {
  builder_->impl_->loadPEDefs[defId_].dataType = type;
  return *this;
}

LoadPEBuilder &LoadPEBuilder::setInterfaceCategory(InterfaceCategory category) {
  builder_->impl_->loadPEDefs[defId_].interface = category;
  return *this;
}

LoadPEBuilder &LoadPEBuilder::setTagWidth(unsigned width) {
  builder_->impl_->loadPEDefs[defId_].tagWidth = width;
  return *this;
}

LoadPEBuilder &LoadPEBuilder::setQueueDepth(unsigned depth) {
  builder_->impl_->loadPEDefs[defId_].queueDepth = depth;
  return *this;
}

LoadPEBuilder &LoadPEBuilder::setHardwareType(HardwareType type) {
  builder_->impl_->loadPEDefs[defId_].hwType = type;
  return *this;
}

LoadPEBuilder::operator LoadPEHandle() const {
  return LoadPEHandle{defId_};
}

//===----------------------------------------------------------------------===//
// StorePEBuilder
//===----------------------------------------------------------------------===//

StorePEBuilder::StorePEBuilder(ADGBuilder *builder, unsigned defId)
    : builder_(builder), defId_(defId) {}

StorePEBuilder &StorePEBuilder::setDataType(Type type) {
  builder_->impl_->storePEDefs[defId_].dataType = type;
  return *this;
}

StorePEBuilder &StorePEBuilder::setInterfaceCategory(InterfaceCategory cat) {
  builder_->impl_->storePEDefs[defId_].interface = cat;
  return *this;
}

StorePEBuilder &StorePEBuilder::setTagWidth(unsigned width) {
  builder_->impl_->storePEDefs[defId_].tagWidth = width;
  return *this;
}

StorePEBuilder &StorePEBuilder::setQueueDepth(unsigned depth) {
  builder_->impl_->storePEDefs[defId_].queueDepth = depth;
  return *this;
}

StorePEBuilder &StorePEBuilder::setHardwareType(HardwareType type) {
  builder_->impl_->storePEDefs[defId_].hwType = type;
  return *this;
}

StorePEBuilder::operator StorePEHandle() const {
  return StorePEHandle{defId_};
}

//===----------------------------------------------------------------------===//
// SwitchBuilder
//===----------------------------------------------------------------------===//

SwitchBuilder::SwitchBuilder(ADGBuilder *builder, unsigned defId)
    : builder_(builder), defId_(defId) {}

SwitchBuilder &SwitchBuilder::setPortCount(unsigned inputs, unsigned outputs) {
  auto &def = builder_->impl_->switchDefs[defId_];
  def.numIn = inputs;
  def.numOut = outputs;
  return *this;
}

SwitchBuilder &
SwitchBuilder::setConnectivity(std::vector<std::vector<bool>> table) {
  builder_->impl_->switchDefs[defId_].connectivity = std::move(table);
  return *this;
}

SwitchBuilder &SwitchBuilder::setType(Type type) {
  builder_->impl_->switchDefs[defId_].portType = type;
  return *this;
}

SwitchBuilder::operator SwitchHandle() const {
  return SwitchHandle{defId_};
}

//===----------------------------------------------------------------------===//
// TemporalPEBuilder
//===----------------------------------------------------------------------===//

TemporalPEBuilder::TemporalPEBuilder(ADGBuilder *builder, unsigned defId)
    : builder_(builder), defId_(defId) {}

TemporalPEBuilder &TemporalPEBuilder::setNumRegisters(unsigned n) {
  builder_->impl_->temporalPEDefs[defId_].numRegisters = n;
  return *this;
}

TemporalPEBuilder &TemporalPEBuilder::setNumInstructions(unsigned n) {
  builder_->impl_->temporalPEDefs[defId_].numInstructions = n;
  return *this;
}

TemporalPEBuilder &TemporalPEBuilder::setRegFifoDepth(unsigned n) {
  builder_->impl_->temporalPEDefs[defId_].regFifoDepth = n;
  return *this;
}

TemporalPEBuilder &TemporalPEBuilder::setInterface(Type taggedType) {
  builder_->impl_->temporalPEDefs[defId_].interfaceType = taggedType;
  return *this;
}

TemporalPEBuilder &TemporalPEBuilder::addFU(PEHandle pe) {
  if (pe.id >= builder_->impl_->peDefs.size()) {
    fprintf(stderr, "error: addFU: invalid PE definition id %u\n", pe.id);
    std::exit(1);
  }
  builder_->impl_->temporalPEDefs[defId_].fuPEDefIndices.push_back(pe.id);
  return *this;
}

TemporalPEBuilder &TemporalPEBuilder::enableShareOperandBuffer(unsigned size) {
  auto &def = builder_->impl_->temporalPEDefs[defId_];
  def.shareModeB = true;
  def.shareBufferSize = size;
  return *this;
}

TemporalPEBuilder::operator TemporalPEHandle() const {
  return TemporalPEHandle{defId_};
}

//===----------------------------------------------------------------------===//
// TemporalSwitchBuilder
//===----------------------------------------------------------------------===//

TemporalSwitchBuilder::TemporalSwitchBuilder(ADGBuilder *builder,
                                              unsigned defId)
    : builder_(builder), defId_(defId) {}

TemporalSwitchBuilder &TemporalSwitchBuilder::setNumRouteTable(unsigned n) {
  builder_->impl_->temporalSwitchDefs[defId_].numRouteTable = n;
  return *this;
}

TemporalSwitchBuilder &
TemporalSwitchBuilder::setPortCount(unsigned inputs, unsigned outputs) {
  auto &def = builder_->impl_->temporalSwitchDefs[defId_];
  def.numIn = inputs;
  def.numOut = outputs;
  return *this;
}

TemporalSwitchBuilder &TemporalSwitchBuilder::setConnectivity(
    std::vector<std::vector<bool>> table) {
  builder_->impl_->temporalSwitchDefs[defId_].connectivity = std::move(table);
  return *this;
}

TemporalSwitchBuilder &TemporalSwitchBuilder::setInterface(Type taggedType) {
  builder_->impl_->temporalSwitchDefs[defId_].interfaceType = taggedType;
  return *this;
}

TemporalSwitchBuilder::operator TemporalSwitchHandle() const {
  return TemporalSwitchHandle{defId_};
}

//===----------------------------------------------------------------------===//
// MemoryBuilder
//===----------------------------------------------------------------------===//

MemoryBuilder::MemoryBuilder(ADGBuilder *builder, unsigned defId)
    : builder_(builder), defId_(defId) {}

MemoryBuilder &MemoryBuilder::setLoadPorts(unsigned count) {
  builder_->impl_->memoryDefs[defId_].ldCount = count;
  return *this;
}

MemoryBuilder &MemoryBuilder::setStorePorts(unsigned count) {
  builder_->impl_->memoryDefs[defId_].stCount = count;
  return *this;
}

MemoryBuilder &MemoryBuilder::setQueueDepth(unsigned depth) {
  builder_->impl_->memoryDefs[defId_].lsqDepth = depth;
  return *this;
}

MemoryBuilder &MemoryBuilder::setPrivate(bool isPrivate) {
  builder_->impl_->memoryDefs[defId_].isPrivate = isPrivate;
  return *this;
}

MemoryBuilder &MemoryBuilder::setShape(MemrefType shape) {
  builder_->impl_->memoryDefs[defId_].shape = shape;
  return *this;
}

MemoryBuilder::operator MemoryHandle() const {
  return MemoryHandle{defId_};
}

//===----------------------------------------------------------------------===//
// ExtMemoryBuilder
//===----------------------------------------------------------------------===//

ExtMemoryBuilder::ExtMemoryBuilder(ADGBuilder *builder, unsigned defId)
    : builder_(builder), defId_(defId) {}

ExtMemoryBuilder &ExtMemoryBuilder::setLoadPorts(unsigned count) {
  builder_->impl_->extMemoryDefs[defId_].ldCount = count;
  return *this;
}

ExtMemoryBuilder &ExtMemoryBuilder::setStorePorts(unsigned count) {
  builder_->impl_->extMemoryDefs[defId_].stCount = count;
  return *this;
}

ExtMemoryBuilder &ExtMemoryBuilder::setQueueDepth(unsigned depth) {
  builder_->impl_->extMemoryDefs[defId_].lsqDepth = depth;
  return *this;
}

ExtMemoryBuilder &ExtMemoryBuilder::setShape(MemrefType shape) {
  builder_->impl_->extMemoryDefs[defId_].shape = shape;
  return *this;
}

ExtMemoryBuilder::operator ExtMemoryHandle() const {
  return ExtMemoryHandle{defId_};
}

//===----------------------------------------------------------------------===//
// AddTagBuilder
//===----------------------------------------------------------------------===//

AddTagBuilder::AddTagBuilder(ADGBuilder *builder, unsigned defId)
    : builder_(builder), defId_(defId), instanceId_(std::make_shared<int>(-1)) {}

AddTagBuilder &AddTagBuilder::setValueType(Type type) {
  builder_->impl_->addTagDefs[defId_].valueType = type;
  return *this;
}

AddTagBuilder &AddTagBuilder::setTagType(Type type) {
  builder_->impl_->addTagDefs[defId_].tagType = type;
  return *this;
}

AddTagBuilder::operator InstanceHandle() const {
  if (*instanceId_ < 0) {
    *instanceId_ = (int)builder_->impl_->instances.size();
    builder_->impl_->instances.push_back(
        {ModuleKind::AddTag, defId_,
         builder_->impl_->addTagDefs[defId_].name});
  }
  return InstanceHandle{(unsigned)*instanceId_};
}

//===----------------------------------------------------------------------===//
// MapTagBuilder
//===----------------------------------------------------------------------===//

MapTagBuilder::MapTagBuilder(ADGBuilder *builder, unsigned defId)
    : builder_(builder), defId_(defId), instanceId_(std::make_shared<int>(-1)) {}

MapTagBuilder &MapTagBuilder::setValueType(Type type) {
  builder_->impl_->mapTagDefs[defId_].valueType = type;
  return *this;
}

MapTagBuilder &MapTagBuilder::setInputTagType(Type type) {
  builder_->impl_->mapTagDefs[defId_].inputTagType = type;
  return *this;
}

MapTagBuilder &MapTagBuilder::setOutputTagType(Type type) {
  builder_->impl_->mapTagDefs[defId_].outputTagType = type;
  return *this;
}

MapTagBuilder &MapTagBuilder::setTableSize(unsigned size) {
  builder_->impl_->mapTagDefs[defId_].tableSize = size;
  return *this;
}

MapTagBuilder::operator InstanceHandle() const {
  if (*instanceId_ < 0) {
    *instanceId_ = (int)builder_->impl_->instances.size();
    builder_->impl_->instances.push_back(
        {ModuleKind::MapTag, defId_,
         builder_->impl_->mapTagDefs[defId_].name});
  }
  return InstanceHandle{(unsigned)*instanceId_};
}

//===----------------------------------------------------------------------===//
// DelTagBuilder
//===----------------------------------------------------------------------===//

DelTagBuilder::DelTagBuilder(ADGBuilder *builder, unsigned defId)
    : builder_(builder), defId_(defId), instanceId_(std::make_shared<int>(-1)) {}

DelTagBuilder &DelTagBuilder::setInputType(Type type) {
  if (!type.isTagged()) {
    fprintf(stderr, "error: setInputType: del_tag requires a tagged type\n");
    std::exit(1);
  }
  builder_->impl_->delTagDefs[defId_].inputType = type;
  return *this;
}

DelTagBuilder::operator InstanceHandle() const {
  if (*instanceId_ < 0) {
    *instanceId_ = (int)builder_->impl_->instances.size();
    builder_->impl_->instances.push_back(
        {ModuleKind::DelTag, defId_,
         builder_->impl_->delTagDefs[defId_].name});
  }
  return InstanceHandle{(unsigned)*instanceId_};
}

//===----------------------------------------------------------------------===//
// FifoBuilder
//===----------------------------------------------------------------------===//

FifoBuilder::FifoBuilder(ADGBuilder *builder, unsigned defId)
    : builder_(builder), defId_(defId) {}

FifoBuilder &FifoBuilder::setDepth(unsigned depth) {
  builder_->impl_->fifoDefs[defId_].depth = depth;
  return *this;
}

FifoBuilder &FifoBuilder::setType(Type type) {
  builder_->impl_->fifoDefs[defId_].elementType = type;
  return *this;
}

FifoBuilder::operator FifoHandle() const {
  return FifoHandle{defId_};
}

} // namespace adg
} // namespace loom
