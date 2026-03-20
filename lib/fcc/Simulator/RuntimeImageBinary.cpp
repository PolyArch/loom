#include "fcc/Simulator/RuntimeImage.h"

#include <cstring>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace fcc {
namespace sim {

namespace {

constexpr char kRuntimeImageMagic[] = {'F', 'C', 'C', 'S', 'I', 'M', 'G', '1'};

template <typename T>
bool writeScalar(std::ofstream &out, T value) {
  static_assert(std::is_trivially_copyable_v<T>);
  out.write(reinterpret_cast<const char *>(&value), sizeof(T));
  return static_cast<bool>(out);
}

template <typename T>
bool readScalar(std::ifstream &in, T &value) {
  static_assert(std::is_trivially_copyable_v<T>);
  in.read(reinterpret_cast<char *>(&value), sizeof(T));
  return static_cast<bool>(in);
}

bool writeString(std::ofstream &out, const std::string &value) {
  uint64_t size = static_cast<uint64_t>(value.size());
  return writeScalar(out, size) &&
         (size == 0 ||
          static_cast<bool>(out.write(value.data(),
                                      static_cast<std::streamsize>(size))));
}

bool readString(std::ifstream &in, std::string &value) {
  uint64_t size = 0;
  if (!readScalar(in, size))
    return false;
  if (size > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
    return false;
  value.resize(static_cast<size_t>(size));
  return size == 0 ||
         static_cast<bool>(
             in.read(value.data(), static_cast<std::streamsize>(size)));
}

template <typename T>
bool writeVector(std::ofstream &out, const std::vector<T> &values) {
  uint64_t size = static_cast<uint64_t>(values.size());
  if (!writeScalar(out, size))
    return false;
  for (const T &value : values) {
    if (!writeScalar(out, value))
      return false;
  }
  return true;
}

template <typename T>
bool readVector(std::ifstream &in, std::vector<T> &values) {
  uint64_t size = 0;
  if (!readScalar(in, size))
    return false;
  if (size > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
    return false;
  values.clear();
  values.resize(static_cast<size_t>(size));
  for (T &value : values) {
    if (!readScalar(in, value))
      return false;
  }
  return true;
}

bool writeNamedIntAttrs(std::ofstream &out,
                        const std::vector<StaticIntAttr> &attrs) {
  uint64_t size = static_cast<uint64_t>(attrs.size());
  if (!writeScalar(out, size))
    return false;
  for (const auto &attr : attrs) {
    if (!writeString(out, attr.name) || !writeScalar(out, attr.value))
      return false;
  }
  return true;
}

bool readNamedIntAttrs(std::ifstream &in, std::vector<StaticIntAttr> &attrs) {
  uint64_t size = 0;
  if (!readScalar(in, size))
    return false;
  attrs.clear();
  attrs.resize(static_cast<size_t>(size));
  for (auto &attr : attrs) {
    if (!readString(in, attr.name) || !readScalar(in, attr.value))
      return false;
  }
  return true;
}

bool writeNamedStringAttrs(std::ofstream &out,
                           const std::vector<StaticStringAttr> &attrs) {
  uint64_t size = static_cast<uint64_t>(attrs.size());
  if (!writeScalar(out, size))
    return false;
  for (const auto &attr : attrs) {
    if (!writeString(out, attr.name) || !writeString(out, attr.value))
      return false;
  }
  return true;
}

bool readNamedStringAttrs(std::ifstream &in,
                          std::vector<StaticStringAttr> &attrs) {
  uint64_t size = 0;
  if (!readScalar(in, size))
    return false;
  attrs.clear();
  attrs.resize(static_cast<size_t>(size));
  for (auto &attr : attrs) {
    if (!readString(in, attr.name) || !readString(in, attr.value))
      return false;
  }
  return true;
}

template <typename AttrT, typename ElemT>
bool writeNamedArrayAttrs(std::ofstream &out, const std::vector<AttrT> &attrs) {
  uint64_t size = static_cast<uint64_t>(attrs.size());
  if (!writeScalar(out, size))
    return false;
  for (const auto &attr : attrs) {
    if (!writeString(out, attr.name) || !writeVector<ElemT>(out, attr.value))
      return false;
  }
  return true;
}

template <typename AttrT, typename ElemT>
bool readNamedArrayAttrs(std::ifstream &in, std::vector<AttrT> &attrs) {
  uint64_t size = 0;
  if (!readScalar(in, size))
    return false;
  attrs.clear();
  attrs.resize(static_cast<size_t>(size));
  for (auto &attr : attrs) {
    if (!readString(in, attr.name) || !readVector<ElemT>(in, attr.value))
      return false;
  }
  return true;
}

bool writeNamedStringArrayAttrs(std::ofstream &out,
                                const std::vector<StaticStringArrayAttr> &attrs) {
  uint64_t size = static_cast<uint64_t>(attrs.size());
  if (!writeScalar(out, size))
    return false;
  for (const auto &attr : attrs) {
    if (!writeString(out, attr.name))
      return false;
    uint64_t elemCount = static_cast<uint64_t>(attr.value.size());
    if (!writeScalar(out, elemCount))
      return false;
    for (const std::string &value : attr.value) {
      if (!writeString(out, value))
        return false;
    }
  }
  return true;
}

bool readNamedStringArrayAttrs(std::ifstream &in,
                               std::vector<StaticStringArrayAttr> &attrs) {
  uint64_t size = 0;
  if (!readScalar(in, size))
    return false;
  attrs.clear();
  attrs.resize(static_cast<size_t>(size));
  for (auto &attr : attrs) {
    if (!readString(in, attr.name))
      return false;
    uint64_t elemCount = 0;
    if (!readScalar(in, elemCount))
      return false;
    attr.value.resize(static_cast<size_t>(elemCount));
    for (std::string &value : attr.value) {
      if (!readString(in, value))
        return false;
    }
  }
  return true;
}

bool writeStaticModel(std::ofstream &out, const StaticMappedModel &model) {
  uint64_t portCount = static_cast<uint64_t>(model.getPorts().size());
  if (!writeScalar(out, portCount))
    return false;
  for (const auto &port : model.getPorts()) {
    if (!writeScalar(out, port.portId) ||
        !writeScalar(out, port.parentNodeId) ||
        !writeScalar(out, static_cast<uint8_t>(port.direction)) ||
        !writeScalar(out, port.isTagged) ||
        !writeScalar(out, port.isMemRef) ||
        !writeScalar(out, port.isNone) ||
        !writeScalar(out, static_cast<uint32_t>(port.valueWidth)) ||
        !writeScalar(out, static_cast<uint32_t>(port.tagWidth)))
      return false;
  }

  uint64_t moduleCount = static_cast<uint64_t>(model.getModules().size());
  if (!writeScalar(out, moduleCount))
    return false;
  for (const auto &module : model.getModules()) {
    if (!writeScalar(out, module.hwNodeId) ||
        !writeScalar(out, static_cast<uint8_t>(module.nodeKind)) ||
        !writeScalar(out, static_cast<uint8_t>(module.kind)) ||
        !writeString(out, module.name) ||
        !writeString(out, module.opKind) ||
        !writeString(out, module.resourceClass) ||
        !writeVector<IdIndex>(out, module.inputPorts) ||
        !writeVector<IdIndex>(out, module.outputPorts) ||
        !writeNamedIntAttrs(out, module.intAttrs) ||
        !writeNamedStringAttrs(out, module.strAttrs) ||
        !writeNamedArrayAttrs<StaticByteArrayAttr, int8_t>(out,
                                                           module.byteArrayAttrs) ||
        !writeNamedArrayAttrs<StaticIntArrayAttr, int64_t>(out,
                                                           module.intArrayAttrs) ||
        !writeNamedStringArrayAttrs(out, module.stringArrayAttrs))
      return false;
  }

  uint64_t channelCount = static_cast<uint64_t>(model.getChannels().size());
  if (!writeScalar(out, channelCount))
    return false;
  for (const auto &channel : model.getChannels()) {
    if (!writeScalar(out, channel.hwEdgeId) ||
        !writeScalar(out, channel.srcPort) ||
        !writeScalar(out, channel.dstPort) ||
        !writeScalar(out, channel.srcNode) ||
        !writeScalar(out, channel.dstNode) ||
        !writeScalar(out, channel.peInputIndex) ||
        !writeScalar(out, channel.peOutputIndex) ||
        !writeScalar(out, channel.touchesBoundaryInput) ||
        !writeScalar(out, channel.touchesBoundaryOutput))
      return false;
  }

  uint64_t peCount = static_cast<uint64_t>(model.getPEs().size());
  if (!writeScalar(out, peCount))
    return false;
  for (const auto &pe : model.getPEs()) {
    if (!writeString(out, pe.peName) ||
        !writeString(out, pe.peKind) ||
        !writeVector<IdIndex>(out, pe.fuNodeIds) ||
        !writeScalar(out, pe.row) ||
        !writeScalar(out, pe.col) ||
        !writeScalar(out, pe.numInputPorts) ||
        !writeScalar(out, pe.numOutputPorts) ||
        !writeScalar(out, pe.numInstruction) ||
        !writeScalar(out, pe.numRegister) ||
        !writeScalar(out, pe.regFifoDepth) ||
        !writeScalar(out, pe.tagWidth) ||
        !writeScalar(out, pe.enableShareOperandBuffer) ||
        !writeScalar(out, pe.operandBufferSize))
      return false;
  }

  auto writeInputBindings = [&](const auto &bindings) -> bool {
    uint64_t count = static_cast<uint64_t>(bindings.size());
    if (!writeScalar(out, count))
      return false;
    for (const auto &binding : bindings) {
      if (!writeScalar(out, binding.boundaryOrdinal) ||
          !writeScalar(out, binding.swNodeId))
        return false;
    }
    return true;
  };
  if (!writeInputBindings(model.getInputBindings()) ||
      !writeInputBindings(model.getOutputBindings()))
    return false;

  if (!writeVector<unsigned>(out, model.getBoundaryInputOrdinals()) ||
      !writeVector<unsigned>(out, model.getBoundaryOutputOrdinals()))
    return false;

  uint64_t memoryBindingCount =
      static_cast<uint64_t>(model.getMemoryBindings().size());
  if (!writeScalar(out, memoryBindingCount))
    return false;
  for (const auto &binding : model.getMemoryBindings()) {
    if (!writeScalar(out, binding.regionId) ||
        !writeScalar(out, binding.swNodeId) ||
        !writeScalar(out, binding.hwNodeId) ||
        !writeScalar(out, binding.startLane) ||
        !writeScalar(out, binding.endLane) ||
        !writeScalar(out, binding.elemSizeLog2) ||
        !writeScalar(out, binding.supportsLoad) ||
        !writeScalar(out, binding.supportsStore))
      return false;
  }

  uint64_t obligationCount =
      static_cast<uint64_t>(model.getCompletionObligations().size());
  if (!writeScalar(out, obligationCount))
    return false;
  for (const auto &obligation : model.getCompletionObligations()) {
    if (!writeScalar(out, static_cast<uint8_t>(obligation.kind)) ||
        !writeScalar(out, obligation.ordinal) ||
        !writeScalar(out, obligation.swNodeId) ||
        !writeString(out, obligation.description))
      return false;
  }

  return true;
}

bool readStaticModel(std::ifstream &in, StaticMappedModel &model) {
  model = StaticMappedModel();

  uint64_t portCount = 0;
  if (!readScalar(in, portCount))
    return false;
  auto &ports = model.mutablePorts();
  ports.resize(static_cast<size_t>(portCount));
  for (auto &port : ports) {
    uint8_t direction = 0;
    uint32_t valueWidth = 0;
    uint32_t tagWidth = 0;
    if (!readScalar(in, port.portId) ||
        !readScalar(in, port.parentNodeId) ||
        !readScalar(in, direction) ||
        !readScalar(in, port.isTagged) ||
        !readScalar(in, port.isMemRef) ||
        !readScalar(in, port.isNone) ||
        !readScalar(in, valueWidth) ||
        !readScalar(in, tagWidth))
      return false;
    port.direction = static_cast<StaticPortDirection>(direction);
    port.valueWidth = valueWidth;
    port.tagWidth = tagWidth;
  }

  uint64_t moduleCount = 0;
  if (!readScalar(in, moduleCount))
    return false;
  auto &modules = model.mutableModules();
  modules.resize(static_cast<size_t>(moduleCount));
  for (auto &module : modules) {
    uint8_t nodeKind = 0;
    uint8_t kind = 0;
    if (!readScalar(in, module.hwNodeId) ||
        !readScalar(in, nodeKind) ||
        !readScalar(in, kind) ||
        !readString(in, module.name) ||
        !readString(in, module.opKind) ||
        !readString(in, module.resourceClass) ||
        !readVector<IdIndex>(in, module.inputPorts) ||
        !readVector<IdIndex>(in, module.outputPorts) ||
        !readNamedIntAttrs(in, module.intAttrs) ||
        !readNamedStringAttrs(in, module.strAttrs) ||
        !readNamedArrayAttrs<StaticByteArrayAttr, int8_t>(in,
                                                          module.byteArrayAttrs) ||
        !readNamedArrayAttrs<StaticIntArrayAttr, int64_t>(in,
                                                          module.intArrayAttrs) ||
        !readNamedStringArrayAttrs(in, module.stringArrayAttrs))
      return false;
    module.nodeKind = static_cast<StaticGraphNodeKind>(nodeKind);
    module.kind = static_cast<StaticModuleKind>(kind);
  }

  uint64_t channelCount = 0;
  if (!readScalar(in, channelCount))
    return false;
  auto &channels = model.mutableChannels();
  channels.resize(static_cast<size_t>(channelCount));
  for (auto &channel : channels) {
    if (!readScalar(in, channel.hwEdgeId) ||
        !readScalar(in, channel.srcPort) ||
        !readScalar(in, channel.dstPort) ||
        !readScalar(in, channel.srcNode) ||
        !readScalar(in, channel.dstNode) ||
        !readScalar(in, channel.peInputIndex) ||
        !readScalar(in, channel.peOutputIndex) ||
        !readScalar(in, channel.touchesBoundaryInput) ||
        !readScalar(in, channel.touchesBoundaryOutput))
      return false;
  }

  uint64_t peCount = 0;
  if (!readScalar(in, peCount))
    return false;
  auto &pes = model.mutablePEs();
  pes.resize(static_cast<size_t>(peCount));
  for (auto &pe : pes) {
    if (!readString(in, pe.peName) ||
        !readString(in, pe.peKind) ||
        !readVector<IdIndex>(in, pe.fuNodeIds) ||
        !readScalar(in, pe.row) ||
        !readScalar(in, pe.col) ||
        !readScalar(in, pe.numInputPorts) ||
        !readScalar(in, pe.numOutputPorts) ||
        !readScalar(in, pe.numInstruction) ||
        !readScalar(in, pe.numRegister) ||
        !readScalar(in, pe.regFifoDepth) ||
        !readScalar(in, pe.tagWidth) ||
        !readScalar(in, pe.enableShareOperandBuffer) ||
        !readScalar(in, pe.operandBufferSize))
      return false;
  }

  auto readBoundaryBindings = [&](auto &bindings) -> bool {
    uint64_t count = 0;
    if (!readScalar(in, count))
      return false;
    bindings.resize(static_cast<size_t>(count));
    for (auto &binding : bindings) {
      if (!readScalar(in, binding.boundaryOrdinal) ||
          !readScalar(in, binding.swNodeId))
        return false;
    }
    return true;
  };
  if (!readBoundaryBindings(model.mutableInputBindings()) ||
      !readBoundaryBindings(model.mutableOutputBindings()))
    return false;

  if (!readVector<unsigned>(in, model.mutableBoundaryInputOrdinals()) ||
      !readVector<unsigned>(in, model.mutableBoundaryOutputOrdinals()))
    return false;

  uint64_t memoryBindingCount = 0;
  if (!readScalar(in, memoryBindingCount))
    return false;
  auto &memoryBindings = model.mutableMemoryBindings();
  memoryBindings.resize(static_cast<size_t>(memoryBindingCount));
  for (auto &binding : memoryBindings) {
    if (!readScalar(in, binding.regionId) ||
        !readScalar(in, binding.swNodeId) ||
        !readScalar(in, binding.hwNodeId) ||
        !readScalar(in, binding.startLane) ||
        !readScalar(in, binding.endLane) ||
        !readScalar(in, binding.elemSizeLog2) ||
        !readScalar(in, binding.supportsLoad) ||
        !readScalar(in, binding.supportsStore))
      return false;
  }

  uint64_t obligationCount = 0;
  if (!readScalar(in, obligationCount))
    return false;
  auto &obligations = model.mutableCompletionObligations();
  obligations.resize(static_cast<size_t>(obligationCount));
  for (auto &obligation : obligations) {
    uint8_t kind = 0;
    if (!readScalar(in, kind) ||
        !readScalar(in, obligation.ordinal) ||
        !readScalar(in, obligation.swNodeId) ||
        !readString(in, obligation.description))
      return false;
    obligation.kind = static_cast<CompletionObligationKind>(kind);
  }

  return true;
}

bool writeConfigImage(std::ofstream &out, const StaticConfigImage &configImage) {
  uint64_t sliceCount = static_cast<uint64_t>(configImage.slices.size());
  if (!writeScalar(out, sliceCount))
    return false;
  for (const auto &slice : configImage.slices) {
    if (!writeString(out, slice.name) ||
        !writeString(out, slice.kind) ||
        !writeScalar(out, slice.hwNode) ||
        !writeScalar(out, slice.wordOffset) ||
        !writeScalar(out, slice.wordCount) ||
        !writeScalar(out, slice.complete))
      return false;
  }
  return writeVector<uint32_t>(out, configImage.words);
}

bool readConfigImage(std::ifstream &in, StaticConfigImage &configImage) {
  uint64_t sliceCount = 0;
  if (!readScalar(in, sliceCount))
    return false;
  configImage.slices.resize(static_cast<size_t>(sliceCount));
  for (auto &slice : configImage.slices) {
    if (!readString(in, slice.name) ||
        !readString(in, slice.kind) ||
        !readScalar(in, slice.hwNode) ||
        !readScalar(in, slice.wordOffset) ||
        !readScalar(in, slice.wordCount) ||
        !readScalar(in, slice.complete))
      return false;
  }
  return readVector<uint32_t>(in, configImage.words);
}

bool writeControlImage(std::ofstream &out,
                       const RuntimeControlImage &controlImage) {
  if (!writeScalar(out, controlImage.startTokenPort))
    return false;

  uint64_t scalarCount = static_cast<uint64_t>(controlImage.scalarBindings.size());
  if (!writeScalar(out, scalarCount))
    return false;
  for (const auto &binding : controlImage.scalarBindings) {
    if (!writeScalar(out, binding.slot) ||
        !writeScalar(out, binding.argIndex) ||
        !writeScalar(out, binding.portIdx))
      return false;
  }

  uint64_t memoryCount = static_cast<uint64_t>(controlImage.memoryBindings.size());
  if (!writeScalar(out, memoryCount))
    return false;
  for (const auto &binding : controlImage.memoryBindings) {
    if (!writeScalar(out, binding.slot) ||
        !writeScalar(out, binding.memrefArgIndex) ||
        !writeScalar(out, binding.regionId) ||
        !writeScalar(out, binding.elemSizeLog2))
      return false;
  }

  uint64_t outputCount = static_cast<uint64_t>(controlImage.outputBindings.size());
  if (!writeScalar(out, outputCount))
    return false;
  for (const auto &binding : controlImage.outputBindings) {
    if (!writeScalar(out, binding.slot) ||
        !writeScalar(out, binding.resultIndex) ||
        !writeScalar(out, binding.portIdx))
      return false;
  }
  return true;
}

bool readControlImage(std::ifstream &in, RuntimeControlImage &controlImage) {
  if (!readScalar(in, controlImage.startTokenPort))
    return false;

  uint64_t scalarCount = 0;
  if (!readScalar(in, scalarCount))
    return false;
  controlImage.scalarBindings.resize(static_cast<size_t>(scalarCount));
  for (auto &binding : controlImage.scalarBindings) {
    if (!readScalar(in, binding.slot) ||
        !readScalar(in, binding.argIndex) ||
        !readScalar(in, binding.portIdx))
      return false;
  }

  uint64_t memoryCount = 0;
  if (!readScalar(in, memoryCount))
    return false;
  controlImage.memoryBindings.resize(static_cast<size_t>(memoryCount));
  for (auto &binding : controlImage.memoryBindings) {
    if (!readScalar(in, binding.slot) ||
        !readScalar(in, binding.memrefArgIndex) ||
        !readScalar(in, binding.regionId) ||
        !readScalar(in, binding.elemSizeLog2))
      return false;
  }

  uint64_t outputCount = 0;
  if (!readScalar(in, outputCount))
    return false;
  controlImage.outputBindings.resize(static_cast<size_t>(outputCount));
  for (auto &binding : controlImage.outputBindings) {
    if (!readScalar(in, binding.slot) ||
        !readScalar(in, binding.resultIndex) ||
        !readScalar(in, binding.portIdx))
      return false;
  }
  return true;
}

} // namespace

bool writeRuntimeImageBinary(const RuntimeImage &image, const std::string &path,
                             std::string &error) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    error = "cannot open runtime image binary for write";
    return false;
  }
  out.write(kRuntimeImageMagic, sizeof(kRuntimeImageMagic));
  uint32_t version = 2;
  if (!writeScalar(out, version) ||
      !writeStaticModel(out, image.staticModel) ||
      !writeConfigImage(out, image.configImage) ||
      !writeControlImage(out, image.controlImage)) {
    error = "failed to serialize runtime image binary";
    return false;
  }
  return true;
}

bool loadRuntimeImageBinary(const std::string &path, RuntimeImage &image,
                            std::string &error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    error = "cannot open runtime image binary";
    return false;
  }
  char magic[sizeof(kRuntimeImageMagic)] = {};
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, kRuntimeImageMagic, sizeof(magic)) != 0) {
    error = "unsupported runtime image binary header";
    return false;
  }
  uint32_t version = 0;
  if (!readScalar(in, version) || version != 2) {
    error = "unsupported runtime image binary version";
    return false;
  }
  RuntimeImage loaded;
  if (!readStaticModel(in, loaded.staticModel) ||
      !readConfigImage(in, loaded.configImage) ||
      !readControlImage(in, loaded.controlImage)) {
    error = "runtime image binary payload is malformed";
    return false;
  }
  image = std::move(loaded);
  return true;
}

} // namespace sim
} // namespace fcc
