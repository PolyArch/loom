#include "fcc/Simulator/RuntimeImage.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace fcc {
namespace sim {

namespace {

using llvm::json::Array;
using llvm::json::Object;
using llvm::json::Value;

template <typename T>
static Value toJsonArray(const std::vector<T> &values) {
  Array array;
  array.reserve(values.size());
  for (const T &value : values)
    array.push_back(value);
  return Value(std::move(array));
}

static Value toJson(const StaticIntAttr &attr) {
  Object object;
  object["name"] = attr.name;
  object["value"] = attr.value;
  return Value(std::move(object));
}

static Value toJson(const StaticStringAttr &attr) {
  Object object;
  object["name"] = attr.name;
  object["value"] = attr.value;
  return Value(std::move(object));
}

static Value toJson(const StaticByteArrayAttr &attr) {
  Object object;
  object["name"] = attr.name;
  Array values;
  values.reserve(attr.value.size());
  for (int8_t value : attr.value)
    values.push_back(static_cast<int64_t>(value));
  object["value"] = std::move(values);
  return Value(std::move(object));
}

static Value toJson(const StaticIntArrayAttr &attr) {
  Object object;
  object["name"] = attr.name;
  object["value"] = toJsonArray(attr.value);
  return Value(std::move(object));
}

static Value toJson(const StaticStringArrayAttr &attr) {
  Object object;
  object["name"] = attr.name;
  object["value"] = toJsonArray(attr.value);
  return Value(std::move(object));
}

static Value toJson(const StaticPortDesc &port) {
  Object object;
  object["port_id"] = static_cast<int64_t>(port.portId);
  object["parent_node_id"] = static_cast<int64_t>(port.parentNodeId);
  object["direction"] =
      (port.direction == StaticPortDirection::Input) ? "input" : "output";
  object["is_tagged"] = port.isTagged;
  object["is_memref"] = port.isMemRef;
  object["is_none"] = port.isNone;
  object["value_width"] = static_cast<int64_t>(port.valueWidth);
  object["tag_width"] = static_cast<int64_t>(port.tagWidth);
  return Value(std::move(object));
}

static Value toJson(const StaticModuleDesc &module) {
  Object object;
  object["hw_node_id"] = static_cast<int64_t>(module.hwNodeId);
  object["node_kind"] = static_cast<int64_t>(module.nodeKind);
  object["kind"] = static_cast<int64_t>(module.kind);
  object["name"] = module.name;
  object["op_kind"] = module.opKind;
  object["resource_class"] = module.resourceClass;
  object["input_ports"] = toJsonArray(module.inputPorts);
  object["output_ports"] = toJsonArray(module.outputPorts);

  Array intAttrs;
  for (const auto &attr : module.intAttrs)
    intAttrs.push_back(toJson(attr));
  object["int_attrs"] = std::move(intAttrs);

  Array strAttrs;
  for (const auto &attr : module.strAttrs)
    strAttrs.push_back(toJson(attr));
  object["str_attrs"] = std::move(strAttrs);

  Array byteArrayAttrs;
  for (const auto &attr : module.byteArrayAttrs)
    byteArrayAttrs.push_back(toJson(attr));
  object["byte_array_attrs"] = std::move(byteArrayAttrs);

  Array intArrayAttrs;
  for (const auto &attr : module.intArrayAttrs)
    intArrayAttrs.push_back(toJson(attr));
  object["int_array_attrs"] = std::move(intArrayAttrs);

  Array stringArrayAttrs;
  for (const auto &attr : module.stringArrayAttrs)
    stringArrayAttrs.push_back(toJson(attr));
  object["string_array_attrs"] = std::move(stringArrayAttrs);
  return Value(std::move(object));
}

static Value toJson(const StaticChannelDesc &channel) {
  Object object;
  object["hw_edge_id"] = static_cast<int64_t>(channel.hwEdgeId);
  object["src_port"] = static_cast<int64_t>(channel.srcPort);
  object["dst_port"] = static_cast<int64_t>(channel.dstPort);
  object["src_node"] = static_cast<int64_t>(channel.srcNode);
  object["dst_node"] = static_cast<int64_t>(channel.dstNode);
  object["pe_input_index"] = channel.peInputIndex;
  object["pe_output_index"] = channel.peOutputIndex;
  object["touches_boundary_input"] = channel.touchesBoundaryInput;
  object["touches_boundary_output"] = channel.touchesBoundaryOutput;
  return Value(std::move(object));
}

static Value toJson(const StaticPEDesc &pe) {
  Object object;
  object["pe_name"] = pe.peName;
  object["pe_kind"] = pe.peKind;
  object["fu_node_ids"] = toJsonArray(pe.fuNodeIds);
  object["row"] = pe.row;
  object["col"] = pe.col;
  object["num_input_ports"] = static_cast<int64_t>(pe.numInputPorts);
  object["num_output_ports"] = static_cast<int64_t>(pe.numOutputPorts);
  object["num_instruction"] = static_cast<int64_t>(pe.numInstruction);
  object["num_register"] = static_cast<int64_t>(pe.numRegister);
  object["reg_fifo_depth"] = static_cast<int64_t>(pe.regFifoDepth);
  object["tag_width"] = static_cast<int64_t>(pe.tagWidth);
  object["enable_share_operand_buffer"] = pe.enableShareOperandBuffer;
  object["operand_buffer_size"] = static_cast<int64_t>(pe.operandBufferSize);
  return Value(std::move(object));
}

static Value toJson(const StaticInputBinding &binding) {
  Object object;
  object["boundary_ordinal"] = static_cast<int64_t>(binding.boundaryOrdinal);
  object["sw_node_id"] = static_cast<int64_t>(binding.swNodeId);
  return Value(std::move(object));
}

static Value toJson(const StaticOutputBinding &binding) {
  Object object;
  object["boundary_ordinal"] = static_cast<int64_t>(binding.boundaryOrdinal);
  object["sw_node_id"] = static_cast<int64_t>(binding.swNodeId);
  return Value(std::move(object));
}

static Value toJson(const StaticMemoryBinding &binding) {
  Object object;
  object["region_id"] = static_cast<int64_t>(binding.regionId);
  object["region_index"] = static_cast<int64_t>(binding.regionIndex);
  object["sw_node_id"] = static_cast<int64_t>(binding.swNodeId);
  object["hw_node_id"] = static_cast<int64_t>(binding.hwNodeId);
  object["start_lane"] = static_cast<int64_t>(binding.startLane);
  object["end_lane"] = static_cast<int64_t>(binding.endLane);
  object["elem_size_log2"] = static_cast<int64_t>(binding.elemSizeLog2);
  object["supports_load"] = binding.supportsLoad;
  object["supports_store"] = binding.supportsStore;
  return Value(std::move(object));
}

static Value toJson(const CompletionObligation &obligation) {
  Object object;
  object["kind"] = static_cast<int64_t>(obligation.kind);
  object["ordinal"] = static_cast<int64_t>(obligation.ordinal);
  object["sw_node_id"] = static_cast<int64_t>(obligation.swNodeId);
  object["description"] = obligation.description;
  return Value(std::move(object));
}

static Value toJson(const StaticConfigSlice &slice) {
  Object object;
  object["name"] = slice.name;
  object["kind"] = slice.kind;
  object["hw_node"] = static_cast<int64_t>(slice.hwNode);
  object["word_offset"] = static_cast<int64_t>(slice.wordOffset);
  object["word_count"] = static_cast<int64_t>(slice.wordCount);
  object["complete"] = slice.complete;
  return Value(std::move(object));
}

static Value toJson(const RuntimeScalarSlotBinding &binding) {
  Object object;
  object["slot"] = static_cast<int64_t>(binding.slot);
  object["arg_index"] = binding.argIndex;
  object["port_idx"] = static_cast<int64_t>(binding.portIdx);
  return Value(std::move(object));
}

static Value toJson(const RuntimeMemorySlotBinding &binding) {
  Object object;
  object["slot"] = static_cast<int64_t>(binding.slot);
  object["memref_arg_index"] = binding.memrefArgIndex;
  object["region_id"] = static_cast<int64_t>(binding.regionId);
  object["elem_size_log2"] = static_cast<int64_t>(binding.elemSizeLog2);
  return Value(std::move(object));
}

static Value toJson(const RuntimeOutputSlotBinding &binding) {
  Object object;
  object["slot"] = static_cast<int64_t>(binding.slot);
  object["result_index"] = binding.resultIndex;
  object["port_idx"] = static_cast<int64_t>(binding.portIdx);
  return Value(std::move(object));
}

static Value toJson(const StaticMappedModel &model) {
  Object object;

  Array modules;
  for (const auto &module : model.getModules())
    modules.push_back(toJson(module));
  object["modules"] = std::move(modules);

  Array channels;
  for (const auto &channel : model.getChannels())
    channels.push_back(toJson(channel));
  object["channels"] = std::move(channels);

  Array ports;
  for (const auto &port : model.getPorts())
    ports.push_back(toJson(port));
  object["ports"] = std::move(ports);

  Array pes;
  for (const auto &pe : model.getPEs())
    pes.push_back(toJson(pe));
  object["pes"] = std::move(pes);

  Array inputBindings;
  for (const auto &binding : model.getInputBindings())
    inputBindings.push_back(toJson(binding));
  object["input_bindings"] = std::move(inputBindings);

  object["boundary_input_ordinals"] =
      toJsonArray(model.getBoundaryInputOrdinals());
  object["boundary_output_ordinals"] =
      toJsonArray(model.getBoundaryOutputOrdinals());

  Array outputBindings;
  for (const auto &binding : model.getOutputBindings())
    outputBindings.push_back(toJson(binding));
  object["output_bindings"] = std::move(outputBindings);

  Array memoryBindings;
  for (const auto &binding : model.getMemoryBindings())
    memoryBindings.push_back(toJson(binding));
  object["memory_bindings"] = std::move(memoryBindings);

  Array obligations;
  for (const auto &obligation : model.getCompletionObligations())
    obligations.push_back(toJson(obligation));
  object["completion_obligations"] = std::move(obligations);
  return Value(std::move(object));
}

static Value toJson(const StaticConfigImage &configImage) {
  Object object;
  Array slices;
  for (const auto &slice : configImage.slices)
    slices.push_back(toJson(slice));
  object["config_slices"] = std::move(slices);
  object["config_words"] = toJsonArray(configImage.words);
  return Value(std::move(object));
}

static Value toJson(const RuntimeControlImage &controlImage) {
  Object object;
  object["start_token_port"] = controlImage.startTokenPort;

  Array scalarBindings;
  for (const auto &binding : controlImage.scalarBindings)
    scalarBindings.push_back(toJson(binding));
  object["scalar_bindings"] = std::move(scalarBindings);

  Array memoryBindings;
  for (const auto &binding : controlImage.memoryBindings)
    memoryBindings.push_back(toJson(binding));
  object["memory_bindings"] = std::move(memoryBindings);

  Array outputBindings;
  for (const auto &binding : controlImage.outputBindings)
    outputBindings.push_back(toJson(binding));
  object["output_bindings"] = std::move(outputBindings);
  return Value(std::move(object));
}

template <typename T>
static std::optional<T> getInteger(const Object &object, llvm::StringRef key) {
  auto value = object.getInteger(key);
  if (!value)
    return std::nullopt;
  return static_cast<T>(*value);
}

static bool readString(const Object &object, llvm::StringRef key,
                       std::string &out) {
  if (std::optional<llvm::StringRef> value = object.getString(key)) {
    out = value->str();
    return true;
  }
  return false;
}

template <typename T>
static bool readIntegerVector(const Object &object, llvm::StringRef key,
                              std::vector<T> &out) {
  const Array *array = object.getArray(key);
  if (!array)
    return false;
  out.clear();
  out.reserve(array->size());
  for (const Value &value : *array) {
    auto integer = value.getAsInteger();
    if (!integer)
      return false;
    out.push_back(static_cast<T>(*integer));
  }
  return true;
}

static bool loadStaticPort(const Object &object, StaticPortDesc &port) {
  auto portId = getInteger<uint32_t>(object, "port_id");
  auto parentNodeId = getInteger<uint32_t>(object, "parent_node_id");
  auto valueWidth = getInteger<unsigned>(object, "value_width");
  auto tagWidth = getInteger<unsigned>(object, "tag_width");
  std::optional<llvm::StringRef> direction = object.getString("direction");
  auto isTagged = object.getBoolean("is_tagged");
  auto isMemRef = object.getBoolean("is_memref");
  auto isNone = object.getBoolean("is_none");
  if (!portId || !parentNodeId || !valueWidth || !tagWidth || !direction ||
      !isTagged || !isMemRef || !isNone)
    return false;
  port.portId = *portId;
  port.parentNodeId = *parentNodeId;
  port.direction = (*direction == "input") ? StaticPortDirection::Input
                                           : StaticPortDirection::Output;
  port.isTagged = *isTagged;
  port.isMemRef = *isMemRef;
  port.isNone = *isNone;
  port.valueWidth = *valueWidth;
  port.tagWidth = *tagWidth;
  return true;
}

static bool loadRuntimeImage(const Object &root, RuntimeImage &image) {
  const Object *modelObj = root.getObject("static_model");
  const Object *configObj = root.getObject("config_image");
  if (!modelObj || !configObj)
    return false;

  RuntimeImage loaded;

  if (const Array *ports = modelObj->getArray("ports")) {
    loaded.staticModel = StaticMappedModel();
    auto &mutablePorts =
        const_cast<std::vector<StaticPortDesc> &>(loaded.staticModel.getPorts());
    mutablePorts.clear();
    for (const Value &value : *ports) {
      const Object *object = value.getAsObject();
      if (!object)
        return false;
      StaticPortDesc port;
      if (!loadStaticPort(*object, port))
        return false;
      mutablePorts.push_back(std::move(port));
    }
  } else {
    return false;
  }

  auto loadModules = [&]() -> bool {
    const Array *array = modelObj->getArray("modules");
    if (!array)
      return false;
    auto &modules =
        const_cast<std::vector<StaticModuleDesc> &>(loaded.staticModel.getModules());
    modules.clear();
    for (const Value &value : *array) {
      const Object *object = value.getAsObject();
      if (!object)
        return false;
      StaticModuleDesc module;
      auto hwNodeId = getInteger<uint32_t>(*object, "hw_node_id");
      auto nodeKind = getInteger<int64_t>(*object, "node_kind");
      auto kind = getInteger<int64_t>(*object, "kind");
      if (!hwNodeId || !nodeKind || !kind)
        return false;
      module.hwNodeId = *hwNodeId;
      module.nodeKind = static_cast<StaticGraphNodeKind>(*nodeKind);
      module.kind = static_cast<StaticModuleKind>(*kind);
      if (!readString(*object, "name", module.name) ||
          !readString(*object, "op_kind", module.opKind) ||
          !readString(*object, "resource_class", module.resourceClass) ||
          !readIntegerVector(*object, "input_ports", module.inputPorts) ||
          !readIntegerVector(*object, "output_ports", module.outputPorts))
        return false;

      auto loadNamedInts = [&](llvm::StringRef key,
                               std::vector<StaticIntAttr> &out) -> bool {
        const Array *attrs = object->getArray(key);
        if (!attrs)
          return true;
        out.clear();
        for (const Value &attrValue : *attrs) {
          const Object *attrObject = attrValue.getAsObject();
          if (!attrObject)
            return false;
          StaticIntAttr attr;
          if (!readString(*attrObject, "name", attr.name))
            return false;
          auto valueInt = attrObject->getInteger("value");
          if (!valueInt)
            return false;
          attr.value = *valueInt;
          out.push_back(std::move(attr));
        }
        return true;
      };
      auto loadNamedStrings = [&](llvm::StringRef key,
                                  std::vector<StaticStringAttr> &out) -> bool {
        const Array *attrs = object->getArray(key);
        if (!attrs)
          return true;
        out.clear();
        for (const Value &attrValue : *attrs) {
          const Object *attrObject = attrValue.getAsObject();
          if (!attrObject)
            return false;
          StaticStringAttr attr;
          if (!readString(*attrObject, "name", attr.name) ||
              !readString(*attrObject, "value", attr.value))
            return false;
          out.push_back(std::move(attr));
        }
        return true;
      };
      auto loadNamedByteArrays =
          [&](llvm::StringRef key, std::vector<StaticByteArrayAttr> &out) -> bool {
        const Array *attrs = object->getArray(key);
        if (!attrs)
          return true;
        out.clear();
        for (const Value &attrValue : *attrs) {
          const Object *attrObject = attrValue.getAsObject();
          if (!attrObject)
            return false;
          StaticByteArrayAttr attr;
          if (!readString(*attrObject, "name", attr.name))
            return false;
          std::vector<int64_t> ints;
          if (!readIntegerVector(*attrObject, "value", ints))
            return false;
          attr.value.assign(ints.begin(), ints.end());
          out.push_back(std::move(attr));
        }
        return true;
      };
      auto loadNamedIntArrays =
          [&](llvm::StringRef key, std::vector<StaticIntArrayAttr> &out) -> bool {
        const Array *attrs = object->getArray(key);
        if (!attrs)
          return true;
        out.clear();
        for (const Value &attrValue : *attrs) {
          const Object *attrObject = attrValue.getAsObject();
          if (!attrObject)
            return false;
          StaticIntArrayAttr attr;
          if (!readString(*attrObject, "name", attr.name) ||
              !readIntegerVector(*attrObject, "value", attr.value))
            return false;
          out.push_back(std::move(attr));
        }
        return true;
      };
      auto loadNamedStringArrays = [&](llvm::StringRef key,
                                       std::vector<StaticStringArrayAttr> &out)
          -> bool {
        const Array *attrs = object->getArray(key);
        if (!attrs)
          return true;
        out.clear();
        for (const Value &attrValue : *attrs) {
          const Object *attrObject = attrValue.getAsObject();
          if (!attrObject)
            return false;
          StaticStringArrayAttr attr;
          if (!readString(*attrObject, "name", attr.name))
            return false;
          const Array *values = attrObject->getArray("value");
          if (!values)
            return false;
          for (const Value &entry : *values) {
            std::optional<llvm::StringRef> stringValue = entry.getAsString();
            if (!stringValue)
              return false;
            attr.value.push_back(stringValue->str());
          }
          out.push_back(std::move(attr));
        }
        return true;
      };

      if (!loadNamedInts("int_attrs", module.intAttrs) ||
          !loadNamedStrings("str_attrs", module.strAttrs) ||
          !loadNamedByteArrays("byte_array_attrs", module.byteArrayAttrs) ||
          !loadNamedIntArrays("int_array_attrs", module.intArrayAttrs) ||
          !loadNamedStringArrays("string_array_attrs", module.stringArrayAttrs))
        return false;
      modules.push_back(std::move(module));
    }
    return true;
  };

  auto loadChannels = [&]() -> bool {
    const Array *array = modelObj->getArray("channels");
    if (!array)
      return false;
    auto &channels = const_cast<std::vector<StaticChannelDesc> &>(
        loaded.staticModel.getChannels());
    channels.clear();
    for (const Value &value : *array) {
      const Object *object = value.getAsObject();
      if (!object)
        return false;
      StaticChannelDesc channel;
      auto hwEdgeId = getInteger<uint32_t>(*object, "hw_edge_id");
      auto srcPort = getInteger<IdIndex>(*object, "src_port");
      auto dstPort = getInteger<IdIndex>(*object, "dst_port");
      auto srcNode = getInteger<IdIndex>(*object, "src_node");
      auto dstNode = getInteger<IdIndex>(*object, "dst_node");
      auto peInputIndex = object->getInteger("pe_input_index");
      auto peOutputIndex = object->getInteger("pe_output_index");
      auto touchesBoundaryInput = object->getBoolean("touches_boundary_input");
      auto touchesBoundaryOutput = object->getBoolean("touches_boundary_output");
      if (!hwEdgeId || !srcPort || !dstPort || !srcNode || !dstNode ||
          !peInputIndex || !peOutputIndex || !touchesBoundaryInput ||
          !touchesBoundaryOutput)
        return false;
      channel.hwEdgeId = *hwEdgeId;
      channel.srcPort = *srcPort;
      channel.dstPort = *dstPort;
      channel.srcNode = *srcNode;
      channel.dstNode = *dstNode;
      channel.peInputIndex = static_cast<int>(*peInputIndex);
      channel.peOutputIndex = static_cast<int>(*peOutputIndex);
      channel.touchesBoundaryInput = *touchesBoundaryInput;
      channel.touchesBoundaryOutput = *touchesBoundaryOutput;
      channels.push_back(std::move(channel));
    }
    return true;
  };

  auto loadPEs = [&]() -> bool {
    const Array *array = modelObj->getArray("pes");
    if (!array)
      return false;
    auto &pes =
        const_cast<std::vector<StaticPEDesc> &>(loaded.staticModel.getPEs());
    pes.clear();
    for (const Value &value : *array) {
      const Object *object = value.getAsObject();
      if (!object)
        return false;
      StaticPEDesc pe;
      auto row = object->getInteger("row");
      auto col = object->getInteger("col");
      auto numInputPorts = object->getInteger("num_input_ports");
      auto numOutputPorts = object->getInteger("num_output_ports");
      auto numInstruction = object->getInteger("num_instruction");
      auto numRegister = object->getInteger("num_register");
      auto regFifoDepth = object->getInteger("reg_fifo_depth");
      auto tagWidth = object->getInteger("tag_width");
      auto enableShareOperandBuffer =
          object->getBoolean("enable_share_operand_buffer");
      auto operandBufferSize = object->getInteger("operand_buffer_size");
      if (!row || !col || !numInputPorts || !numOutputPorts || !numInstruction ||
          !numRegister || !regFifoDepth || !tagWidth ||
          !enableShareOperandBuffer || !operandBufferSize ||
          !readString(*object, "pe_name", pe.peName) ||
          !readString(*object, "pe_kind", pe.peKind) ||
          !readIntegerVector(*object, "fu_node_ids", pe.fuNodeIds))
        return false;
      pe.row = static_cast<int>(*row);
      pe.col = static_cast<int>(*col);
      pe.numInputPorts = static_cast<unsigned>(*numInputPorts);
      pe.numOutputPorts = static_cast<unsigned>(*numOutputPorts);
      pe.numInstruction = static_cast<unsigned>(*numInstruction);
      pe.numRegister = static_cast<unsigned>(*numRegister);
      pe.regFifoDepth = static_cast<unsigned>(*regFifoDepth);
      pe.tagWidth = static_cast<unsigned>(*tagWidth);
      pe.enableShareOperandBuffer = *enableShareOperandBuffer;
      pe.operandBufferSize = static_cast<unsigned>(*operandBufferSize);
      pes.push_back(std::move(pe));
    }
    return true;
  };

  auto loadInputBindings = [&]() -> bool {
    const Array *array = modelObj->getArray("input_bindings");
    if (!array)
      return false;
    auto &bindings = const_cast<std::vector<StaticInputBinding> &>(
        loaded.staticModel.getInputBindings());
    bindings.clear();
    for (const Value &value : *array) {
      const Object *object = value.getAsObject();
      if (!object)
        return false;
      StaticInputBinding binding;
      auto ordinal = getInteger<unsigned>(*object, "boundary_ordinal");
      auto swNodeId = getInteger<IdIndex>(*object, "sw_node_id");
      if (!ordinal || !swNodeId)
        return false;
      binding.boundaryOrdinal = *ordinal;
      binding.swNodeId = *swNodeId;
      bindings.push_back(std::move(binding));
    }
    return true;
  };

  auto loadBoundaryOrdinals = [&]() -> bool {
    if (!readIntegerVector(*modelObj, "boundary_input_ordinals",
                           loaded.staticModel.mutableBoundaryInputOrdinals()) ||
        !readIntegerVector(*modelObj, "boundary_output_ordinals",
                           loaded.staticModel.mutableBoundaryOutputOrdinals()))
      return false;
    return true;
  };

  auto loadOutputBindings = [&]() -> bool {
    const Array *array = modelObj->getArray("output_bindings");
    if (!array)
      return false;
    auto &bindings = const_cast<std::vector<StaticOutputBinding> &>(
        loaded.staticModel.getOutputBindings());
    bindings.clear();
    for (const Value &value : *array) {
      const Object *object = value.getAsObject();
      if (!object)
        return false;
      StaticOutputBinding binding;
      auto ordinal = getInteger<unsigned>(*object, "boundary_ordinal");
      auto swNodeId = getInteger<IdIndex>(*object, "sw_node_id");
      if (!ordinal || !swNodeId)
        return false;
      binding.boundaryOrdinal = *ordinal;
      binding.swNodeId = *swNodeId;
      bindings.push_back(std::move(binding));
    }
    return true;
  };

  auto loadMemoryBindings = [&]() -> bool {
    const Array *array = modelObj->getArray("memory_bindings");
    if (!array)
      return false;
    auto &bindings = const_cast<std::vector<StaticMemoryBinding> &>(
        loaded.staticModel.getMemoryBindings());
    bindings.clear();
    for (const Value &value : *array) {
      const Object *object = value.getAsObject();
      if (!object)
        return false;
      StaticMemoryBinding binding;
      auto regionId = getInteger<unsigned>(*object, "region_id");
      auto regionIndex = getInteger<unsigned>(*object, "region_index");
      auto swNodeId = getInteger<IdIndex>(*object, "sw_node_id");
      auto hwNodeId = getInteger<IdIndex>(*object, "hw_node_id");
      auto startLane = getInteger<unsigned>(*object, "start_lane");
      auto endLane = getInteger<unsigned>(*object, "end_lane");
      auto elemSizeLog2 = getInteger<unsigned>(*object, "elem_size_log2");
      auto supportsLoad = object->getBoolean("supports_load");
      auto supportsStore = object->getBoolean("supports_store");
      if (!regionId || !regionIndex || !swNodeId || !hwNodeId || !startLane || !endLane ||
          !elemSizeLog2 || !supportsLoad || !supportsStore)
        return false;
      binding.regionId = *regionId;
      binding.regionIndex = *regionIndex;
      binding.swNodeId = *swNodeId;
      binding.hwNodeId = *hwNodeId;
      binding.startLane = *startLane;
      binding.endLane = *endLane;
      binding.elemSizeLog2 = *elemSizeLog2;
      binding.supportsLoad = *supportsLoad;
      binding.supportsStore = *supportsStore;
      bindings.push_back(std::move(binding));
    }
    return true;
  };

  auto loadObligations = [&]() -> bool {
    const Array *array = modelObj->getArray("completion_obligations");
    if (!array)
      return false;
    auto &obligations = const_cast<std::vector<CompletionObligation> &>(
        loaded.staticModel.getCompletionObligations());
    obligations.clear();
    for (const Value &value : *array) {
      const Object *object = value.getAsObject();
      if (!object)
        return false;
      CompletionObligation obligation;
      auto kind = getInteger<int64_t>(*object, "kind");
      auto ordinal = getInteger<unsigned>(*object, "ordinal");
      auto swNodeId = getInteger<IdIndex>(*object, "sw_node_id");
      if (!kind || !ordinal || !swNodeId ||
          !readString(*object, "description", obligation.description))
        return false;
      obligation.kind = static_cast<CompletionObligationKind>(*kind);
      obligation.ordinal = *ordinal;
      obligation.swNodeId = *swNodeId;
      obligations.push_back(std::move(obligation));
    }
    return true;
  };

  auto loadConfig = [&]() -> bool {
    const Array *slices = configObj->getArray("config_slices");
    const Array *words = configObj->getArray("config_words");
    if (!slices || !words)
      return false;
    loaded.configImage.slices.clear();
    for (const Value &value : *slices) {
      const Object *object = value.getAsObject();
      if (!object)
        return false;
      StaticConfigSlice slice;
      auto hwNode = getInteger<IdIndex>(*object, "hw_node");
      auto wordOffset = getInteger<uint32_t>(*object, "word_offset");
      auto wordCount = getInteger<uint32_t>(*object, "word_count");
      auto complete = object->getBoolean("complete");
      if (!hwNode || !wordOffset || !wordCount || !complete ||
          !readString(*object, "name", slice.name) ||
          !readString(*object, "kind", slice.kind))
        return false;
      slice.hwNode = *hwNode;
      slice.wordOffset = *wordOffset;
      slice.wordCount = *wordCount;
      slice.complete = *complete;
      loaded.configImage.slices.push_back(std::move(slice));
    }
    loaded.configImage.words.clear();
    for (const Value &value : *words) {
      auto integer = value.getAsInteger();
      if (!integer)
        return false;
      loaded.configImage.words.push_back(static_cast<uint32_t>(*integer));
    }
    return true;
  };

  auto loadControl = [&]() -> bool {
    const Object *controlObj = root.getObject("control_image");
    if (!controlObj)
      return false;

    auto startTokenPort = getInteger<int32_t>(*controlObj, "start_token_port");
    const Array *scalarBindings = controlObj->getArray("scalar_bindings");
    const Array *memoryBindings = controlObj->getArray("memory_bindings");
    const Array *outputBindings = controlObj->getArray("output_bindings");
    if (!startTokenPort || !scalarBindings || !memoryBindings || !outputBindings)
      return false;

    loaded.controlImage.startTokenPort = *startTokenPort;
    loaded.controlImage.scalarBindings.clear();
    for (const Value &value : *scalarBindings) {
      const Object *object = value.getAsObject();
      if (!object)
        return false;
      RuntimeScalarSlotBinding binding;
      auto slot = getInteger<uint32_t>(*object, "slot");
      auto argIndex = getInteger<int64_t>(*object, "arg_index");
      auto portIdx = getInteger<uint32_t>(*object, "port_idx");
      if (!slot || !argIndex || !portIdx)
        return false;
      binding.slot = *slot;
      binding.argIndex = *argIndex;
      binding.portIdx = *portIdx;
      loaded.controlImage.scalarBindings.push_back(std::move(binding));
    }

    loaded.controlImage.memoryBindings.clear();
    for (const Value &value : *memoryBindings) {
      const Object *object = value.getAsObject();
      if (!object)
        return false;
      RuntimeMemorySlotBinding binding;
      auto slot = getInteger<uint32_t>(*object, "slot");
      auto argIndex = getInteger<int64_t>(*object, "memref_arg_index");
      auto regionId = getInteger<uint32_t>(*object, "region_id");
      auto elemSizeLog2 = getInteger<uint32_t>(*object, "elem_size_log2");
      if (!slot || !argIndex || !regionId || !elemSizeLog2)
        return false;
      binding.slot = *slot;
      binding.memrefArgIndex = *argIndex;
      binding.regionId = *regionId;
      binding.elemSizeLog2 = *elemSizeLog2;
      loaded.controlImage.memoryBindings.push_back(std::move(binding));
    }

    loaded.controlImage.outputBindings.clear();
    for (const Value &value : *outputBindings) {
      const Object *object = value.getAsObject();
      if (!object)
        return false;
      RuntimeOutputSlotBinding binding;
      auto slot = getInteger<uint32_t>(*object, "slot");
      auto resultIndex = getInteger<int64_t>(*object, "result_index");
      auto portIdx = getInteger<uint32_t>(*object, "port_idx");
      if (!slot || !resultIndex || !portIdx)
        return false;
      binding.slot = *slot;
      binding.resultIndex = *resultIndex;
      binding.portIdx = *portIdx;
      loaded.controlImage.outputBindings.push_back(std::move(binding));
    }
    return true;
  };

  if (!loadModules() || !loadChannels() || !loadPEs() || !loadInputBindings() ||
      !loadBoundaryOrdinals() || !loadOutputBindings() ||
      !loadMemoryBindings() || !loadObligations() || !loadConfig() ||
      !loadControl())
    return false;

  image = std::move(loaded);
  return true;
}

static bool writeJsonFile(const Value &value, const std::string &path,
                          std::string &error) {
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    error = ec.message();
    return false;
  }
  out << llvm::formatv("{0:2}", value) << "\n";
  return true;
}

} // namespace

bool writeRuntimeImageJson(const RuntimeImage &image, const std::string &path,
                           std::string &error) {
  Object root;
  root["version"] = 2;
  root["image_kind"] = "fcc_runtime_image";
  root["static_model"] = toJson(image.staticModel);
  root["config_image"] = toJson(image.configImage);
  root["control_image"] = toJson(image.controlImage);
  return writeJsonFile(Value(std::move(root)), path, error);
}

bool loadRuntimeImageJson(const std::string &path, RuntimeImage &image,
                          std::string &error) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot open runtime image";
    return false;
  }
  auto parsed = llvm::json::parse(buffer.get()->getBuffer());
  if (!parsed) {
    error = "runtime image is not valid JSON";
    return false;
  }
  const Object *root = parsed->getAsObject();
  if (!root) {
    error = "runtime image root must be an object";
    return false;
  }
  auto version = root->getInteger("version");
  std::optional<llvm::StringRef> imageKind = root->getString("image_kind");
  if (!version || *version != 2 || !imageKind ||
      *imageKind != "fcc_runtime_image") {
    error = "unsupported runtime image header";
    return false;
  }
  if (!loadRuntimeImage(*root, image)) {
    error = "runtime image payload is malformed";
    return false;
  }
  return true;
}

} // namespace sim
} // namespace fcc
