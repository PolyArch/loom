#include "fcc/Simulator/SimBundle.h"

#include "fcc/Mapper/TypeCompat.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <string>

namespace fcc {
namespace sim {

namespace {

using llvm::json::Array;
using llvm::json::Object;
using llvm::json::Value;

bool isMemrefType(mlir::Type type) { return mlir::isa<mlir::MemRefType>(type); }

std::optional<int64_t> getInt64(const Value &value) {
  if (auto v = value.getAsInteger())
    return *v;
  return std::nullopt;
}

template <typename T>
bool parseIntegerVector(const Value *value, std::vector<T> &out,
                        const char *fieldName, std::string &error) {
  if (!value) {
    std::ostringstream oss;
    oss << "missing field '" << fieldName << "'";
    error = oss.str();
    return false;
  }
  const Array *array = value->getAsArray();
  if (!array) {
    std::ostringstream oss;
    oss << "field '" << fieldName << "' must be an array";
    error = oss.str();
    return false;
  }
  out.clear();
  out.reserve(array->size());
  for (size_t i = 0; i < array->size(); ++i) {
    auto intValue = getInt64((*array)[i]);
    if (!intValue) {
      std::ostringstream oss;
      oss << "field '" << fieldName << "' element " << i
          << " must be an integer";
      error = oss.str();
      return false;
    }
    if (*intValue < 0 ||
        static_cast<uint64_t>(*intValue) >
            static_cast<uint64_t>(std::numeric_limits<T>::max())) {
      std::ostringstream oss;
      oss << "field '" << fieldName << "' element " << i
          << " is out of range";
      error = oss.str();
      return false;
    }
    out.push_back(static_cast<T>(*intValue));
  }
  return true;
}

bool parseScalarInput(const Object &object, SimBundleScalarInput &input,
                      std::string &error) {
  auto argIndex = object.getInteger("arg_index");
  if (!argIndex) {
    error = "scalar_inputs entries must include integer field 'arg_index'";
    return false;
  }
  input.argIndex = *argIndex;
  if (!parseIntegerVector<uint64_t>(object.get("data"), input.data, "data",
                                    error))
    return false;
  if (const Value *tags = object.get("tags")) {
    if (!parseIntegerVector<uint16_t>(tags, input.tags, "tags", error))
      return false;
    if (input.tags.size() != input.data.size()) {
      error = "scalar_inputs tags length must match data length";
      return false;
    }
  } else {
    input.tags.clear();
  }
  return true;
}

bool parseMemoryImage(const Object &object, SimBundleMemoryImage &image,
                      std::string &error) {
  auto memrefArgIndex = object.getInteger("memref_arg_index");
  if (!memrefArgIndex) {
    error =
        "memory_regions entries must include integer field 'memref_arg_index'";
    return false;
  }
  image.memrefArgIndex = *memrefArgIndex;
  image.elemSizeBytes = static_cast<uint32_t>(object.getInteger("elem_size_bytes")
                                                  .value_or(0));
  if (!parseIntegerVector<uint64_t>(object.get("values"), image.values,
                                    "values", error))
    return false;
  return true;
}

bool parseExpectedOutput(const Object &object, SimBundleExpectedOutput &output,
                         std::string &error) {
  auto resultIndex = object.getInteger("result_index");
  if (!resultIndex) {
    error =
        "expected_outputs entries must include integer field 'result_index'";
    return false;
  }
  output.resultIndex = *resultIndex;
  if (!parseIntegerVector<uint64_t>(object.get("data"), output.data, "data",
                                    error))
    return false;
  if (const Value *tags = object.get("tags")) {
    if (!parseIntegerVector<uint16_t>(tags, output.tags, "tags", error))
      return false;
    if (output.tags.size() != output.data.size()) {
      error = "expected_outputs tags length must match data length";
      return false;
    }
  } else {
    output.tags.clear();
  }
  return true;
}

bool parseExpectedMemory(const Object &object, SimBundleExpectedMemory &memory,
                         std::string &error) {
  auto memrefArgIndex = object.getInteger("memref_arg_index");
  if (!memrefArgIndex) {
    error = "expected_memory_regions entries must include integer field "
            "'memref_arg_index'";
    return false;
  }
  memory.memrefArgIndex = *memrefArgIndex;
  memory.elemSizeBytes =
      static_cast<uint32_t>(object.getInteger("elem_size_bytes").value_or(0));
  if (!parseIntegerVector<uint64_t>(object.get("values"), memory.values,
                                    "values", error))
    return false;
  return true;
}

std::vector<unsigned> buildBoundaryOrdinals(const Graph &graph, Node::Kind kind) {
  std::vector<unsigned> ordinals(graph.nodes.size(),
                                 std::numeric_limits<unsigned>::max());
  unsigned nextOrdinal = 0;
  for (IdIndex nodeId = 0; nodeId < static_cast<IdIndex>(graph.nodes.size());
       ++nodeId) {
    const Node *node = graph.getNode(nodeId);
    if (!node || node->kind != kind)
      continue;
    ordinals[nodeId] = nextOrdinal++;
  }
  return ordinals;
}

std::optional<IdIndex> findScalarInputNodeByArgIndex(const Graph &dfg,
                                                     int64_t argIndex) {
  for (IdIndex nodeId = 0; nodeId < static_cast<IdIndex>(dfg.nodes.size());
       ++nodeId) {
    const Node *node = dfg.getNode(nodeId);
    if (!node || node->kind != Node::ModuleInputNode || node->outputPorts.empty())
      continue;
    const Port *port = dfg.getPort(node->outputPorts.front());
    if (!port || isMemrefType(port->type) || mlir::isa<mlir::NoneType>(port->type))
      continue;
    if (getNodeAttrInt(node, "arg_index", -1) == argIndex)
      return nodeId;
  }
  return std::nullopt;
}

std::optional<IdIndex> findStartTokenInputNode(const Graph &dfg) {
  for (IdIndex nodeId = 0; nodeId < static_cast<IdIndex>(dfg.nodes.size());
       ++nodeId) {
    const Node *node = dfg.getNode(nodeId);
    if (!node || node->kind != Node::ModuleInputNode || node->outputPorts.empty())
      continue;
    const Port *port = dfg.getPort(node->outputPorts.front());
    if (port && mlir::isa<mlir::NoneType>(port->type))
      return nodeId;
  }
  return std::nullopt;
}

std::optional<unsigned> resolveInputOrdinal(const Graph &dfg, const Graph &adg,
                                            const MappingState &mapping,
                                            IdIndex swNodeId) {
  if (swNodeId < 0 ||
      swNodeId >= static_cast<IdIndex>(mapping.swNodeToHwNode.size()))
    return std::nullopt;
  IdIndex hwNodeId = mapping.swNodeToHwNode[swNodeId];
  if (hwNodeId == INVALID_ID)
    return std::nullopt;
  std::vector<unsigned> ordinals =
      buildBoundaryOrdinals(adg, Node::ModuleInputNode);
  if (hwNodeId >= static_cast<IdIndex>(ordinals.size()))
    return std::nullopt;
  if (ordinals[hwNodeId] == std::numeric_limits<unsigned>::max())
    return std::nullopt;
  return ordinals[hwNodeId];
}

std::optional<unsigned>
resolveOutputOrdinalByResultIndex(const Graph &dfg, const Graph &adg,
                                  const MappingState &mapping,
                                  int64_t resultIndex) {
  for (IdIndex nodeId = 0; nodeId < static_cast<IdIndex>(dfg.nodes.size());
       ++nodeId) {
    const Node *node = dfg.getNode(nodeId);
    if (!node || node->kind != Node::ModuleOutputNode)
      continue;
    if (getNodeAttrInt(node, "result_index", -1) != resultIndex)
      continue;
    if (nodeId >= static_cast<IdIndex>(mapping.swNodeToHwNode.size()))
      return std::nullopt;
    IdIndex hwNodeId = mapping.swNodeToHwNode[nodeId];
    if (hwNodeId == INVALID_ID)
      return std::nullopt;
    std::vector<unsigned> ordinals =
        buildBoundaryOrdinals(adg, Node::ModuleOutputNode);
    if (hwNodeId >= static_cast<IdIndex>(ordinals.size()))
      return std::nullopt;
    if (ordinals[hwNodeId] == std::numeric_limits<unsigned>::max())
      return std::nullopt;
    return ordinals[hwNodeId];
  }
  return std::nullopt;
}

bool fillRegionBytes(std::vector<uint8_t> &bytes, uint32_t elemSizeBytes,
                     const std::vector<uint64_t> &values, std::string &error) {
  if (elemSizeBytes == 0 || (elemSizeBytes != 1 && elemSizeBytes != 2 &&
                             elemSizeBytes != 4 && elemSizeBytes != 8)) {
    error = "elem_size_bytes must be one of 1, 2, 4, or 8";
    return false;
  }
  bytes.assign(values.size() * static_cast<size_t>(elemSizeBytes), 0);
  for (size_t index = 0; index < values.size(); ++index) {
    uint64_t value = values[index];
    unsigned bitWidth = elemSizeBytes * 8;
    if (bitWidth < 64 && (value >> bitWidth) != 0) {
      std::ostringstream oss;
      oss << "value " << value << " does not fit in " << bitWidth
          << " bits";
      error = oss.str();
      return false;
    }
    for (uint32_t byte = 0; byte < elemSizeBytes; ++byte) {
      bytes[index * elemSizeBytes + byte] =
          static_cast<uint8_t>((value >> (byte * 8)) & 0xffu);
    }
  }
  return true;
}

template <typename RangeT>
bool parseObjectArray(const Object &root, llvm::StringRef fieldName,
                      RangeT &output, std::string &error,
                      bool (*parseOne)(const Object &, typename RangeT::value_type &,
                                       std::string &)) {
  output.clear();
  if (const Value *value = root.get(fieldName)) {
    const Array *array = value->getAsArray();
    if (!array) {
      std::ostringstream oss;
      oss << "field '" << fieldName.str() << "' must be an array";
      error = oss.str();
      return false;
    }
    output.reserve(array->size());
    for (size_t i = 0; i < array->size(); ++i) {
      const Object *object = (*array)[i].getAsObject();
      if (!object) {
        std::ostringstream oss;
        oss << "field '" << fieldName.str() << "' element " << i
            << " must be an object";
        error = oss.str();
        return false;
      }
      typename RangeT::value_type parsed;
      if (!parseOne(*object, parsed, error))
        return false;
      output.push_back(std::move(parsed));
    }
  }
  return true;
}

bool compareTaggedVector(llvm::ArrayRef<uint16_t> actual,
                         llvm::ArrayRef<uint16_t> expected,
                         llvm::raw_ostream &details, unsigned portIdx,
                         unsigned &mismatches) {
  if (expected.empty())
    return true;
  bool pass = true;
  size_t count = std::max(actual.size(), expected.size());
  for (size_t i = 0; i < count; ++i) {
    uint16_t actualValue = i < actual.size() ? actual[i] : 0;
    uint16_t expectedValue = i < expected.size() ? expected[i] : 0;
    if (actualValue == expectedValue)
      continue;
    pass = false;
    ++mismatches;
    if (mismatches <= 10) {
      details << "port " << portIdx << " tag " << i << ": expected "
              << expectedValue << " got " << actualValue << "\n";
    }
  }
  return pass;
}

} // namespace

bool loadSimulationBundle(const std::string &path, SimulationBundle &bundle,
                          std::string &error) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot open simulation bundle: " + path;
    return false;
  }
  auto parsed = llvm::json::parse(buffer.get()->getBuffer());
  if (!parsed) {
    error = "cannot parse simulation bundle JSON";
    return false;
  }
  const Object *root = parsed->getAsObject();
  if (!root) {
    error = "simulation bundle root must be an object";
    return false;
  }

  bundle = SimulationBundle();
  if (auto caseName = root->getString("case_name"))
    bundle.caseName = caseName->str();
  if (auto startTokenCount = root->getInteger("start_token_count")) {
    if (*startTokenCount < 0) {
      error = "start_token_count must be non-negative";
      return false;
    }
    bundle.startTokenCount = static_cast<unsigned>(*startTokenCount);
  }

  if (!parseObjectArray(*root, "scalar_inputs", bundle.scalarInputs, error,
                        &parseScalarInput))
    return false;
  if (!parseObjectArray(*root, "memory_regions", bundle.memoryRegions, error,
                        &parseMemoryImage))
    return false;
  if (!parseObjectArray(*root, "expected_outputs", bundle.expectedOutputs,
                        error, &parseExpectedOutput))
    return false;
  if (!parseObjectArray(*root, "expected_memory_regions",
                        bundle.expectedMemoryRegions, error,
                        &parseExpectedMemory))
    return false;
  return true;
}

bool resolveSimulationBundle(const SimulationBundle &bundle, const Graph &dfg,
                             const Graph &adg, const MappingState &mapping,
                             ResolvedSimulationBundle &resolved,
                             std::string &error) {
  resolved = ResolvedSimulationBundle();
  resolved.caseName = bundle.caseName;
  resolved.setup = synthesizeSimulationSetup(dfg, adg, mapping);
  resolved.setup.inputs.clear();

  for (const SimBundleScalarInput &input : bundle.scalarInputs) {
    auto swNodeId = findScalarInputNodeByArgIndex(dfg, input.argIndex);
    if (!swNodeId) {
      std::ostringstream oss;
      oss << "cannot resolve scalar input arg_index=" << input.argIndex;
      error = oss.str();
      return false;
    }
    auto portIdx = resolveInputOrdinal(dfg, adg, mapping, *swNodeId);
    if (!portIdx) {
      std::ostringstream oss;
      oss << "cannot resolve scalar input port for arg_index="
          << input.argIndex;
      error = oss.str();
      return false;
    }
    SynthesizedInputPort synthesized;
    synthesized.portIdx = *portIdx;
    synthesized.type = "explicit";
    synthesized.data = input.data;
    synthesized.tags = input.tags;
    resolved.setup.inputs.push_back(std::move(synthesized));
  }

  if (bundle.startTokenCount > 0) {
    auto startNodeId = findStartTokenInputNode(dfg);
    if (!startNodeId) {
      error = "simulation bundle requested start tokens but no start token "
              "input exists";
      return false;
    }
    auto portIdx = resolveInputOrdinal(dfg, adg, mapping, *startNodeId);
    if (!portIdx) {
      error = "cannot resolve start token input port";
      return false;
    }
    SynthesizedInputPort start;
    start.portIdx = *portIdx;
    start.type = "none";
    start.data.assign(bundle.startTokenCount, 0);
    resolved.setup.inputs.push_back(std::move(start));
  }

  std::sort(resolved.setup.inputs.begin(), resolved.setup.inputs.end(),
            [](const SynthesizedInputPort &lhs,
               const SynthesizedInputPort &rhs) {
              return lhs.portIdx < rhs.portIdx;
            });

  for (const SimBundleMemoryImage &memory : bundle.memoryRegions) {
    auto it = std::find_if(resolved.setup.memoryRegions.begin(),
                           resolved.setup.memoryRegions.end(),
                           [&](const SynthesizedMemoryRegion &region) {
                             return region.memrefArgIndex ==
                                    memory.memrefArgIndex;
                           });
    if (it == resolved.setup.memoryRegions.end()) {
      std::ostringstream oss;
      oss << "cannot resolve memory region memref_arg_index="
          << memory.memrefArgIndex;
      error = oss.str();
      return false;
    }
    uint32_t elemSizeBytes = uint32_t{1} << it->elemSizeLog2;
    if (memory.elemSizeBytes != 0 && memory.elemSizeBytes != elemSizeBytes) {
      std::ostringstream oss;
      oss << "memory region memref_arg_index=" << memory.memrefArgIndex
          << " uses elem_size_bytes=" << memory.elemSizeBytes
          << " but mapped region requires " << elemSizeBytes;
      error = oss.str();
      return false;
    }
    if (!fillRegionBytes(it->data, elemSizeBytes, memory.values, error))
      return false;
  }

  for (const SimBundleExpectedOutput &expected : bundle.expectedOutputs) {
    auto portIdx =
        resolveOutputOrdinalByResultIndex(dfg, adg, mapping, expected.resultIndex);
    if (!portIdx) {
      std::ostringstream oss;
      oss << "cannot resolve expected output result_index="
          << expected.resultIndex;
      error = oss.str();
      return false;
    }
    ResolvedExpectedOutput resolvedOutput;
    resolvedOutput.portIdx = *portIdx;
    resolvedOutput.resultIndex = expected.resultIndex;
    resolvedOutput.data = expected.data;
    resolvedOutput.tags = expected.tags;
    resolved.expectedOutputs.push_back(std::move(resolvedOutput));
  }

  for (const SimBundleExpectedMemory &expected : bundle.expectedMemoryRegions) {
    auto it = std::find_if(resolved.setup.memoryRegions.begin(),
                           resolved.setup.memoryRegions.end(),
                           [&](const SynthesizedMemoryRegion &region) {
                             return region.memrefArgIndex ==
                                    expected.memrefArgIndex;
                           });
    if (it == resolved.setup.memoryRegions.end()) {
      std::ostringstream oss;
      oss << "cannot resolve expected memory region memref_arg_index="
          << expected.memrefArgIndex;
      error = oss.str();
      return false;
    }
    uint32_t elemSizeBytes = uint32_t{1} << it->elemSizeLog2;
    if (expected.elemSizeBytes != 0 && expected.elemSizeBytes != elemSizeBytes) {
      std::ostringstream oss;
      oss << "expected memory region memref_arg_index="
          << expected.memrefArgIndex << " uses elem_size_bytes="
          << expected.elemSizeBytes << " but mapped region requires "
          << elemSizeBytes;
      error = oss.str();
      return false;
    }
    ResolvedExpectedMemory resolvedMemory;
    resolvedMemory.regionId = it->regionId;
    resolvedMemory.memrefArgIndex = expected.memrefArgIndex;
    if (!fillRegionBytes(resolvedMemory.data, elemSizeBytes, expected.values,
                         error))
      return false;
    resolved.expectedMemoryRegions.push_back(std::move(resolvedMemory));
  }

  return true;
}

SimValidationReport validateSimulationBundle(
    const SimSession &session, const ResolvedSimulationBundle &bundle) {
  SimValidationReport report;
  report.pass = true;

  for (const ResolvedExpectedOutput &output : bundle.expectedOutputs) {
    ++report.totalChecks;
    std::vector<uint64_t> actualData = session.getOutput(output.portIdx);
    size_t count = std::max(actualData.size(), output.data.size());
    std::string detailsString;
    llvm::raw_string_ostream details(detailsString);
    bool pass = true;
    for (size_t i = 0; i < count; ++i) {
      uint64_t actualValue = i < actualData.size() ? actualData[i] : 0;
      uint64_t expectedValue = i < output.data.size() ? output.data[i] : 0;
      if (actualValue == expectedValue)
        continue;
      pass = false;
      ++report.mismatches;
      if (report.mismatches <= 10) {
        details << "port " << output.portIdx << " value " << i << ": expected "
                << expectedValue << " got " << actualValue << "\n";
      }
    }
    std::vector<uint16_t> actualTags = session.getOutputTags(output.portIdx);
    pass &= compareTaggedVector(actualTags, output.tags, details, output.portIdx,
                                report.mismatches);
    if (!pass) {
      report.pass = false;
      std::ostringstream oss;
      oss << "expected output result_index=" << output.resultIndex
          << " failed";
      if (!detailsString.empty())
        oss << ": " << detailsString;
      report.diagnostics.push_back(oss.str());
    }
  }

  for (const ResolvedExpectedMemory &memory : bundle.expectedMemoryRegions) {
    ++report.totalChecks;
    CompareResult compare =
        session.compareMemoryRegion(memory.regionId, memory.data);
    if (compare.pass)
      continue;
    report.pass = false;
    report.mismatches += compare.mismatches;
    std::ostringstream oss;
    oss << "expected memory memref_arg_index=" << memory.memrefArgIndex
        << " failed";
    if (!compare.details.empty())
      oss << ": " << compare.details;
    report.diagnostics.push_back(oss.str());
  }

  return report;
}

bool writeValidationReport(const SimValidationReport &report,
                           const std::string &path) {
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "SimBundle: cannot open " << path << ": "
                 << ec.message() << "\n";
    return false;
  }

  out << "{\n";
  out << "  \"pass\": " << (report.pass ? "true" : "false") << ",\n";
  out << "  \"total_checks\": " << report.totalChecks << ",\n";
  out << "  \"mismatches\": " << report.mismatches << ",\n";
  out << "  \"diagnostics\": [\n";
  for (size_t i = 0; i < report.diagnostics.size(); ++i) {
    if (i > 0)
      out << ",\n";
    out << "    ";
    out.write_escaped(report.diagnostics[i], true);
  }
  out << "\n  ]\n";
  out << "}\n";
  return true;
}

} // namespace sim
} // namespace fcc
