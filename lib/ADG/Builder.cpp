#include "BuilderInternal.h"

#include "Dataflow/IR/DataflowEnums.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/ImplementationFamily.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cctype>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <optional>
#include <system_error>

using namespace loom::adg;
using namespace loom::adg::detail;

namespace {

llvm::Error validateFabricModule(llvm::StringRef text) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<::fabric::FabricDialect>();

  std::string diagnostic;
  mlir::ScopedDiagnosticHandler capture(
      &context, [&](mlir::Diagnostic &emitted) {
        if (diagnostic.empty() &&
            emitted.getSeverity() == mlir::DiagnosticSeverity::Error)
          diagnostic = emitted.str();
        return mlir::success();
      });
  mlir::ParserConfig config(&context, /*verifyAfterParse=*/true);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(text, config, "<adg-builder>");
  if (module)
    return llvm::Error::success();

  if (diagnostic.empty())
    diagnostic = "Fabric parser or verifier rejected the generated module";
  return llvm::createStringError(std::errc::invalid_argument,
                                 "ADG Fabric validation failed: %s",
                                 diagnostic.c_str());
}

llvm::StringRef scheduleName(Schedule schedule) {
  switch (schedule) {
  case Schedule::Spatial:
    return "spatial";
  case Schedule::Temporal:
    return "temporal";
  }
  llvm_unreachable("unknown ADG schedule");
}

llvm::Error validateFifoSpec(const FifoSpec &fifo) {
  if (fifo.resultName.empty() || fifo.sourceName.empty() ||
      fifo.resultType.empty())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "ADG fifo specification is incomplete");
  if (fifo.maxDepth == 0)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "ADG fifo max depth must be greater than zero");
  if (fifo.maxDepth >
      static_cast<unsigned>(std::numeric_limits<std::int32_t>::max()))
    return llvm::createStringError(
        std::errc::invalid_argument,
        "ADG fifo max depth exceeds signed i32 range");
  if (fifo.bypassed && !fifo.bypassable)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "ADG fifo bypass configuration requires bypassable hardware");
  return llvm::Error::success();
}

llvm::Error validateBoundarySpec(const BoundarySpec &boundary) {
  if (!::fabric::symbolizeBoundaryDirection(
          static_cast<std::uint32_t>(boundary.direction)))
    return llvm::createStringError(std::errc::invalid_argument,
                                   "ADG boundary direction is invalid");

  switch (boundary.direction) {
  case ::fabric::BoundaryDirection::S2t:
    if (boundary.inputs.size() != 2 || boundary.resultNames.size() != 1)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "ADG s2t boundary requires exactly two inputs and one result");
    break;
  case ::fabric::BoundaryDirection::T2s:
    if (boundary.inputs.size() != 1 || boundary.resultNames.empty() ||
        boundary.resultNames.size() > 2)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "ADG t2s boundary requires exactly one input and one or two "
          "results");
    break;
  case ::fabric::BoundaryDirection::T2t:
    return llvm::createStringError(
        std::errc::not_supported,
        "ADG t2t boundary construction is not supported");
  }

  if (boundary.resultTypes.size() != boundary.resultNames.size())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "ADG boundary result type count must match result name count");
  for (const BoundaryInput &input : boundary.inputs) {
    if (input.sourceName.empty())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "ADG boundary specification contains an empty source name");
    if (input.destinationType && input.destinationType->empty())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "ADG boundary specification contains an empty destination type");
  }
  for (auto [resultName, resultType] :
       llvm::zip(boundary.resultNames, boundary.resultTypes))
    if (resultName.empty() || resultType.empty())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "ADG boundary specification contains an incomplete result");
  return llvm::Error::success();
}

std::string canonicalValueName(llvm::StringRef name) {
  name.consume_front("%");
  return name.str();
}

std::string valueName(llvm::StringRef name) {
  return "%" + canonicalValueName(name);
}

bool isSpatialTransportType(llvm::StringRef type) {
  return type.starts_with("!fabric.bits<");
}

bool isTemporalTransportType(llvm::StringRef type) {
  return type.starts_with("!fabric.bits_tag<");
}

bool isTransportType(llvm::StringRef type) {
  return isSpatialTransportType(type) || isTemporalTransportType(type);
}

bool fragmentHidesDirectUse(llvm::StringRef fragment) {
  for (std::size_t percent = fragment.find('%');
       percent != llvm::StringRef::npos;
       percent = fragment.find('%', percent + 1)) {
    std::size_t end = percent + 1;
    while (end < fragment.size()) {
      unsigned char c = static_cast<unsigned char>(fragment[end]);
      if (!std::isalnum(c) && c != '_' && c != '-' && c != '.' && c != '$')
        break;
      ++end;
    }
    if (end == percent + 1)
      return true;
    std::size_t next = end;
    while (next < fragment.size() &&
           std::isspace(static_cast<unsigned char>(fragment[next])))
      ++next;
    if (next >= fragment.size() || fragment[next] != '=')
      return true;
  }
  return false;
}

llvm::Expected<unsigned> temporalTagDomainSize(llvm::StringRef type) {
  llvm::StringRef spelling = type.trim();
  if (!spelling.consume_front("!fabric.bits_tag<") ||
      !spelling.consume_back(">"))
    return llvm::createStringError(
        std::errc::invalid_argument,
        "ADG temporal transport type %s is not a bits_tag type",
        type.str().c_str());

  auto [dataWidth, tagWidthText] = spelling.split(',');
  dataWidth = dataWidth.trim();
  tagWidthText = tagWidthText.trim();
  if (dataWidth.empty() || tagWidthText.empty() || tagWidthText.contains(','))
    return llvm::createStringError(
        std::errc::invalid_argument,
        "ADG temporal transport type %s has an invalid bits_tag shape",
        type.str().c_str());

  unsigned dataWidthValue = 0;
  unsigned tagWidth = 0;
  if (dataWidth.getAsInteger(10, dataWidthValue) ||
      tagWidthText.getAsInteger(10, tagWidth) || tagWidth == 0)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "ADG temporal transport type %s has an invalid tag width",
        type.str().c_str());
  if (tagWidth >= std::numeric_limits<int32_t>::digits)
    return llvm::createStringError(
        std::errc::value_too_large,
        "ADG temporal transport tag width %u requires a route table size "
        "that does not fit Fabric's signed i32 field",
        tagWidth);
  return 1u << tagWidth;
}

struct TransportFanout {
  std::string sourceName;
  std::string type;
  Schedule schedule;
  unsigned temporalRouteTableSize;
  std::vector<std::string> resultNames;
};

struct TransportPlan {
  std::vector<std::string> resolvedUses;
  std::vector<TransportFanout> fanouts;
};

llvm::Expected<TransportPlan>
buildTransportPlan(llvm::ArrayRef<std::string> useSources,
                   const llvm::StringMap<std::string> &valueTypes,
                   llvm::StringSet<> reservedNames) {
  struct SourceUses {
    std::string sourceName;
    std::string type;
    std::vector<std::size_t> useIds;
  };

  TransportPlan plan;
  plan.resolvedUses.assign(useSources.begin(), useSources.end());
  llvm::StringMap<std::size_t> sourceIndices;
  std::vector<SourceUses> sources;
  for (auto [useId, sourceName] : llvm::enumerate(useSources)) {
    auto typeIt = valueTypes.find(sourceName);
    if (typeIt == valueTypes.end())
      return llvm::createStringError(std::errc::invalid_argument,
                                     "ADG direct body source %s is unknown",
                                     sourceName.c_str());
    if (!isTransportType(typeIt->second))
      continue;

    auto [indexIt, inserted] = sourceIndices.try_emplace(sourceName);
    if (inserted) {
      indexIt->second = sources.size();
      sources.push_back(
          SourceUses{sourceName, typeIt->second, std::vector<std::size_t>()});
    }
    sources[indexIt->second].useIds.push_back(useId);
  }

  for (const SourceUses &source : sources) {
    if (source.useIds.size() == 1)
      continue;

    std::size_t fanoutIndex = plan.fanouts.size();
    Schedule schedule = isTemporalTransportType(source.type)
                            ? Schedule::Temporal
                            : Schedule::Spatial;
    unsigned temporalRouteTableSize = 0;
    if (schedule == Schedule::Temporal) {
      auto domainSize = temporalTagDomainSize(source.type);
      if (!domainSize)
        return domainSize.takeError();
      temporalRouteTableSize = *domainSize;
    }
    TransportFanout fanout{
        source.sourceName, source.type, schedule, temporalRouteTableSize, {}};
    for (auto [consumerIndex, useId] : llvm::enumerate(source.useIds)) {
      std::string resultName =
          (llvm::Twine("transport_fanout") + llvm::Twine(fanoutIndex) + "_out" +
           llvm::Twine(consumerIndex))
              .str();
      if (!reservedNames.insert(resultName).second)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "ADG generated transport fanout result %s conflicts with an "
            "existing value",
            resultName.c_str());
      fanout.resultNames.push_back(resultName);
      plan.resolvedUses[useId] = std::move(resultName);
    }
    plan.fanouts.push_back(std::move(fanout));
  }
  return plan;
}

void printTypeList(llvm::raw_ostream &os, llvm::ArrayRef<std::string> types) {
  os << '(';
  for (std::size_t i = 0; i < types.size(); ++i) {
    if (i)
      os << ", ";
    os << types[i];
  }
  os << ')';
}

void printTypeSequence(llvm::raw_ostream &os,
                       llvm::ArrayRef<std::string> types) {
  for (std::size_t i = 0; i < types.size(); ++i) {
    if (i)
      os << ", ";
    os << types[i];
  }
}

void printResultTypes(llvm::raw_ostream &os,
                      llvm::ArrayRef<std::string> types) {
  if (types.empty()) {
    os << "()";
    return;
  }
  if (types.size() == 1) {
    os << types.front();
    return;
  }
  printTypeList(os, types);
}

void printFifo(llvm::raw_ostream &os, const FifoSpec &fifo,
               llvm::StringRef resolvedSource, llvm::StringRef sourceType) {
  os << "  " << valueName(fifo.resultName) << " = fabric.fifo "
     << valueName(resolvedSource) << " [max_depth = " << fifo.maxDepth
     << ", bypassable = " << (fifo.bypassable ? "true" : "false") << ']';
  if (fifo.bypassed)
    os << " {bypassed = " << (*fifo.bypassed ? "true" : "false") << '}';
  os << "\n    : " << sourceType;
  if (sourceType != fifo.resultType)
    os << " to " << fifo.resultType;
  os << '\n';
}

void printBoundary(llvm::raw_ostream &os, const BoundarySpec &boundary,
                   llvm::ArrayRef<std::string> resolvedSources,
                   llvm::ArrayRef<std::string> sourceTypes) {
  for (std::size_t index = 0; index < boundary.resultNames.size(); ++index) {
    os << (index == 0 ? "  " : ", ") << valueName(boundary.resultNames[index]);
  }
  os << " = fabric.boundary ["
     << ::fabric::stringifyBoundaryDirection(boundary.direction) << "] ";
  for (std::size_t index = 0; index < resolvedSources.size(); ++index) {
    if (index)
      os << ", ";
    os << valueName(resolvedSources[index]);
  }
  os << " : ";
  bool multipleInputs = sourceTypes.size() != 1;
  if (multipleInputs)
    os << '(';
  for (std::size_t index = 0; index < sourceTypes.size(); ++index) {
    if (index)
      os << ", ";
    os << sourceTypes[index];
    const std::optional<std::string> &destinationType =
        boundary.inputs[index].destinationType;
    if (destinationType && sourceTypes[index] != *destinationType)
      os << " to " << *destinationType;
  }
  if (multipleInputs)
    os << ')';
  os << " -> ";
  printResultTypes(os, boundary.resultTypes);
  os << '\n';
}

void printBindings(llvm::raw_ostream &os, llvm::ArrayRef<PortBinding> bindings,
                   llvm::StringRef indent) {
  for (std::size_t i = 0; i < bindings.size(); ++i) {
    const PortBinding &binding = bindings[i];
    if (i)
      os << ",\n" << indent;
    os << valueName(binding.localName) << " = " << valueName(binding.sourceName)
       << " : " << binding.type;
    if (!binding.castType.empty())
      os << " to " << binding.castType;
  }
}

void printDirectBindings(llvm::raw_ostream &os,
                         llvm::ArrayRef<PortBinding> bindings,
                         llvm::ArrayRef<std::size_t> useIds,
                         llvm::ArrayRef<std::string> resolvedUses,
                         llvm::StringRef indent) {
  assert(bindings.size() == useIds.size());
  for (std::size_t i = 0; i < bindings.size(); ++i) {
    const PortBinding &binding = bindings[i];
    if (i)
      os << ",\n" << indent;
    os << valueName(binding.localName) << " = "
       << valueName(resolvedUses[useIds[i]]) << " : " << binding.type;
    if (!binding.castType.empty())
      os << " to " << binding.castType;
  }
}

void printStringArray(llvm::raw_ostream &os,
                      llvm::ArrayRef<std::string> values) {
  os << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i)
      os << ", ";
    os << '"' << values[i] << '"';
  }
  os << ']';
}

void printFabricOp(llvm::raw_ostream &os, ::mlir::MLIRContext &context,
                   const FabricOpSpec &op) {
  os << "      ";
  for (std::size_t i = 0; i < op.results.size(); ++i) {
    if (i)
      os << ", ";
    os << valueName(op.results[i]);
  }
  if (!op.results.empty())
    os << " = ";
  os << "fabric.op [";
  // Text is derived only at MLIR construction boundaries; the schema owns it.
  llvm::interleaveComma(
      op.enabledSchemas, os, [&](::dataflow::OperationSchemaId schema) {
        os << '@' << ::dataflow::operationSchemaSpelling(schema);
      });
  os << "] (";
  for (std::size_t i = 0; i < op.operands.size(); ++i) {
    if (i)
      os << ", ";
    os << valueName(op.operands[i]);
  }
  os << ')';
  os << ' ';
  printFabricOpAttrs(os, context, op.capability);
  os << " : ";
  printTypeList(os, op.operandTypes);
  os << " -> ";
  printResultTypes(os, op.resultTypes);
  os << '\n';
}

void printFu(llvm::raw_ostream &os, ::mlir::MLIRContext &context,
             const FuSpec &fu) {
  os << "    fabric.fu(";
  printBindings(os, fu.inputs, "              ");
  os << ") -> ";
  printResultTypes(os, fu.resultTypes);
  os << " {\n";
  for (const FabricOpSpec &op : fu.operations)
    printFabricOp(os, context, op);
  os << "      fabric.yield";
  if (!fu.yieldValues.empty()) {
    os << ' ';
    bool hasYieldTypes = fu.yieldTypes.size() == fu.yieldValues.size();
    bool needsPerValue = hasYieldTypes;
    if (hasYieldTypes && fu.resultTypes.size() == fu.yieldValues.size()) {
      needsPerValue = false;
      for (std::size_t i = 0; i < fu.yieldValues.size(); ++i) {
        if (fu.yieldTypes[i] != fu.resultTypes[i]) {
          needsPerValue = true;
          break;
        }
      }
    }
    if (!needsPerValue) {
      for (std::size_t i = 0; i < fu.yieldValues.size(); ++i) {
        if (i)
          os << ", ";
        os << valueName(fu.yieldValues[i]);
      }
      os << " : ";
      printTypeSequence(os, fu.resultTypes);
    } else {
      for (std::size_t i = 0; i < fu.yieldValues.size(); ++i) {
        if (i)
          os << ", ";
        std::string declaredType =
            i < fu.resultTypes.size() ? fu.resultTypes[i] : fu.yieldTypes[i];
        os << valueName(fu.yieldValues[i]) << " : " << fu.yieldTypes[i];
        if (fu.yieldTypes[i] != declaredType)
          os << " to " << declaredType;
      }
    }
  }
  os << "\n";
  os << "    }\n";
}

llvm::Error validateFu(const FuSpec &fu) {
  if (fu.yieldValues.size() != fu.resultTypes.size())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "ADG fu yield value count must match result type count");
  if (!fu.yieldTypes.empty() && fu.yieldTypes.size() != fu.yieldValues.size())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "ADG fu yield type count must match yield value count");
  for (const FabricOpSpec &op : fu.operations) {
    if (op.enabledSchemas.empty())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "ADG fabric.op must enable at least one registered member");
    if (llvm::Error error = validateFabricOpCapability(op.capability))
      return error;
    for (::dataflow::OperationSchemaId schema : op.enabledSchemas)
      if (!::fabric::admitsOperationSchema(op.capability.family, schema))
        return llvm::createStringError(
            std::errc::invalid_argument,
            "ADG fabric.op enables %s, which implementation family %s does "
            "not admit",
            ::dataflow::operationSchemaSpelling(schema).str().c_str(),
            ::fabric::implementationFamilyKeyword(op.capability.family)
                .str()
                .c_str());
  }
  return llvm::Error::success();
}

llvm::Error validateTemporalPe(const TemporalPeConfig &config) {
  if (config.operandBufferSize == 0)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "temporal ADG pe requires a positive operand buffer size in every "
        "operand buffer mode");
  return llvm::Error::success();
}

void printTemporalPeAttributes(llvm::raw_ostream &os,
                               const TemporalPeConfig &config) {
  os << " attributes {\n"
     << "         tag_width = " << config.tagWidth << " : i32,\n"
     << "         num_instruction = " << config.numInstruction << " : i32,\n";
  if (config.numRegFifo)
    os << "         num_reg_fifo = " << config.numRegFifo << " : i32,\n";
  if (config.regFifoDepth)
    os << "         reg_fifo_depth = " << config.regFifoDepth << " : i32,\n";
  if (config.regFifoPorts)
    os << "         reg_fifo_ports = " << config.regFifoPorts << " : i32,\n";
  os << "         fu_config_mode = \"" << config.fuConfigMode << "\",\n"
     << "         operand_buffer_mode = #fabric.operand_buffer_mode<"
     << ::fabric::stringifyOperandBufferMode(config.operandBufferMode) << ">,\n"
     << "         operand_buffer_size = " << config.operandBufferSize
     << " : i32\n       }";
}

void printPe(llvm::raw_ostream &os, ::mlir::MLIRContext &context,
             const PeSpec &pe, llvm::ArrayRef<std::size_t> useIds,
             llvm::ArrayRef<std::string> resolvedUses) {
  os << "  ";
  if (!pe.resultNames.empty()) {
    for (std::size_t i = 0; i < pe.resultNames.size(); ++i) {
      if (i)
        os << ", ";
      os << valueName(pe.resultNames[i]);
    }
    os << " = ";
  }
  os << "fabric.pe [" << scheduleName(pe.schedule) << "] (";
  printDirectBindings(os, pe.inputs, useIds, resolvedUses,
                      "                    ");
  os << ") -> ";
  printResultTypes(os, pe.resultTypes);
  if (pe.schedule == Schedule::Temporal)
    printTemporalPeAttributes(os, pe.temporal);
  os << " {\n";
  for (const FuSpec &fu : pe.fus)
    printFu(os, context, fu);
  os << "  }\n";
}

void printSwitchHwParams(llvm::raw_ostream &os, const SwitchSpec &sw) {
  os << "[{connectivity_table = ";
  printStringArray(os, sw.connectivityTable);
  if (sw.schedule == Schedule::Temporal)
    os << ", route_table_size = " << sw.temporalRouteTableSize << " : i32";
  os << "}]";
}

void printSwitch(llvm::raw_ostream &os, const SwitchSpec &sw,
                 std::size_t switchIndex, llvm::ArrayRef<std::size_t> useIds,
                 llvm::ArrayRef<std::string> resolvedUses,
                 llvm::ArrayRef<std::string> operandTypes) {
  os << "  ";
  for (std::size_t i = 0; i < sw.resultTypes.size(); ++i) {
    if (i)
      os << ", ";
    os << "%sw" << switchIndex << "_out" << i;
  }
  os << " = fabric.switch [" << scheduleName(sw.schedule) << "]";
  for (std::size_t i = 0; i < sw.inputs.size(); ++i) {
    if (i)
      os << ',';
    os << ' ' << valueName(resolvedUses[useIds[i]]);
  }
  os << "\n         ";
  printSwitchHwParams(os, sw);
  os << "\n         : ";
  printTypeList(os, operandTypes);
  os << "\n        -> ";
  printResultTypes(os, sw.resultTypes);
  os << '\n';
}

void printTransportFanout(llvm::raw_ostream &os,
                          const TransportFanout &fanout) {
  os << "  ";
  for (std::size_t i = 0; i < fanout.resultNames.size(); ++i) {
    if (i)
      os << ", ";
    os << valueName(fanout.resultNames[i]);
  }
  os << " = fabric.switch [" << scheduleName(fanout.schedule) << "] "
     << valueName(fanout.sourceName) << "\n"
     << "         [{connectivity_table = [";
  for (std::size_t i = 0; i < fanout.resultNames.size(); ++i) {
    if (i)
      os << ", ";
    os << "\"1\"";
  }
  os << ']';
  if (fanout.schedule == Schedule::Temporal)
    os << ", route_table_size = " << fanout.temporalRouteTableSize << " : i32";
  os << "}]\n"
     << "         : (" << fanout.type << ")\n"
     << "        -> ";
  std::vector<std::string> resultTypes(fanout.resultNames.size(), fanout.type);
  printResultTypes(os, resultTypes);
  os << '\n';
}

llvm::Error validateSwitch(const SwitchSpec &sw,
                           const llvm::StringMap<std::string> &inputTypes) {
  if (sw.inputs.empty())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "ADG switch has no inputs");
  if (sw.resultTypes.empty())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "ADG switch has no result types");
  if (sw.connectivityTable.size() != sw.resultTypes.size())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "ADG switch connectivity rows must match result count");
  if (sw.schedule == Schedule::Spatial && sw.temporalRouteTableSize != 0)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "spatial ADG switch must not carry temporal route table size");
  if (sw.schedule == Schedule::Temporal && sw.temporalRouteTableSize == 0)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "temporal ADG switch requires temporal route table size");

  std::vector<bool> columnHasConnection(sw.inputs.size(), false);
  for (const std::string &input : sw.inputs) {
    std::string inputName = canonicalValueName(input);
    if (!inputTypes.contains(inputName))
      return llvm::createStringError(std::errc::invalid_argument,
                                     "ADG switch input %s is unknown",
                                     inputName.c_str());
  }
  for (std::size_t rowIndex = 0; rowIndex < sw.connectivityTable.size();
       ++rowIndex) {
    const std::string &row = sw.connectivityTable[rowIndex];
    if (row.size() != sw.inputs.size())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "ADG switch connectivity row %zu has width %zu, expected %zu",
          rowIndex, row.size(), sw.inputs.size());
    bool rowHasConnection = false;
    for (std::size_t column = 0; column < row.size(); ++column) {
      if (row[column] != '0' && row[column] != '1')
        return llvm::createStringError(
            std::errc::invalid_argument,
            "ADG switch connectivity row %zu contains non-binary entry",
            rowIndex);
      if (row[column] == '1') {
        rowHasConnection = true;
        columnHasConnection[column] = true;
      }
    }
    if (!rowHasConnection)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "ADG switch connectivity row %zu has no connection", rowIndex);
  }
  for (std::size_t column = 0; column < columnHasConnection.size(); ++column) {
    if (!columnHasConnection[column])
      return llvm::createStringError(
          std::errc::invalid_argument,
          "ADG switch input column %zu has no connection", column);
  }
  return llvm::Error::success();
}

llvm::Error validateMem(const MemSpec &mem) {
  if (mem.managerInputs.empty())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "ADG mem has no manager endpoints");
  if (llvm::any_of(mem.managerInputs,
                   [](const std::string &manager) { return manager.empty(); }))
    return llvm::createStringError(std::errc::invalid_argument,
                                   "ADG mem manager endpoint is empty");
  for (const MemSubordinateOutput &subordinate : mem.subordinateOutputs)
    if (subordinate.name.empty() || subordinate.type.empty())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "ADG mem subordinate output capability is incomplete");

  unsigned physicalPortCount =
      static_cast<unsigned>(mem.loads.size() + mem.stores.size());
  if (physicalPortCount == 0)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "ADG mem has no operation ports");
  if (mem.dataWidth == 0)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "ADG mem requires operation data width");

  auto validateDomains = [&](llvm::StringRef name,
                             const std::vector<std::vector<unsigned>> &domains,
                             std::size_t sourceCount,
                             llvm::StringRef sourceCountName) -> llvm::Error {
    if (domains.size() != sourceCount)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "ADG mem %s length %zu must equal %s %zu", name.str().c_str(),
          domains.size(), sourceCountName.str().c_str(), sourceCount);
    for (auto [source, domain] : llvm::enumerate(domains)) {
      if (domain.empty())
        return llvm::createStringError(std::errc::invalid_argument,
                                       "ADG mem %s entry #%zu is empty",
                                       name.str().c_str(), source);
      std::optional<unsigned> previous;
      for (unsigned manager : domain) {
        if (manager >= mem.managerInputs.size())
          return llvm::createStringError(
              std::errc::invalid_argument,
              "ADG mem %s entry #%zu manager target %u is outside [0, %zu)",
              name.str().c_str(), source, manager, mem.managerInputs.size());
        if (previous && manager <= *previous)
          return llvm::createStringError(
              std::errc::invalid_argument,
              "ADG mem %s entry #%zu is not strictly increasing",
              name.str().c_str(), source);
        previous = manager;
      }
    }
    return llvm::Error::success();
  };
  if (llvm::Error err =
          validateDomains("operation_port_requests",
                          mem.dispatchEligibility.operationPortRequests,
                          physicalPortCount, "physical operation port count"))
    return err;
  if (llvm::Error err = validateDomains(
          "subordinate_requests", mem.dispatchEligibility.subordinateRequests,
          mem.subordinateOutputs.size(), "subordinate endpoint count"))
    return err;

  if (mem.schedule == Schedule::Spatial) {
    if (mem.temporalTagWidth != 0 || mem.temporalOperationTableSize != 0)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "spatial ADG mem must not carry temporal hardware capability");
    return llvm::Error::success();
  }

  if (mem.temporalTagWidth == 0)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "temporal ADG mem requires tag width");
  if (mem.temporalOperationTableSize == 0)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "temporal ADG mem requires operation table size");

  uint64_t representableRows = std::numeric_limits<uint64_t>::max();
  if (mem.temporalTagWidth < std::numeric_limits<uint64_t>::digits) {
    uint64_t tagCount = uint64_t{1} << mem.temporalTagWidth;
    if (physicalPortCount <= std::numeric_limits<uint64_t>::max() / tagCount)
      representableRows = static_cast<uint64_t>(physicalPortCount) * tagCount;
  }
  if (mem.temporalOperationTableSize > representableRows)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "temporal ADG mem operation table size %u exceeds representable row "
        "capacity %llu",
        mem.temporalOperationTableSize,
        static_cast<unsigned long long>(representableRows));
  return llvm::Error::success();
}

std::string memDataType(const MemSpec &mem) {
  if (mem.schedule == Schedule::Temporal)
    return (llvm::Twine("!fabric.bits_tag<") + llvm::Twine(mem.dataWidth) +
            ", " + llvm::Twine(mem.temporalTagWidth) + ">")
        .str();
  return (llvm::Twine("!fabric.bits<") + llvm::Twine(mem.dataWidth) + ">")
      .str();
}

void printMemManagerTargetDomains(
    llvm::raw_ostream &os, const std::vector<std::vector<unsigned>> &domains) {
  os << '[';
  for (auto [source, domain] : llvm::enumerate(domains)) {
    if (source)
      os << ", ";
    os << '[';
    for (auto [index, manager] : llvm::enumerate(domain)) {
      if (index)
        os << ", ";
      os << manager << " : i32";
    }
    os << ']';
  }
  os << ']';
}

void printMemDispatchEligibility(llvm::raw_ostream &os, const MemSpec &mem) {
  os << "{operation_port_requests = ";
  printMemManagerTargetDomains(os,
                               mem.dispatchEligibility.operationPortRequests);
  os << ", subordinate_requests = ";
  printMemManagerTargetDomains(os, mem.dispatchEligibility.subordinateRequests);
  os << '}';
}

} // namespace

ModuleBuilder::ModuleBuilder(std::string name) : name(std::move(name)) {}

ModuleBuilder &ModuleBuilder::addInput(std::string inputName,
                                       std::string type) {
  inputs.push_back(Input{std::move(inputName), std::move(type)});
  return *this;
}

ModuleBuilder &ModuleBuilder::addOutput(std::string sourceName) {
  std::size_t useId = registerDirectUse(std::move(sourceName));
  outputs.push_back(Output{useId});
  return *this;
}

std::size_t ModuleBuilder::registerDirectUse(std::string sourceName) {
  directUses.push_back(DirectUse{canonicalValueName(sourceName)});
  return directUses.size() - 1;
}

ModuleBuilder &ModuleBuilder::addPe(PeSpec pe) {
  std::vector<std::size_t> useIds;
  useIds.reserve(pe.inputs.size());
  for (const PortBinding &input : pe.inputs)
    useIds.push_back(registerDirectUse(input.sourceName));
  pes.push_back(PeEntry{std::move(pe), std::move(useIds)});
  return *this;
}

ModuleBuilder &ModuleBuilder::addSwitch(SwitchSpec sw) {
  std::vector<std::size_t> useIds;
  useIds.reserve(sw.inputs.size());
  for (const std::string &input : sw.inputs)
    useIds.push_back(registerDirectUse(input));
  switches.push_back(SwitchEntry{std::move(sw), std::move(useIds)});
  return *this;
}

ModuleBuilder &ModuleBuilder::addFifo(FifoSpec fifo) {
  std::size_t useId = registerDirectUse(fifo.sourceName);
  bodyEntries.emplace_back(FifoEntry{std::move(fifo), useId});
  return *this;
}

ModuleBuilder &ModuleBuilder::addBoundary(BoundarySpec boundary) {
  std::vector<std::size_t> useIds;
  useIds.reserve(boundary.inputs.size());
  for (const BoundaryInput &input : boundary.inputs)
    useIds.push_back(registerDirectUse(input.sourceName));
  bodyEntries.emplace_back(
      BoundaryEntry{std::move(boundary), std::move(useIds)});
  return *this;
}

ModuleBuilder &ModuleBuilder::addMem(MemSpec mem) {
  std::vector<std::size_t> useIds;
  useIds.reserve(mem.managerInputs.size() + mem.loads.size() * 2 +
                 mem.stores.size() * 3);
  for (const std::string &manager : mem.managerInputs)
    useIds.push_back(registerDirectUse(manager));
  for (const MemLoadPort &load : mem.loads) {
    useIds.push_back(registerDirectUse(load.address));
    useIds.push_back(registerDirectUse(load.control));
  }
  for (const MemStorePort &store : mem.stores) {
    useIds.push_back(registerDirectUse(store.address));
    useIds.push_back(registerDirectUse(store.data));
    useIds.push_back(registerDirectUse(store.control));
  }
  mems.push_back(MemEntry{std::move(mem), std::move(useIds)});
  return *this;
}

ModuleBuilder &ModuleBuilder::addBodyOp(BodyOpSpec op) {
  std::vector<std::vector<std::size_t>> lineUseIds;
  lineUseIds.reserve(op.lines.size());
  for (const BodyLineSpec &line : op.lines) {
    std::vector<std::size_t> useIds;
    useIds.reserve(line.operands.size());
    for (const std::string &operand : line.operands)
      useIds.push_back(registerDirectUse(operand));
    lineUseIds.push_back(std::move(useIds));
  }
  bodyEntries.emplace_back(BodyOpEntry{std::move(op), std::move(lineUseIds)});
  return *this;
}

ModuleBuilder &ModuleBuilder::addAttribute(std::string attrName,
                                           std::string value) {
  attributes.push_back(Attribute{std::move(attrName), std::move(value)});
  return *this;
}

SystemBuilder::SystemBuilder(std::string name, std::string memoryModel)
    : name(std::move(name)), memoryModel(std::move(memoryModel)) {}

namespace {

SystemNodeSpec makeSystemNode(std::string name, std::string kind,
                              std::vector<std::string> ports) {
  SystemNodeSpec node;
  node.name = std::move(name);
  node.kind = std::move(kind);
  node.ports = std::move(ports);
  return node;
}

} // namespace

SystemBuilder &SystemBuilder::addHostCore(std::string nodeName,
                                          std::string scalar,
                                          std::vector<std::string> ports) {
  SystemNodeSpec node =
      makeSystemNode(std::move(nodeName), "host_core", std::move(ports));
  node.scalar = std::move(scalar);
  nodes.push_back(std::move(node));
  return *this;
}

SystemBuilder &SystemBuilder::addSpatialAccelerator(
    std::string nodeName, std::string spatialModule, std::string scalar,
    std::vector<std::string> ports) {
  SystemNodeSpec node =
      makeSystemNode(std::move(nodeName), "acc_core", std::move(ports));
  node.spatialModule = std::move(spatialModule);
  node.scalar = std::move(scalar);
  nodes.push_back(std::move(node));
  return *this;
}

SystemBuilder &
SystemBuilder::addFixedAccelerator(std::string nodeName, std::string function,
                                   std::vector<std::string> ports) {
  SystemNodeSpec node = makeSystemNode(std::move(nodeName), "fixed_accelerator",
                                       std::move(ports));
  node.function = std::move(function);
  nodes.push_back(std::move(node));
  return *this;
}

SystemBuilder &SystemBuilder::addCache(std::string nodeName,
                                       std::uint64_t lineBytes,
                                       std::uint64_t capacityBytes,
                                       std::vector<std::string> ports) {
  SystemNodeSpec node =
      makeSystemNode(std::move(nodeName), "cache", std::move(ports));
  node.params = {{"capacity_bytes", capacityBytes}, {"line_bytes", lineBytes}};
  nodes.push_back(std::move(node));
  return *this;
}

SystemBuilder &SystemBuilder::addDmaEngine(std::string nodeName,
                                           std::uint64_t queueDepth,
                                           std::vector<std::string> ports) {
  SystemNodeSpec node =
      makeSystemNode(std::move(nodeName), "dma_engine", std::move(ports));
  node.params = {{"queue_depth", queueDepth}};
  nodes.push_back(std::move(node));
  return *this;
}

SystemBuilder &SystemBuilder::addMemory(std::string nodeName,
                                        std::uint64_t bytes,
                                        std::vector<std::string> ports) {
  SystemNodeSpec node =
      makeSystemNode(std::move(nodeName), "memory", std::move(ports));
  node.bytes = bytes;
  nodes.push_back(std::move(node));
  return *this;
}

SystemBuilder &SystemBuilder::connect(std::string srcNode, std::string srcPort,
                                      std::string srcChannel,
                                      std::string dstNode, std::string dstPort,
                                      std::string dstChannel) {
  links.push_back(SystemLinkSpec{std::move(srcNode), std::move(srcPort),
                                 std::move(srcChannel), std::move(dstNode),
                                 std::move(dstPort), std::move(dstChannel)});
  return *this;
}

ModuleBuilder &ModuleBuilder::addUnsupportedResource(std::string detail) {
  if (!llvm::is_contained(unsupportedResources, detail))
    unsupportedResources.push_back(std::move(detail));
  return *this;
}

llvm::Error ModuleBuilder::print(llvm::raw_ostream &destination) const {
  if (name.empty())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "ADG module name is empty");
  // A catalog resource the normative implementation-family registry cannot
  // express fails construction here. Emitting it would produce Fabric that no
  // verifier accepts, and inventing a family for it would make the registry
  // follow the helper instead of the hardware.
  if (!unsupportedResources.empty())
    return llvm::createStringError(
        std::errc::not_supported,
        "ADG target '%s' requires %zu resource(s) with no registered "
        "implementation family: %s",
        name.c_str(), unsupportedResources.size(),
        llvm::join(unsupportedResources, "; ").c_str());
  llvm::StringSet<> seenInputs;
  llvm::StringSet<> valueNames;
  llvm::StringMap<std::string> valueTypes;
  auto defineValue = [&](llvm::StringRef valueName,
                         llvm::StringRef type) -> llvm::Error {
    std::string canonicalName = canonicalValueName(valueName);
    if (canonicalName.empty() || type.empty())
      return llvm::createStringError(std::errc::invalid_argument,
                                     "ADG body value is incomplete");
    if (!valueNames.insert(canonicalName).second)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "duplicate ADG body value %s",
                                     canonicalName.c_str());
    valueTypes[canonicalName] = type;
    return llvm::Error::success();
  };
  for (const Input &input : inputs) {
    if (input.name.empty() || input.type.empty())
      return llvm::createStringError(std::errc::invalid_argument,
                                     "ADG module input is incomplete");
    std::string inputName = canonicalValueName(input.name);
    if (!seenInputs.insert(inputName).second)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "duplicate ADG module input %s",
                                     inputName.c_str());
    if (llvm::Error err = defineValue(inputName, input.type))
      return err;
  }
  llvm::StringSet<> seenAttributes;
  for (const Attribute &attribute : attributes) {
    if (attribute.name.empty() || attribute.value.empty())
      return llvm::createStringError(std::errc::invalid_argument,
                                     "ADG module attribute is incomplete");
    if (!seenAttributes.insert(attribute.name).second)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "duplicate ADG module attribute %s",
                                     attribute.name.c_str());
  }

  for (const PeEntry &entry : pes) {
    const PeSpec &pe = entry.spec;
    if (!pe.resultNames.empty() &&
        pe.resultNames.size() != pe.resultTypes.size())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "ADG pe result name count must match result type count");
    if (entry.useIds.size() != pe.inputs.size())
      return llvm::createStringError(std::errc::invalid_argument,
                                     "ADG pe direct use count is invalid");
    if (pe.schedule == Schedule::Temporal)
      if (llvm::Error err = validateTemporalPe(pe.temporal))
        return err;
    for (const FuSpec &fu : pe.fus)
      if (llvm::Error err = validateFu(fu))
        return err;
    for (auto [resultName, resultType] :
         llvm::zip(pe.resultNames, pe.resultTypes))
      if (llvm::Error err = defineValue(resultName, resultType))
        return err;
  }
  for (auto [switchIndex, entry] : llvm::enumerate(switches)) {
    const SwitchSpec &sw = entry.spec;
    if (entry.useIds.size() != sw.inputs.size())
      return llvm::createStringError(std::errc::invalid_argument,
                                     "ADG switch direct use count is invalid");
    for (auto [resultIndex, resultType] : llvm::enumerate(sw.resultTypes)) {
      std::string resultName = (llvm::Twine("sw") + llvm::Twine(switchIndex) +
                                "_out" + llvm::Twine(resultIndex))
                                   .str();
      if (llvm::Error err = defineValue(resultName, resultType))
        return err;
    }
  }
  for (const BodyEntry &body : bodyEntries) {
    if (const auto *entry = std::get_if<BodyOpEntry>(&body)) {
      const BodyOpSpec &op = entry->spec;
      if (!op.results.empty() && op.lines.empty())
        return llvm::createStringError(
            std::errc::invalid_argument,
            "ADG body op with results must have an operation line");
      if (entry->lineUseIds.size() != op.lines.size())
        return llvm::createStringError(
            std::errc::invalid_argument,
            "ADG body op direct use line count is invalid");
      for (auto [line, useIds] : llvm::zip(op.lines, entry->lineUseIds)) {
        if (line.fragments.size() != line.operands.size() + 1)
          return llvm::createStringError(
              std::errc::invalid_argument,
              "ADG body line fragment count must be one greater than operand "
              "count");
        if (useIds.size() != line.operands.size())
          return llvm::createStringError(
              std::errc::invalid_argument,
              "ADG body line direct use count is invalid");
        if (line.moduleScope)
          for (const std::string &fragment : line.fragments)
            if (fragmentHidesDirectUse(fragment))
              return llvm::createStringError(
                  std::errc::invalid_argument,
                  "ADG body literal fragment hides a module SSA value; "
                  "direct uses must be declared as operands");
      }
      for (const BodyResultSpec &result : op.results)
        if (llvm::Error err = defineValue(result.name, result.type))
          return err;
      continue;
    }

    if (const auto *entry = std::get_if<FifoEntry>(&body)) {
      if (llvm::Error err = validateFifoSpec(entry->spec))
        return err;
      if (llvm::Error err =
              defineValue(entry->spec.resultName, entry->spec.resultType))
        return err;
      continue;
    }

    const auto &entry = std::get<BoundaryEntry>(body);
    if (llvm::Error err = validateBoundarySpec(entry.spec))
      return err;
    if (entry.useIds.size() != entry.spec.inputs.size())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "ADG boundary direct use count is invalid");
    for (auto [resultName, resultType] :
         llvm::zip(entry.spec.resultNames, entry.spec.resultTypes))
      if (llvm::Error err = defineValue(resultName, resultType))
        return err;
  }
  for (auto [memIndex, entry] : llvm::enumerate(mems)) {
    const MemSpec &mem = entry.spec;
    if (llvm::Error err = validateMem(mem))
      return err;
    std::size_t expectedUseCount =
        mem.managerInputs.size() + mem.loads.size() * 2 + mem.stores.size() * 3;
    if (entry.useIds.size() != expectedUseCount)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "ADG mem direct use count is invalid");
    for (const MemSubordinateOutput &subordinate : mem.subordinateOutputs)
      if (llvm::Error err = defineValue(subordinate.name, subordinate.type))
        return err;
    std::size_t useIndex = mem.managerInputs.size();
    for (std::size_t i = 0; i < mem.loads.size(); ++i) {
      std::string addressType =
          valueTypes.lookup(directUses[entry.useIds[useIndex]].sourceName);
      std::string controlType =
          valueTypes.lookup(directUses[entry.useIds[useIndex + 1]].sourceName);
      if (addressType.empty() || controlType.empty())
        return llvm::createStringError(std::errc::invalid_argument,
                                       "ADG mem load source is unknown");
      if (llvm::Error err =
              defineValue((llvm::Twine("mem") + llvm::Twine(memIndex) +
                           "_data" + llvm::Twine(i))
                              .str(),
                          memDataType(mem)))
        return err;
      if (llvm::Error err =
              defineValue((llvm::Twine("mem") + llvm::Twine(memIndex) +
                           "_done" + llvm::Twine(i))
                              .str(),
                          controlType))
        return err;
      useIndex += 2;
    }
    for (std::size_t i = 0; i < mem.stores.size(); ++i) {
      std::string controlType =
          valueTypes.lookup(directUses[entry.useIds[useIndex + 2]].sourceName);
      if (controlType.empty())
        return llvm::createStringError(std::errc::invalid_argument,
                                       "ADG mem store source is unknown");
      if (llvm::Error err =
              defineValue((llvm::Twine("mem") + llvm::Twine(memIndex) +
                           "_store_done" + llvm::Twine(i))
                              .str(),
                          controlType))
        return err;
      useIndex += 3;
    }
  }

  for (const BodyEntry &body : bodyEntries) {
    if (const auto *entry = std::get_if<FifoEntry>(&body)) {
      llvm::StringRef sourceName = directUses[entry->useId].sourceName;
      if (!valueTypes.contains(sourceName))
        return llvm::createStringError(std::errc::invalid_argument,
                                       "ADG fifo source %s is unknown",
                                       sourceName.str().c_str());
      continue;
    }
    const auto *entry = std::get_if<BoundaryEntry>(&body);
    if (!entry)
      continue;
    for (std::size_t useId : entry->useIds) {
      llvm::StringRef sourceName = directUses[useId].sourceName;
      if (!valueTypes.contains(sourceName))
        return llvm::createStringError(std::errc::invalid_argument,
                                       "ADG boundary source %s is unknown",
                                       sourceName.str().c_str());
    }
  }

  llvm::SmallVector<std::string> outputTypes;
  outputTypes.reserve(outputs.size());
  for (const Output &output : outputs) {
    std::string sourceType =
        valueTypes.lookup(directUses[output.useId].sourceName);
    if (sourceType.empty())
      return llvm::createStringError(std::errc::invalid_argument,
                                     "ADG module output source is unknown");
    outputTypes.push_back(std::move(sourceType));
  }

  for (const SwitchEntry &entry : switches)
    if (llvm::Error err = validateSwitch(entry.spec, valueTypes))
      return err;

  std::vector<std::string> useSources;
  useSources.reserve(directUses.size());
  for (const DirectUse &use : directUses)
    useSources.push_back(use.sourceName);
  auto planOrErr =
      buildTransportPlan(useSources, valueTypes, std::move(valueNames));
  if (!planOrErr)
    return planOrErr.takeError();
  TransportPlan plan = std::move(*planOrErr);

  ::mlir::MLIRContext capabilityContext;
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "fabric.module @" << name << '(';
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    if (i)
      os << ",\n                                    ";
    os << valueName(inputs[i].name) << " : " << inputs[i].type;
  }
  os << ')';
  if (!outputs.empty()) {
    os << "\n    -> (";
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      if (i)
        os << ", ";
      os << outputTypes[i];
    }
    os << ')';
  }
  if (!attributes.empty()) {
    os << " attributes {";
    for (std::size_t i = 0; i < attributes.size(); ++i) {
      if (i)
        os << ", ";
      os << attributes[i].name << " = " << attributes[i].value;
    }
    os << '}';
  }
  os << " {\n";
  for (const PeEntry &entry : pes)
    printPe(os, capabilityContext, entry.spec, entry.useIds, plan.resolvedUses);
  for (std::size_t switchIndex = 0; switchIndex < switches.size();
       ++switchIndex) {
    const SwitchEntry &entry = switches[switchIndex];
    const SwitchSpec &sw = entry.spec;
    llvm::SmallVector<std::string> operandTypes;
    for (std::size_t useId : entry.useIds)
      operandTypes.push_back(valueTypes.lookup(directUses[useId].sourceName));
    printSwitch(os, sw, switchIndex, entry.useIds, plan.resolvedUses,
                operandTypes);
  }
  for (std::size_t memIndex = 0; memIndex < mems.size(); ++memIndex) {
    const MemEntry &entry = mems[memIndex];
    const MemSpec &mem = entry.spec;
    std::size_t useIndex = 0;
    os << "  ";
    bool hasResult = false;
    for (const MemSubordinateOutput &subordinate : mem.subordinateOutputs) {
      if (hasResult)
        os << ", ";
      os << valueName(subordinate.name);
      hasResult = true;
    }
    for (std::size_t i = 0; i < mem.loads.size(); ++i) {
      if (hasResult)
        os << ", ";
      os << "%mem" << memIndex << "_data" << i << ", %mem" << memIndex
         << "_done" << i;
      hasResult = true;
    }
    for (std::size_t i = 0; i < mem.stores.size(); ++i) {
      if (hasResult)
        os << ", ";
      os << "%mem" << memIndex << "_store_done" << i;
      hasResult = true;
    }
    os << " = fabric.mem [" << scheduleName(mem.schedule) << "] mgr(";
    for (std::size_t i = 0; i < mem.managerInputs.size(); ++i) {
      if (i)
        os << ", ";
      os << valueName(plan.resolvedUses[entry.useIds[useIndex++]]);
    }
    os << ')';
    if (!mem.loads.empty()) {
      os << " load(";
      for (std::size_t i = 0; i < mem.loads.size(); ++i) {
        if (i)
          os << ", ";
        os << valueName(plan.resolvedUses[entry.useIds[useIndex++]]) << ", "
           << valueName(plan.resolvedUses[entry.useIds[useIndex++]]);
      }
      os << ')';
    }
    if (!mem.stores.empty()) {
      os << " store(";
      for (std::size_t i = 0; i < mem.stores.size(); ++i) {
        if (i)
          os << ", ";
        os << valueName(plan.resolvedUses[entry.useIds[useIndex++]]) << ", "
           << valueName(plan.resolvedUses[entry.useIds[useIndex++]]) << ", "
           << valueName(plan.resolvedUses[entry.useIds[useIndex++]]);
      }
      os << ')';
    }
    os << "\n        [{load_group_size = "
       << static_cast<unsigned>(mem.loads.size())
       << " : i32, store_group_size = "
       << static_cast<unsigned>(mem.stores.size())
       << " : i32, data_width = " << mem.dataWidth << " : i32";
    if (mem.schedule == Schedule::Temporal) {
      os << ", tag_width = " << mem.temporalTagWidth
         << " : i32, operation_table_size = " << mem.temporalOperationTableSize
         << " : i32";
    }
    os << ", dispatch_eligibility = ";
    printMemDispatchEligibility(os, mem);
    os << "}]\n";

    llvm::SmallVector<std::string> operandTypes;
    for (std::size_t useId : entry.useIds)
      operandTypes.push_back(valueTypes.lookup(directUses[useId].sourceName));
    llvm::SmallVector<std::string> resultTypes;
    for (const MemSubordinateOutput &subordinate : mem.subordinateOutputs)
      resultTypes.push_back(subordinate.type);
    useIndex = mem.managerInputs.size();
    for (const MemLoadPort &load : mem.loads) {
      (void)load;
      ++useIndex;
      resultTypes.push_back(memDataType(mem));
      resultTypes.push_back(
          valueTypes.lookup(directUses[entry.useIds[useIndex++]].sourceName));
    }
    for (const MemStorePort &store : mem.stores) {
      (void)store;
      useIndex += 2;
      resultTypes.push_back(
          valueTypes.lookup(directUses[entry.useIds[useIndex++]].sourceName));
    }
    os << "        : ";
    printTypeList(os, operandTypes);
    os << "\n        -> ";
    printResultTypes(os, resultTypes);
    os << '\n';
  }
  for (const BodyEntry &body : bodyEntries) {
    if (const auto *entry = std::get_if<BodyOpEntry>(&body)) {
      for (std::size_t lineIndex = 0; lineIndex < entry->spec.lines.size();
           ++lineIndex) {
        const BodyLineSpec &line = entry->spec.lines[lineIndex];
        llvm::ArrayRef<std::size_t> useIds = entry->lineUseIds[lineIndex];
        os << "  ";
        if (lineIndex == 0 && !entry->spec.results.empty()) {
          for (std::size_t i = 0; i < entry->spec.results.size(); ++i) {
            if (i)
              os << ", ";
            os << valueName(entry->spec.results[i].name);
          }
          os << " = ";
        }
        os << line.fragments.front();
        for (std::size_t i = 0; i < line.operands.size(); ++i)
          os << valueName(plan.resolvedUses[useIds[i]])
             << line.fragments[i + 1];
        os << '\n';
      }
      continue;
    }

    if (const auto *entry = std::get_if<FifoEntry>(&body)) {
      std::string sourceType =
          valueTypes.lookup(directUses[entry->useId].sourceName);
      printFifo(os, entry->spec, plan.resolvedUses[entry->useId], sourceType);
      continue;
    }

    const auto &entry = std::get<BoundaryEntry>(body);
    llvm::SmallVector<std::string> resolvedSources;
    llvm::SmallVector<std::string> sourceTypes;
    resolvedSources.reserve(entry.useIds.size());
    sourceTypes.reserve(entry.useIds.size());
    for (std::size_t useId : entry.useIds) {
      resolvedSources.push_back(plan.resolvedUses[useId]);
      sourceTypes.push_back(valueTypes.lookup(directUses[useId].sourceName));
    }
    printBoundary(os, entry.spec, resolvedSources, sourceTypes);
  }
  // Resolve original top-level source names before parsing generated fanouts.
  for (const TransportFanout &fanout : plan.fanouts)
    printTransportFanout(os, fanout);
  os << "  fabric.yield";
  if (!outputs.empty()) {
    os << ' ';
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      if (i)
        os << ", ";
      os << valueName(plan.resolvedUses[outputs[i].useId]);
    }
    os << " : ";
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      if (i)
        os << ", ";
      os << outputTypes[i];
    }
  }
  os << '\n';
  os << "}\n";
  os.flush();
  if (llvm::Error err = validateFabricModule(text))
    return err;
  destination << text;
  return llvm::Error::success();
}

llvm::Error SystemBuilder::print(llvm::raw_ostream &os) const {
  if (name.empty())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "ADG system name is empty");
  if (memoryModel.empty())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "ADG system memory model is empty");
  llvm::StringSet<> nodeNames;
  for (const SystemNodeSpec &node : nodes) {
    if (node.name.empty() || node.kind.empty() || node.ports.empty())
      return llvm::createStringError(std::errc::invalid_argument,
                                     "ADG system node is incomplete");
    if (!nodeNames.insert(node.name).second)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "duplicate ADG system node %s",
                                     node.name.c_str());
  }
  for (const SystemLinkSpec &link : links) {
    if (!nodeNames.contains(link.srcNode))
      return llvm::createStringError(std::errc::invalid_argument,
                                     "ADG system link source %s is unknown",
                                     link.srcNode.c_str());
    if (!nodeNames.contains(link.dstNode))
      return llvm::createStringError(
          std::errc::invalid_argument,
          "ADG system link destination %s is unknown", link.dstNode.c_str());
  }

  os << "fabric.system @" << name << " memory_model = \"" << memoryModel
     << "\" {\n";
  for (const SystemNodeSpec &node : nodes) {
    os << "  fabric.node @" << node.name << " kind = \"" << node.kind << "\"\n"
       << "      ports = ";
    printStringArray(os, node.ports);
    os << " attributes {";
    bool firstAttr = true;
    auto printComma = [&]() {
      if (!firstAttr)
        os << ", ";
      firstAttr = false;
    };
    if (!node.spatialModule.empty()) {
      printComma();
      os << "spatial = @" << node.spatialModule;
    }
    if (!node.scalar.empty()) {
      printComma();
      os << "scalar = \"" << node.scalar << "\"";
    }
    if (!node.function.empty()) {
      printComma();
      os << "function = \"" << node.function << "\"";
    }
    if (node.bytes) {
      printComma();
      os << "bytes = " << *node.bytes << " : i64";
    }
    if (!node.params.empty()) {
      printComma();
      os << "params = {";
      bool firstParam = true;
      for (const auto &[key, value] : node.params) {
        if (!firstParam)
          os << ", ";
        firstParam = false;
        os << key << " = " << value << " : i64";
      }
      os << '}';
    }
    os << "}\n";
  }
  if (!links.empty())
    os << '\n';
  for (const SystemLinkSpec &link : links) {
    os << "  fabric.link src = @" << link.srcNode << " src_port = \""
       << link.srcPort << "\""
       << " src_channel = \"" << link.srcChannel << "\""
       << " dst = @" << link.dstNode << " dst_port = \"" << link.dstPort
       << "\" dst_channel = \"" << link.dstChannel << "\"\n";
  }
  os << "}\n";
  return llvm::Error::success();
}
