#include "ADG/Builder.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <initializer_list>
#include <iterator>
#include <optional>
#include <system_error>

using namespace loom::adg;

namespace {

llvm::StringRef scheduleName(Schedule schedule) {
  switch (schedule) {
  case Schedule::Spatial:
    return "spatial";
  case Schedule::Temporal:
    return "temporal";
  }
  llvm_unreachable("unknown ADG schedule");
}

std::string valueName(llvm::StringRef name) {
  if (name.consume_front("%"))
    return ("%" + name).str();
  return ("%" + name).str();
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

void printHwParams(
    llvm::raw_ostream &os,
    const std::map<std::string, std::vector<std::string>> &hwParams) {
  if (hwParams.empty())
    return;
  os << "hw_params = [{";
  bool first = true;
  for (const auto &[key, values] : hwParams) {
    if (!first)
      os << ", ";
    first = false;
    os << key << " = ";
    printStringArray(os, values);
  }
  os << "}]";
}

void printSwConfigs(llvm::raw_ostream &os,
                    const std::map<std::string, std::string> &swConfigs) {
  if (swConfigs.empty())
    return;
  os << "sw_configs = {";
  bool first = true;
  for (const auto &[key, value] : swConfigs) {
    if (!first)
      os << ", ";
    first = false;
    os << key << " = " << '"' << value << '"';
  }
  os << '}';
}

void printOpAttrs(llvm::raw_ostream &os, const FabricOpSpec &op) {
  if (op.hwParams.empty() && op.swConfigs.empty())
    return;
  os << " {";
  if (!op.hwParams.empty())
    printHwParams(os, op.hwParams);
  if (!op.hwParams.empty() && !op.swConfigs.empty())
    os << ", ";
  if (!op.swConfigs.empty())
    printSwConfigs(os, op.swConfigs);
  os << '}';
}

void printFabricOp(llvm::raw_ostream &os, const FabricOpSpec &op) {
  os << "      ";
  for (std::size_t i = 0; i < op.results.size(); ++i) {
    if (i)
      os << ", ";
    os << valueName(op.results[i]);
  }
  if (!op.results.empty())
    os << " = ";
  os << "fabric.op [";
  for (std::size_t i = 0; i < op.opList.size(); ++i) {
    if (i)
      os << ", ";
    os << '@' << op.opList[i];
  }
  os << "] (";
  for (std::size_t i = 0; i < op.operands.size(); ++i) {
    if (i)
      os << ", ";
    os << valueName(op.operands[i]);
  }
  os << ')';
  printOpAttrs(os, op);
  os << " : ";
  printTypeList(os, op.operandTypes);
  os << " -> ";
  printResultTypes(os, op.resultTypes);
  os << '\n';
}

void printFu(llvm::raw_ostream &os, const FuSpec &fu) {
  os << "    fabric.fu(";
  printBindings(os, fu.inputs, "              ");
  os << ") -> ";
  printResultTypes(os, fu.resultTypes);
  os << " {\n";
  for (const FabricOpSpec &op : fu.operations)
    printFabricOp(os, op);
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
     << "         operand_buffer_mode = \"" << config.operandBufferMode << "\"";
  if (config.operandBufferSize)
    os << ",\n         operand_buffer_size = " << config.operandBufferSize
       << " : i32";
  os << "\n       }";
}

void printPe(llvm::raw_ostream &os, const PeSpec &pe) {
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
  printBindings(os, pe.inputs, "                    ");
  os << ") -> ";
  printResultTypes(os, pe.resultTypes);
  if (pe.schedule == Schedule::Temporal)
    printTemporalPeAttributes(os, pe.temporal);
  os << " {\n";
  for (const FuSpec &fu : pe.fus)
    printFu(os, fu);
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
                 std::size_t switchIndex,
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
    const std::string &input = sw.inputs[i];
    os << ' ' << valueName(input);
  }
  os << "\n         ";
  printSwitchHwParams(os, sw);
  os << "\n         : ";
  printTypeList(os, operandTypes);
  os << "\n        -> ";
  printResultTypes(os, sw.resultTypes);
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
    if (!inputTypes.contains(input))
      return llvm::createStringError(std::errc::invalid_argument,
                                     "ADG switch input %s is unknown",
                                     input.c_str());
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

PeSpec makeMinimalAddPe(Schedule schedule, std::string lhsSource,
                        std::string rhsSource, std::string boundaryType,
                        std::string fuType,
                        TemporalPeConfig temporal = TemporalPeConfig()) {
  PeSpec pe;
  pe.schedule = schedule;
  pe.inputs = {{"pa", std::move(lhsSource), boundaryType, ""},
               {"pb", std::move(rhsSource), boundaryType, ""}};
  pe.resultTypes = {boundaryType};
  pe.temporal = std::move(temporal);

  FuSpec addFu;
  addFu.inputs = {{"fa", "pa", fuType, ""}, {"fb", "pb", fuType, ""}};
  addFu.resultTypes = {fuType};
  addFu.operations.push_back(FabricOpSpec{{"sum"},
                                          {"arith.addi"},
                                          {"fa", "fb"},
                                          {fuType, fuType},
                                          {fuType},
                                          {},
                                          {}});
  addFu.yieldValues = {"sum"};
  pe.fus.push_back(std::move(addFu));
  return pe;
}

PeSpec makeMinimalAddPe(Schedule schedule, std::string boundaryType,
                        std::string fuType,
                        TemporalPeConfig temporal = TemporalPeConfig()) {
  return makeMinimalAddPe(schedule, "lhs", "rhs", std::move(boundaryType),
                          std::move(fuType), std::move(temporal));
}

} // namespace

ModuleBuilder::ModuleBuilder(std::string name) : name(std::move(name)) {}

ModuleBuilder &ModuleBuilder::addInput(std::string inputName,
                                       std::string type) {
  inputs.push_back(Input{std::move(inputName), std::move(type)});
  return *this;
}

ModuleBuilder &ModuleBuilder::addPe(PeSpec pe) {
  pes.push_back(std::move(pe));
  return *this;
}

ModuleBuilder &ModuleBuilder::addSwitch(SwitchSpec sw) {
  switches.push_back(std::move(sw));
  return *this;
}

ModuleBuilder &ModuleBuilder::addMem(MemSpec mem) {
  mems.push_back(std::move(mem));
  return *this;
}

ModuleBuilder &ModuleBuilder::addExactBodyLine(std::string line) {
  exactBodyLines.push_back(std::move(line));
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

llvm::Error ModuleBuilder::print(llvm::raw_ostream &os) const {
  if (name.empty())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "ADG module name is empty");
  llvm::StringSet<> seenInputs;
  llvm::StringMap<std::string> inputTypes;
  for (const Input &input : inputs) {
    if (input.name.empty() || input.type.empty())
      return llvm::createStringError(std::errc::invalid_argument,
                                     "ADG module input is incomplete");
    if (!seenInputs.insert(input.name).second)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "duplicate ADG module input %s",
                                     input.name.c_str());
    inputTypes[input.name] = input.type;
  }

  os << "fabric.module @" << name << '(';
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    if (i)
      os << ",\n                                    ";
    os << valueName(inputs[i].name) << " : " << inputs[i].type;
  }
  os << ") {\n";
  for (const PeSpec &pe : pes)
    if (!pe.resultNames.empty() &&
        pe.resultNames.size() != pe.resultTypes.size())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "ADG pe result name count must match result type count");
  for (const PeSpec &pe : pes)
    for (const FuSpec &fu : pe.fus)
      if (llvm::Error err = validateFu(fu))
        return err;
  for (const PeSpec &pe : pes)
    printPe(os, pe);
  for (std::size_t switchIndex = 0; switchIndex < switches.size();
       ++switchIndex) {
    const SwitchSpec &sw = switches[switchIndex];
    if (llvm::Error err = validateSwitch(sw, inputTypes))
      return err;
    llvm::SmallVector<std::string> operandTypes;
    for (const std::string &input : sw.inputs)
      operandTypes.push_back(inputTypes.lookup(input));
    printSwitch(os, sw, switchIndex, operandTypes);
  }
  for (std::size_t memIndex = 0; memIndex < mems.size(); ++memIndex) {
    const MemSpec &mem = mems[memIndex];
    if (!inputTypes.contains(mem.manager))
      return llvm::createStringError(std::errc::invalid_argument,
                                     "ADG mem manager input %s is unknown",
                                     mem.manager.c_str());
    for (const MemLoadPort &load : mem.loads) {
      if (!inputTypes.contains(load.address))
        return llvm::createStringError(std::errc::invalid_argument,
                                       "ADG mem load address input %s is "
                                       "unknown",
                                       load.address.c_str());
      if (!inputTypes.contains(load.control))
        return llvm::createStringError(std::errc::invalid_argument,
                                       "ADG mem load control input %s is "
                                       "unknown",
                                       load.control.c_str());
    }
    for (const MemStorePort &store : mem.stores) {
      if (!inputTypes.contains(store.address))
        return llvm::createStringError(std::errc::invalid_argument,
                                       "ADG mem store address input %s is "
                                       "unknown",
                                       store.address.c_str());
      if (!inputTypes.contains(store.data))
        return llvm::createStringError(std::errc::invalid_argument,
                                       "ADG mem store data input %s is "
                                       "unknown",
                                       store.data.c_str());
      if (!inputTypes.contains(store.control))
        return llvm::createStringError(std::errc::invalid_argument,
                                       "ADG mem store control input %s is "
                                       "unknown",
                                       store.control.c_str());
    }
    os << "  ";
    bool hasResult = false;
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
    os << " = fabric.mem [" << scheduleName(mem.schedule) << "] mgr("
       << valueName(mem.manager) << ')';
    if (!mem.loads.empty()) {
      os << " load(";
      for (std::size_t i = 0; i < mem.loads.size(); ++i) {
        if (i)
          os << ", ";
        const MemLoadPort &load = mem.loads[i];
        os << valueName(load.address) << ", " << valueName(load.control);
      }
      os << ')';
    }
    if (!mem.stores.empty()) {
      os << " store(";
      for (std::size_t i = 0; i < mem.stores.size(); ++i) {
        if (i)
          os << ", ";
        const MemStorePort &store = mem.stores[i];
        os << valueName(store.address) << ", " << valueName(store.data) << ", "
           << valueName(store.control);
      }
      os << ')';
    }
    os << "\n        [{load_group_size = "
       << static_cast<unsigned>(mem.loads.size())
       << " : i32, store_group_size = "
       << static_cast<unsigned>(mem.stores.size()) << " : i32";
    if (mem.schedule == Schedule::Temporal)
      os << ", tag_width = " << mem.temporalTagWidth
         << " : i32, addr_table_size = " << mem.temporalAddrTableSize
         << " : i32";
    os << "}]\n";

    llvm::SmallVector<std::string> operandTypes;
    operandTypes.push_back(inputTypes.lookup(mem.manager));
    for (const MemLoadPort &load : mem.loads) {
      operandTypes.push_back(inputTypes.lookup(load.address));
      operandTypes.push_back(inputTypes.lookup(load.control));
    }
    for (const MemStorePort &store : mem.stores) {
      operandTypes.push_back(inputTypes.lookup(store.address));
      operandTypes.push_back(inputTypes.lookup(store.data));
      operandTypes.push_back(inputTypes.lookup(store.control));
    }
    llvm::SmallVector<std::string> resultTypes;
    for (const MemLoadPort &load : mem.loads) {
      resultTypes.push_back(inputTypes.lookup(load.address));
      resultTypes.push_back(inputTypes.lookup(load.control));
    }
    for (const MemStorePort &store : mem.stores)
      resultTypes.push_back(inputTypes.lookup(store.control));
    os << "        : ";
    printTypeList(os, operandTypes);
    os << "\n        -> ";
    printResultTypes(os, resultTypes);
    os << '\n';
  }
  for (const std::string &line : exactBodyLines)
    os << "  " << line << '\n';
  os << "  fabric.yield\n";
  os << "}\n";
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

namespace {

std::vector<std::string> axiManagerPort(std::string port) {
  return {port + ".aw:output", port + ".w:output", port + ".b:input",
          port + ".ar:output", port + ".r:input"};
}

std::vector<std::string> axiSubordinatePort(std::string port) {
  return {port + ".aw:input", port + ".w:input", port + ".b:output",
          port + ".ar:input", port + ".r:output"};
}

void appendPorts(std::vector<std::string> &dst, std::vector<std::string> src) {
  dst.insert(dst.end(), std::make_move_iterator(src.begin()),
             std::make_move_iterator(src.end()));
}

void connectAxiMemoryPort(SystemBuilder &system, llvm::StringRef managerNode,
                          llvm::StringRef managerPort,
                          llvm::StringRef memoryNode,
                          llvm::StringRef memoryPort) {
  system.connect(managerNode.str(), managerPort.str(), "aw", memoryNode.str(),
                 memoryPort.str(), "aw");
  system.connect(managerNode.str(), managerPort.str(), "w", memoryNode.str(),
                 memoryPort.str(), "w");
  system.connect(memoryNode.str(), memoryPort.str(), "b", managerNode.str(),
                 managerPort.str(), "b");
  system.connect(managerNode.str(), managerPort.str(), "ar", memoryNode.str(),
                 memoryPort.str(), "ar");
  system.connect(memoryNode.str(), memoryPort.str(), "r", managerNode.str(),
                 managerPort.str(), "r");
}

ModuleBuilder makeTopologyMatrixModule(llvm::StringRef name,
                                       bool includeTemporal = false) {
  ModuleBuilder module(name.str());
  module.addInput("mgr", "memref<?x!fabric.bits<32>>")
      .addInput("a", "!fabric.bits<32>")
      .addInput("b", "!fabric.bits<32>")
      .addInput("c", "!fabric.bits<32>")
      .addInput("d", "!fabric.bits<32>")
      .addInput("addr", "!fabric.bits<32>")
      .addInput("ctrl", "!fabric.bits<0>");
  if (includeTemporal) {
    module.addInput("lhs_t", "!fabric.bits_tag<32, 4>")
        .addInput("rhs_t", "!fabric.bits_tag<32, 4>");
  }
  return module;
}

std::string valueList(llvm::ArrayRef<llvm::StringRef> names) {
  std::string text;
  for (llvm::StringRef name : names) {
    if (!text.empty())
      text += ", ";
    text += valueName(name);
  }
  return text;
}

std::string bits32TypeList(std::size_t count) {
  std::string text = "(";
  for (std::size_t index = 0; index < count; ++index) {
    if (index)
      text += ", ";
    text += "!fabric.bits<32>";
  }
  text += ")";
  return text;
}

std::string uniformTypeList(std::size_t count, llvm::StringRef type) {
  std::string text = "(";
  for (std::size_t index = 0; index < count; ++index) {
    if (index)
      text += ", ";
    text += type;
  }
  text += ")";
  return text;
}

std::string switchConnectivity(llvm::ArrayRef<llvm::StringRef> rows) {
  std::string text = "[{connectivity_table = [";
  for (std::size_t index = 0; index < rows.size(); ++index) {
    if (index)
      text += ", ";
    text += "\"";
    text += rows[index].str();
    text += "\"";
  }
  text += "]}]";
  return text;
}

void addSpatialMemLoad(ModuleBuilder &module) {
  module.addExactBodyLine("%data, %done =");
  module.addExactBodyLine(
      "    fabric.mem [spatial] mgr(%mgr) load(%addr, %ctrl)");
  module.addExactBodyLine(
      "      [{load_group_size = 1 : i32, store_group_size = 0 : i32}]");
  module.addExactBodyLine(
      "      : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, "
      "!fabric.bits<0>)");
  module.addExactBodyLine("      -> (!fabric.bits<32>, !fabric.bits<0>)");
}

void addSpatialSwitch(ModuleBuilder &module,
                      llvm::ArrayRef<llvm::StringRef> results,
                      llvm::ArrayRef<llvm::StringRef> inputs,
                      llvm::ArrayRef<llvm::StringRef> rows) {
  module.addExactBodyLine(valueList(results) + " =");
  module.addExactBodyLine("    fabric.switch [spatial] " + valueList(inputs));
  module.addExactBodyLine("      " + switchConnectivity(rows));
  module.addExactBodyLine("      : " + bits32TypeList(inputs.size()));
  module.addExactBodyLine("      -> " + (results.size() == 1
                                             ? std::string("!fabric.bits<32>")
                                             : bits32TypeList(results.size())));
}

void addSpatialAddPe(ModuleBuilder &module, llvm::StringRef result,
                     llvm::StringRef lhs, llvm::StringRef rhs,
                     llvm::StringRef opName = "arith.addi") {
  module.addExactBodyLine(valueName(result) + " =");
  module.addExactBodyLine("    fabric.pe [spatial] (%lhs = " + valueName(lhs) +
                          " : !fabric.bits<32>,");
  module.addExactBodyLine("                         %rhs = " + valueName(rhs) +
                          " : !fabric.bits<32>)");
  module.addExactBodyLine("        -> !fabric.bits<32> {");
  module.addExactBodyLine("      fabric.fu(%fu_lhs = %lhs : !fabric.bits<32>,");
  module.addExactBodyLine("                %fu_rhs = %rhs : !fabric.bits<32>) "
                          "-> !fabric.bits<32> {");
  module.addExactBodyLine("        %value = fabric.op [@" + opName.str() +
                          "] (%fu_lhs, %fu_rhs)");
  module.addExactBodyLine(
      "                 : (!fabric.bits<32>, !fabric.bits<32>) -> "
      "!fabric.bits<32>");
  module.addExactBodyLine("        fabric.yield %value : !fabric.bits<32>");
  module.addExactBodyLine("      }");
  module.addExactBodyLine("    }");
}

ModuleBuilder buildChain1DAdg() {
  ModuleBuilder module = makeTopologyMatrixModule("matrix_chain1d_adg");
  addSpatialMemLoad(module);
  addSpatialAddPe(module, "p0", "data", "a");
  addSpatialSwitch(module, {"s0"}, {"p0", "b"}, {"11"});
  addSpatialAddPe(module, "p1", "s0", "c");
  addSpatialAddPe(module, "p2", "p1", "d");
  return module;
}

ModuleBuilder buildMesh2DAdg() {
  ModuleBuilder module = makeTopologyMatrixModule("matrix_mesh2d_adg");
  addSpatialMemLoad(module);
  addSpatialAddPe(module, "n00", "data", "a");
  addSpatialAddPe(module, "n01", "data", "b");
  addSpatialSwitch(module, {"east", "south"}, {"n00", "n01"}, {"11", "11"});
  addSpatialAddPe(module, "n10", "east", "c");
  addSpatialAddPe(module, "n11", "south", "n10");
  return module;
}

ModuleBuilder buildTorusEdgeAdg() {
  ModuleBuilder module = makeTopologyMatrixModule("matrix_torus_edge_adg");
  addSpatialMemLoad(module);
  addSpatialAddPe(module, "n00", "data", "a");
  addSpatialAddPe(module, "n01", "data", "b");
  addSpatialSwitch(module, {"east", "south"}, {"n00", "n01", "c"},
                   {"110", "101"});
  addSpatialAddPe(module, "n10", "east", "south");
  addSpatialSwitch(module, {"wrap_north", "wrap_west"}, {"n10", "n00", "d"},
                   {"110", "101"});
  addSpatialAddPe(module, "n11", "wrap_north", "wrap_west");
  return module;
}

ModuleBuilder buildSystolicArrayAdg() {
  ModuleBuilder module = makeTopologyMatrixModule("matrix_systolic_array_adg");
  addSpatialMemLoad(module);
  addSpatialSwitch(module, {"broadcast"}, {"data", "a", "b"}, {"111"});
  addSpatialAddPe(module, "cell0", "broadcast", "c", "arith.mulf");
  addSpatialAddPe(module, "cell1", "cell0", "d", "arith.addf");
  addSpatialAddPe(module, "cell2", "cell1", "broadcast", "arith.addf");
  return module;
}

ModuleBuilder buildClusteredArrayAdg() {
  ModuleBuilder module = makeTopologyMatrixModule("matrix_clustered_array_adg");
  addSpatialMemLoad(module);
  addSpatialAddPe(module, "c0a", "data", "a");
  addSpatialAddPe(module, "c0b", "data", "b");
  addSpatialSwitch(module, {"cluster0"}, {"c0a", "c0b"}, {"11"});
  addSpatialAddPe(module, "c1a", "c", "d");
  addSpatialAddPe(module, "c1b", "cluster0", "c1a");
  addSpatialSwitch(module, {"cluster1"}, {"c1a", "c1b"}, {"11"});
  addSpatialAddPe(module, "out", "cluster0", "cluster1");
  return module;
}

ModuleBuilder buildReductionTreeAdg() {
  ModuleBuilder module = makeTopologyMatrixModule("matrix_reduction_tree_adg");
  addSpatialMemLoad(module);
  addSpatialAddPe(module, "leaf0", "data", "a");
  addSpatialAddPe(module, "leaf1", "b", "c");
  addSpatialSwitch(module, {"tree0", "tree1"}, {"leaf0", "leaf1"},
                   {"10", "01"});
  addSpatialAddPe(module, "root", "tree0", "tree1");
  return module;
}

ModuleBuilder buildCrossCoupledSwitchAdg() {
  ModuleBuilder module =
      makeTopologyMatrixModule("matrix_cross_coupled_switch_adg");
  addSpatialMemLoad(module);
  addSpatialAddPe(module, "left", "data", "a");
  addSpatialAddPe(module, "right", "b", "c");
  addSpatialSwitch(module, {"x0", "x1"}, {"left", "right"}, {"01", "10"});
  addSpatialSwitch(module, {"x2", "x3"}, {"x0", "x1", "d"}, {"111", "111"});
  addSpatialAddPe(module, "merged", "x2", "x3");
  return module;
}

ModuleBuilder buildSparseLongLinkAdg() {
  ModuleBuilder module =
      makeTopologyMatrixModule("matrix_sparse_long_link_adg");
  addSpatialMemLoad(module);
  addSpatialAddPe(module, "near0", "data", "a");
  addSpatialAddPe(module, "near1", "b", "c");
  addSpatialAddPe(module, "far0", "near1", "d");
  addSpatialSwitch(module, {"long0", "bypass"}, {"near0", "far0", "data"},
                   {"101", "010"});
  addSpatialAddPe(module, "far1", "long0", "bypass");
  return module;
}

ModuleBuilder buildHeterogeneousIslandsAdg() {
  ModuleBuilder module =
      makeTopologyMatrixModule("matrix_heterogeneous_islands_adg", true);
  TemporalPeConfig temporal;
  temporal.tagWidth = 4;
  temporal.numInstruction = 2;
  temporal.fuConfigMode = "per_fu_config";
  temporal.operandBufferMode = "per_input_port";
  temporal.operandBufferSize = 2;
  module.addPe(makeMinimalAddPe(Schedule::Temporal, "lhs_t", "rhs_t",
                                "!fabric.bits_tag<32, 4>", "!fabric.bits<32>",
                                std::move(temporal)));
  addSpatialMemLoad(module);
  addSpatialAddPe(module, "int_island", "data", "a", "arith.addi");
  addSpatialAddPe(module, "float_island", "b", "c", "arith.mulf");
  addSpatialSwitch(module, {"island_mux"}, {"int_island", "float_island", "d"},
                   {"111"});
  addSpatialAddPe(module, "bridge", "island_mux", "int_island");
  return module;
}

} // namespace

ModuleBuilder loom::adg::buildMinimalSpatialAdg() {
  ModuleBuilder module("minimal_spatial_adg");
  module.addInput("mgr", "memref<?x!fabric.bits<32>>")
      .addInput("lhs", "!fabric.bits<32>")
      .addInput("rhs", "!fabric.bits<32>")
      .addInput("addr", "!fabric.bits<32>")
      .addInput("ctrl", "!fabric.bits<0>");

  module.addPe(makeMinimalAddPe(Schedule::Spatial, "!fabric.bits<32>",
                                "!fabric.bits<32>"));

  module.addSwitch(SwitchSpec{Schedule::Spatial,
                              {"lhs", "rhs"},
                              {"!fabric.bits<32>", "!fabric.bits<32>"},
                              {"11", "11"},
                              0});
  module.addMem(MemSpec{Schedule::Spatial, "mgr", {{"addr", "ctrl"}}, {}});
  return module;
}

ModuleBuilder loom::adg::buildMinimalTemporalAdg() {
  ModuleBuilder module("minimal_temporal_adg");
  module.addInput("mgr", "memref<?x!fabric.bits<32>>")
      .addInput("lhs", "!fabric.bits_tag<32, 4>")
      .addInput("rhs", "!fabric.bits_tag<32, 4>")
      .addInput("addr", "!fabric.bits_tag<32, 4>")
      .addInput("ctrl", "!fabric.bits_tag<0, 4>");

  TemporalPeConfig temporal;
  temporal.tagWidth = 4;
  temporal.numInstruction = 1;
  temporal.fuConfigMode = "per_fu_config";
  temporal.operandBufferMode = "per_instruction";
  module.addPe(makeMinimalAddPe(Schedule::Temporal, "!fabric.bits_tag<32, 4>",
                                "!fabric.bits<32>", std::move(temporal)));

  module.addSwitch(
      SwitchSpec{Schedule::Temporal,
                 {"lhs", "rhs"},
                 {"!fabric.bits_tag<32, 4>", "!fabric.bits_tag<32, 4>"},
                 {"11", "11"},
                 1});

  MemSpec mem;
  mem.schedule = Schedule::Temporal;
  mem.manager = "mgr";
  mem.loads = {{"addr", "ctrl"}};
  mem.temporalTagWidth = 4;
  mem.temporalAddrTableSize = 1;
  module.addMem(std::move(mem));
  return module;
}

ModuleBuilder loom::adg::buildSharedReductionAdg() {
  ModuleBuilder module("shared_reduction_adg");
  module.addInput("mgr", "memref<?x!fabric.bits<32>>")
      .addInput("i64a", "!fabric.bits<64>")
      .addInput("i64b", "!fabric.bits<64>")
      .addInput("i64c", "!fabric.bits<64>")
      .addInput("i32a", "!fabric.bits<32>")
      .addInput("i32b", "!fabric.bits<32>")
      .addInput("i32c", "!fabric.bits<32>")
      .addInput("i32d", "!fabric.bits<32>")
      .addInput("ctrl", "!fabric.bits<0>");

  PeSpec streamPe;
  streamPe.inputs = {{"pa", "i64a", "!fabric.bits<64>", "!fabric.bits<32>"},
                     {"pb", "i64b", "!fabric.bits<64>", "!fabric.bits<32>"},
                     {"pc", "i64c", "!fabric.bits<64>", "!fabric.bits<32>"},
                     {"pd", "stream_sum_lhs", "!fabric.bits<32>", ""},
                     {"pe", "stream_sum_rhs", "!fabric.bits<32>", ""},
                     {"pi", "scan_init", "!fabric.bits<32>", ""},
                     {"pn", "scan_feedback", "!fabric.bits<32>", ""},
                     {"ps", "scan_scale", "!fabric.bits<32>", ""}};
  streamPe.resultNames = {"idx", "running", "carried_scan", "reduction_scale",
                          "fp_gate"};
  streamPe.resultTypes = {"!fabric.bits<32>", "!fabric.bits<32>",
                          "!fabric.bits<32>", "!fabric.bits<32>",
                          "!fabric.bits<32>"};
  FuSpec streamFu;
  streamFu.inputs = {{"fa", "pa", "!fabric.bits<32>", ""},
                     {"fb", "pb", "!fabric.bits<32>", ""},
                     {"fc", "pc", "!fabric.bits<32>", ""},
                     {"sum_lhs", "pd", "!fabric.bits<32>", ""},
                     {"sum_rhs", "pe", "!fabric.bits<32>", ""},
                     {"init", "pi", "!fabric.bits<32>", ""},
                     {"next", "pn", "!fabric.bits<32>", ""},
                     {"scale", "ps", "!fabric.bits<32>", ""}};
  streamFu.resultTypes = {"!fabric.bits<32>", "!fabric.bits<32>",
                          "!fabric.bits<32>", "!fabric.bits<32>",
                          "!fabric.bits<32>"};
  streamFu.operations.push_back(
      FabricOpSpec{{"idx", "rwc"},
                   {"dataflow.stream"},
                   {"fa", "fb", "fc"},
                   {"!fabric.bits<32>", "!fabric.bits<32>", "!fabric.bits<32>"},
                   {"!fabric.bits<32>", "!fabric.bits<1>"},
                   {{"cont_cond", {"<", ">"}}, {"step_op", {"+="}}},
                   {{"cont_cond", "<"}, {"step_op", "+="}}});
  streamFu.operations.push_back(
      FabricOpSpec{{"carried"},
                   {"dataflow.carry"},
                   {"rwc", "init", "next"},
                   {"!fabric.bits<1>", "!fabric.bits<32>", "!fabric.bits<32>"},
                   {"!fabric.bits<32>"},
                   {},
                   {}});
  streamFu.operations.push_back(
      FabricOpSpec{{"sum"},
                   {"arith.addi"},
                   {"sum_lhs", "sum_rhs"},
                   {"!fabric.bits<32>", "!fabric.bits<32>"},
                   {"!fabric.bits<32>"},
                   {},
                   {}});
  streamFu.operations.push_back(
      FabricOpSpec{{"stable_scale"},
                   {"dataflow.invariant"},
                   {"rwc", "scale"},
                   {"!fabric.bits<1>", "!fabric.bits<32>"},
                   {"!fabric.bits<32>"},
                   {},
                   {}});
  streamFu.yieldValues = {"idx", "sum", "carried", "stable_scale", "rwc"};
  streamFu.yieldTypes = {"!fabric.bits<32>", "!fabric.bits<32>",
                         "!fabric.bits<32>", "!fabric.bits<32>",
                         "!fabric.bits<1>"};
  streamPe.fus.push_back(std::move(streamFu));
  module.addPe(std::move(streamPe));

  PeSpec auxStreamPe;
  auxStreamPe.inputs = {{"pa", "aux_stream_lb", "!fabric.bits<32>", ""},
                        {"pb", "aux_stream_ub", "!fabric.bits<32>", ""},
                        {"pc", "aux_stream_step", "!fabric.bits<32>", ""}};
  auxStreamPe.resultNames = {"aux_idx", "aux_rwc"};
  auxStreamPe.resultTypes = {"!fabric.bits<32>", "!fabric.bits<32>"};
  FuSpec auxStreamFu;
  auxStreamFu.inputs = {{"fa", "pa", "!fabric.bits<32>", ""},
                        {"fb", "pb", "!fabric.bits<32>", ""},
                        {"fc", "pc", "!fabric.bits<32>", ""}};
  auxStreamFu.resultTypes = {"!fabric.bits<32>", "!fabric.bits<32>"};
  auxStreamFu.operations.push_back(
      FabricOpSpec{{"aux_op_idx", "aux_op_rwc"},
                   {"dataflow.stream"},
                   {"fa", "fb", "fc"},
                   {"!fabric.bits<32>", "!fabric.bits<32>", "!fabric.bits<32>"},
                   {"!fabric.bits<32>", "!fabric.bits<1>"},
                   {{"cont_cond", {"<", ">"}}, {"step_op", {"+="}}},
                   {{"cont_cond", "<"}, {"step_op", "+="}}});
  auxStreamFu.yieldValues = {"aux_op_idx", "aux_op_rwc"};
  auxStreamFu.yieldTypes = {"!fabric.bits<32>", "!fabric.bits<1>"};
  auxStreamPe.fus.push_back(std::move(auxStreamFu));
  module.addPe(std::move(auxStreamPe));

  PeSpec auxGatePe;
  auxGatePe.inputs = {{"pa", "gate_cond", "!fabric.bits<32>", ""},
                      {"pb", "gate_value", "!fabric.bits<32>", ""}};
  auxGatePe.resultNames = {"aux_gate_cond", "aux_active_idx"};
  auxGatePe.resultTypes = {"!fabric.bits<32>", "!fabric.bits<32>"};
  auto makeGateFu = []() {
    FuSpec fu;
    fu.inputs = {{"cond", "pa", "!fabric.bits<32>", "!fabric.bits<1>"},
                 {"value", "pb", "!fabric.bits<32>", ""}};
    fu.resultTypes = {"!fabric.bits<32>", "!fabric.bits<32>"};
    fu.operations.push_back(
        FabricOpSpec{{"after_cond", "after_value"},
                     {"dataflow.gate"},
                     {"cond", "value"},
                     {"!fabric.bits<1>", "!fabric.bits<32>"},
                     {"!fabric.bits<1>", "!fabric.bits<32>"},
                     {},
                     {{"value_kind", "data"}}});
    fu.yieldValues = {"after_cond", "after_value"};
    fu.yieldTypes = {"!fabric.bits<1>", "!fabric.bits<32>"};
    return fu;
  };
  auxGatePe.fus.push_back(makeGateFu());
  auxGatePe.fus.push_back(makeGateFu());
  auxGatePe.fus.push_back(makeGateFu());
  auxGatePe.fus.push_back(makeGateFu());
  module.addPe(std::move(auxGatePe));

  PeSpec absPe;
  absPe.inputs = {{"pa", "data0", "!fabric.bits<32>", ""}};
  absPe.resultNames = {"abs_data"};
  absPe.resultTypes = {"!fabric.bits<32>"};
  absPe.fus.push_back(FuSpec{{{"value", "pa", "!fabric.bits<32>", ""}},
                             {"!fabric.bits<32>"},
                             {FabricOpSpec{{"abs"},
                                           {"llvm.intr.abs"},
                                           {"value"},
                                           {"!fabric.bits<32>"},
                                           {"!fabric.bits<32>"},
                                           {},
                                           {}}},
                             {"abs"}});
  absPe.fus.push_back(FuSpec{{{"value", "pa", "!fabric.bits<32>", ""}},
                             {"!fabric.bits<32>"},
                             {FabricOpSpec{{"abs"},
                                           {"llvm.intr.fabs"},
                                           {"value"},
                                           {"!fabric.bits<32>"},
                                           {"!fabric.bits<32>"},
                                           {},
                                           {}}},
                             {"abs"}});
  module.addPe(std::move(absPe));

  PeSpec squaredPe;
  squaredPe.inputs = {{"pa", "mul_lhs_input", "!fabric.bits<32>", ""},
                      {"pb", "mul_rhs_input", "!fabric.bits<32>", ""}};
  squaredPe.resultNames = {"squared_data"};
  squaredPe.resultTypes = {"!fabric.bits<32>"};
  squaredPe.fus.push_back(
      FuSpec{{{"lhs", "pa", "!fabric.bits<32>", ""},
              {"rhs", "pb", "!fabric.bits<32>", ""}},
             {"!fabric.bits<32>"},
             {FabricOpSpec{{"product"},
                           {"arith.muli"},
                           {"lhs", "rhs"},
                           {"!fabric.bits<32>", "!fabric.bits<32>"},
                           {"!fabric.bits<32>"},
                           {},
                           {}}},
             {"product"}});
  module.addPe(std::move(squaredPe));

  auto addFpBinaryPe = [&](std::string resultName, std::string lhsInput,
                           std::string rhsInput, llvm::StringRef valueName,
                           llvm::StringRef opName) {
    PeSpec pe;
    pe.inputs = {{"pa", std::move(lhsInput), "!fabric.bits<32>", ""},
                 {"pb", std::move(rhsInput), "!fabric.bits<32>", ""}};
    pe.resultNames = {std::move(resultName)};
    pe.resultTypes = {"!fabric.bits<32>"};
    pe.fus.push_back(
        FuSpec{{{"lhs", "pa", "!fabric.bits<32>", ""},
                {"rhs", "pb", "!fabric.bits<32>", ""}},
               {"!fabric.bits<32>"},
               {FabricOpSpec{{valueName.str()},
                             {opName.str()},
                             {"lhs", "rhs"},
                             {"!fabric.bits<32>", "!fabric.bits<32>"},
                             {"!fabric.bits<32>"},
                             {},
                             {}}},
               {valueName.str()}});
    module.addPe(std::move(pe));
  };

  addFpBinaryPe("fp_running", "fp_lhs", "fp_rhs", "sum", "arith.addf");
  addFpBinaryPe("fp_running_aux", "fp_lhs_aux", "fp_rhs_aux", "sum",
                "arith.addf");

  PeSpec fpInvariantPe;
  fpInvariantPe.inputs = {{"pa", "fp_gate", "!fabric.bits<32>", ""},
                          {"pb", "fp_invariant_value", "!fabric.bits<32>", ""}};
  fpInvariantPe.resultNames = {"fp_invariant"};
  fpInvariantPe.resultTypes = {"!fabric.bits<32>"};
  fpInvariantPe.fus.push_back(
      FuSpec{{{"cond", "pa", "!fabric.bits<32>", "!fabric.bits<1>"},
              {"value", "pb", "!fabric.bits<32>", ""}},
             {"!fabric.bits<32>"},
             {FabricOpSpec{{"stable"},
                           {"dataflow.invariant"},
                           {"cond", "value"},
                           {"!fabric.bits<1>", "!fabric.bits<32>"},
                           {"!fabric.bits<32>"},
                           {},
                           {}}},
             {"stable"}});
  module.addPe(std::move(fpInvariantPe));

  auto addInvariantPe = [&](std::string resultName, std::string valueInput,
                            std::string condInput = "fp_gate") {
    PeSpec pe;
    pe.inputs = {{"pa", std::move(condInput), "!fabric.bits<32>", ""},
                 {"pb", std::move(valueInput), "!fabric.bits<32>", ""}};
    pe.resultNames = {std::move(resultName)};
    pe.resultTypes = {"!fabric.bits<32>"};
    pe.fus.push_back(
        FuSpec{{{"cond", "pa", "!fabric.bits<32>", "!fabric.bits<1>"},
                {"value", "pb", "!fabric.bits<32>", ""}},
               {"!fabric.bits<32>"},
               {FabricOpSpec{{"stable"},
                             {"dataflow.invariant"},
                             {"cond", "value"},
                             {"!fabric.bits<1>", "!fabric.bits<32>"},
                             {"!fabric.bits<32>"},
                             {},
                             {}}},
               {"stable"}});
    module.addPe(std::move(pe));
  };
  addInvariantPe("bit_invariant", "i32d");
  addInvariantPe("bit_invariant_aux0", "i32c");
  addInvariantPe("aux_invariant2", "bit_invariant_aux1_value",
                 "aux_invariant_cond");
  addInvariantPe("bit_invariant_aux1", "bit_invariant_aux1_value");
  auto addAuxInvariantPe = [&](std::string resultName, std::string valueInput) {
    PeSpec pe;
    pe.inputs = {{"pa", "aux_invariant_cond", "!fabric.bits<32>", ""},
                 {"pb", std::move(valueInput), "!fabric.bits<32>", ""}};
    pe.resultNames = {std::move(resultName)};
    pe.resultTypes = {"!fabric.bits<32>"};
    pe.fus.push_back(
        FuSpec{{{"cond", "pa", "!fabric.bits<32>", "!fabric.bits<1>"},
                {"value", "pb", "!fabric.bits<32>", ""}},
               {"!fabric.bits<32>"},
               {FabricOpSpec{{"stable"},
                             {"dataflow.invariant"},
                             {"cond", "value"},
                             {"!fabric.bits<1>", "!fabric.bits<32>"},
                             {"!fabric.bits<32>"},
                             {},
                             {}}},
               {"stable"}});
    module.addPe(std::move(pe));
  };
  addAuxInvariantPe("aux_invariant0", "aux_invariant0_value");
  addAuxInvariantPe("aux_invariant1", "aux_invariant1_value");

  addFpBinaryPe("fp_diff", "fp_diff_lhs", "fp_diff_rhs", "diff", "arith.subf");
  addFpBinaryPe("fp_diff_aux", "fp_diff_aux_lhs", "fp_diff_aux_rhs", "diff",
                "arith.subf");

  PeSpec scaledReductionPe;
  scaledReductionPe.inputs = {
      {"pa", "scaled_reduction_lhs", "!fabric.bits<32>", ""},
      {"pb", "scaled_reduction_rhs", "!fabric.bits<32>", ""}};
  scaledReductionPe.resultNames = {"scaled_reduction"};
  scaledReductionPe.resultTypes = {"!fabric.bits<32>"};
  scaledReductionPe.fus.push_back(
      FuSpec{{{"lhs", "pa", "!fabric.bits<32>", ""},
              {"rhs", "pb", "!fabric.bits<32>", ""}},
             {"!fabric.bits<32>"},
             {FabricOpSpec{{"product"},
                           {"arith.mulf"},
                           {"lhs", "rhs"},
                           {"!fabric.bits<32>", "!fabric.bits<32>"},
                           {"!fabric.bits<32>"},
                           {},
                           {}}},
             {"product"}});
  module.addPe(std::move(scaledReductionPe));

  auto makeCarryFu = []() {
    return FuSpec{{{"cond", "pa", "!fabric.bits<32>", "!fabric.bits<1>"},
                   {"init", "pb", "!fabric.bits<32>", ""},
                   {"next", "pc", "!fabric.bits<32>", ""}},
                  {"!fabric.bits<32>"},
                  {FabricOpSpec{{"carried"},
                                {"dataflow.carry"},
                                {"cond", "init", "next"},
                                {"!fabric.bits<1>", "!fabric.bits<32>",
                                 "!fabric.bits<32>"},
                                {"!fabric.bits<32>"},
                                {},
                                {}}},
                  {"carried"}};
  };
  PeSpec carryPe;
  carryPe.inputs = {{"pa", "bit_carry_cond", "!fabric.bits<32>", ""},
                    {"pb", "bit_carry_init", "!fabric.bits<32>", ""},
                    {"pc", "bit_carry_next", "!fabric.bits<32>", ""}};
  carryPe.resultNames = {"bit_carry"};
  carryPe.resultTypes = {"!fabric.bits<32>"};
  carryPe.fus.push_back(makeCarryFu());
  carryPe.fus.push_back(makeCarryFu());
  module.addPe(std::move(carryPe));

  auto makeBinary32Fu = [](std::string resultName,
                           std::vector<std::string> opList) {
    std::string yieldName = resultName;
    return FuSpec{{{"lhs", "pa", "!fabric.bits<32>", ""},
                   {"rhs", "pb", "!fabric.bits<32>", ""}},
                  {"!fabric.bits<32>"},
                  {FabricOpSpec{{std::move(resultName)},
                                std::move(opList),
                                {"lhs", "rhs"},
                                {"!fabric.bits<32>", "!fabric.bits<32>"},
                                {"!fabric.bits<32>"},
                                {},
                                {}}},
                  {std::move(yieldName)}};
  };
  auto addBinary32Pe = [&](std::string peResultName, std::string lhsInput,
                           std::string rhsInput, std::string opResultName,
                           std::vector<std::string> opList) {
    PeSpec pe;
    pe.inputs = {{"pa", std::move(lhsInput), "!fabric.bits<32>", ""},
                 {"pb", std::move(rhsInput), "!fabric.bits<32>", ""}};
    pe.resultNames = {std::move(peResultName)};
    pe.resultTypes = {"!fabric.bits<32>"};
    pe.fus.push_back(
        makeBinary32Fu(std::move(opResultName), std::move(opList)));
    module.addPe(std::move(pe));
  };
  addBinary32Pe("int_sum", "int_add_lhs", "int_add_rhs", "sum",
                {"arith.addi", "arith.subi"});
  addBinary32Pe("int_product", "int_mul_lhs", "int_mul_rhs", "product",
                {"arith.muli"});
  addBinary32Pe("int_product_aux", "int_mul_aux_lhs", "int_mul_aux_rhs",
                "product", {"arith.muli"});
  addBinary32Pe("int_div0", "int_div0_lhs", "int_div0_rhs", "quotient",
                {"arith.divsi"});
  addBinary32Pe("int_div1", "int_div1_lhs", "int_div1_rhs", "quotient",
                {"arith.divsi"});
  addBinary32Pe("int_rem", "int_rem_lhs", "int_rem_rhs", "remainder",
                {"arith.remsi"});
  addBinary32Pe("uint_rem", "uint_rem_lhs", "uint_rem_rhs", "remainder",
                {"arith.divui", "arith.remui"});
  addBinary32Pe("fp_div", "fp_div_lhs", "fp_div_rhs", "quotient",
                {"arith.divf", "arith.remf"});
  auto addConfigurableConstPe = [&](std::string resultName) {
    PeSpec constPe;
    constPe.inputs = {{"pa", "ctrl", "!fabric.bits<0>", "!fabric.bits<32>"}};
    constPe.resultNames = {std::move(resultName)};
    constPe.resultTypes = {"!fabric.bits<32>"};
    constPe.fus.push_back(FuSpec{
        {{"ctrl_in", "pa", "!fabric.bits<32>", "!fabric.bits<0>"}},
        {"!fabric.bits<32>"},
        {FabricOpSpec{
            {"value"},
            {"dataflow.constant"},
            {"ctrl_in"},
            {"!fabric.bits<0>"},
            {"!fabric.bits<32>"},
            {{"const_hex_value", {"0x00000000", "0x00000001", "0x00000002"}}},
            {}}},
        {"value"}});
    module.addPe(std::move(constPe));
  };
  addConfigurableConstPe("addr_shift_const");
  addConfigurableConstPe("addr_aux_const");
  addConfigurableConstPe("addr_bias_const");

  PeSpec addrUnscalePe;
  addrUnscalePe.inputs = {{"pa", "addr_unscale_lhs", "!fabric.bits<32>", ""},
                          {"pb", "addr_unscale_rhs", "!fabric.bits<32>", ""}};
  addrUnscalePe.resultNames = {"addr_unscaled"};
  addrUnscalePe.resultTypes = {"!fabric.bits<32>"};
  addrUnscalePe.fus.push_back(
      FuSpec{{{"lhs", "pa", "!fabric.bits<32>", ""},
              {"rhs", "pb", "!fabric.bits<32>", ""}},
             {"!fabric.bits<32>"},
             {FabricOpSpec{{"shifted"},
                           {"arith.shrui"},
                           {"lhs", "rhs"},
                           {"!fabric.bits<32>", "!fabric.bits<32>"},
                           {"!fabric.bits<32>"},
                           {},
                           {}}},
             {"shifted"}});
  module.addPe(std::move(addrUnscalePe));
  PeSpec addrShiftPe;
  addrShiftPe.inputs = {{"pa", "addr_shift_lhs", "!fabric.bits<32>", ""},
                        {"pb", "addr_shift_rhs", "!fabric.bits<32>", ""}};
  addrShiftPe.resultNames = {"addr_shifted"};
  addrShiftPe.resultTypes = {"!fabric.bits<32>"};
  addrShiftPe.fus.push_back(
      FuSpec{{{"lhs", "pa", "!fabric.bits<32>", ""},
              {"rhs", "pb", "!fabric.bits<32>", ""}},
             {"!fabric.bits<32>"},
             {FabricOpSpec{{"shifted"},
                           {"arith.shli"},
                           {"lhs", "rhs"},
                           {"!fabric.bits<32>", "!fabric.bits<32>"},
                           {"!fabric.bits<32>"},
                           {},
                           {}}},
             {"shifted"}});
  module.addPe(std::move(addrShiftPe));
  PeSpec logicMaskPe;
  logicMaskPe.inputs = {{"pa", "logic_mask_lhs", "!fabric.bits<32>", ""},
                        {"pb", "logic_mask_rhs", "!fabric.bits<32>", ""}};
  logicMaskPe.resultNames = {"logic_masked"};
  logicMaskPe.resultTypes = {"!fabric.bits<32>"};
  logicMaskPe.fus.push_back(
      FuSpec{{{"lhs", "pa", "!fabric.bits<32>", ""},
              {"rhs", "pb", "!fabric.bits<32>", ""}},
             {"!fabric.bits<32>"},
             {FabricOpSpec{{"masked"},
                           {"arith.andi"},
                           {"lhs", "rhs"},
                           {"!fabric.bits<32>", "!fabric.bits<32>"},
                           {"!fabric.bits<32>"},
                           {},
                           {}}},
             {"masked"}});
  module.addPe(std::move(logicMaskPe));
  addBinary32Pe("int_or", "int_or_lhs", "int_or_rhs", "combined",
                {"arith.ori"});
  addBinary32Pe("int_xor", "int_xor_lhs", "int_xor_rhs", "combined",
                {"arith.xori"});
  PeSpec packedSatPe;
  packedSatPe.inputs = {{"pa", "packed_sat_lhs", "!fabric.bits<32>", ""},
                        {"pb", "packed_sat_rhs", "!fabric.bits<32>", ""}};
  packedSatPe.resultNames = {"packed_sat"};
  packedSatPe.resultTypes = {"!fabric.bits<32>"};
  packedSatPe.fus.push_back(FuSpec{
      {{"lhs", "pa", "!fabric.bits<32>", ""},
       {"rhs", "pb", "!fabric.bits<32>", ""}},
      {"!fabric.bits<32>"},
      {FabricOpSpec{{"packed"},
                    {"llvm.arm.qadd16", "llvm.arm.qsub16", "llvm.arm.qsub8"},
                    {"lhs", "rhs"},
                    {"!fabric.bits<32>", "!fabric.bits<32>"},
                    {"!fabric.bits<32>"},
                    {},
                    {}}},
      {"packed"}});
  module.addPe(std::move(packedSatPe));

  auto addFmulAddPe = [&](std::string resultName, std::string lhsInput,
                          std::string rhsInput, std::string accInput) {
    PeSpec pe;
    pe.inputs = {{"pa", std::move(lhsInput), "!fabric.bits<32>", ""},
                 {"pb", std::move(rhsInput), "!fabric.bits<32>", ""},
                 {"pc", std::move(accInput), "!fabric.bits<32>", ""}};
    pe.resultNames = {std::move(resultName)};
    pe.resultTypes = {"!fabric.bits<32>"};
    pe.fus.push_back(
        FuSpec{{{"lhs", "pa", "!fabric.bits<32>", ""},
                {"rhs", "pb", "!fabric.bits<32>", ""},
                {"acc", "pc", "!fabric.bits<32>", ""}},
               {"!fabric.bits<32>"},
               {FabricOpSpec{
                   {"mac"},
                   {"llvm.intr.fmuladd"},
                   {"lhs", "rhs", "acc"},
                   {"!fabric.bits<32>", "!fabric.bits<32>", "!fabric.bits<32>"},
                   {"!fabric.bits<32>"},
                   {},
                   {}}},
               {"mac"}});
    module.addPe(std::move(pe));
  };
  addFmulAddPe("mac_result", "mac_lhs", "mac_rhs", "mac_acc");
  addFmulAddPe("mac_result1", "mac1_lhs", "mac1_rhs", "mac1_acc");

  auto makeUnary32YieldFu = [](std::string resultName, std::string opName) {
    std::string yieldName = resultName;
    return FuSpec{{{"value", "pa", "!fabric.bits<32>", ""}},
                  {"!fabric.bits<32>"},
                  {FabricOpSpec{{std::move(resultName)},
                                {std::move(opName)},
                                {"value"},
                                {"!fabric.bits<32>"},
                                {"!fabric.bits<32>"},
                                {},
                                {}}},
                  {std::move(yieldName)}};
  };
  auto addUnary32YieldPe = [&](std::string resultName, std::string opName,
                               std::string inputName = "i32a") {
    std::string peResultName = resultName;
    PeSpec pe;
    pe.inputs = {{"pa", std::move(inputName), "!fabric.bits<32>", ""}};
    pe.resultNames = {std::move(peResultName)};
    pe.resultTypes = {"!fabric.bits<32>"};
    pe.fus.push_back(
        makeUnary32YieldFu(std::move(resultName), std::move(opName)));
    module.addPe(std::move(pe));
  };

  PeSpec fshlPe;
  fshlPe.inputs = {{"pa", "rotate_lhs", "!fabric.bits<32>", ""},
                   {"pb", "rotate_rhs", "!fabric.bits<32>", ""},
                   {"pc", "rotate_amount", "!fabric.bits<32>", ""}};
  fshlPe.resultNames = {"rotated"};
  fshlPe.resultTypes = {"!fabric.bits<32>"};
  auto makeFshlFu = []() {
    return FuSpec{{{"lhs", "pa", "!fabric.bits<32>", ""},
                   {"rhs", "pb", "!fabric.bits<32>", ""},
                   {"amount", "pc", "!fabric.bits<32>", ""}},
                  {"!fabric.bits<32>"},
                  {FabricOpSpec{{"rotated_value"},
                                {"llvm.intr.fshl"},
                                {"lhs", "rhs", "amount"},
                                {"!fabric.bits<32>", "!fabric.bits<32>",
                                 "!fabric.bits<32>"},
                                {"!fabric.bits<32>"},
                                {},
                                {}}},
                  {"rotated_value"}};
  };
  fshlPe.fus.push_back(makeFshlFu());
  fshlPe.fus.push_back(makeFshlFu());
  module.addPe(std::move(fshlPe));

  addUnary32YieldPe("abs", "llvm.intr.abs");
  addUnary32YieldPe("swapped", "llvm.intr.bswap");

  auto addCastBankPe = [&]() {
    constexpr unsigned kCastLanes = 4;
    const char *ports[] = {"pa", "pb", "pc", "pd"};
    PeSpec pe;
    FuSpec fu;
    for (unsigned i = 0; i < kCastLanes; ++i) {
      std::string index = std::to_string(i);
      std::string value = "value" + index;
      std::string converted = "converted" + index;
      pe.inputs.push_back(
          {ports[i], "cast" + index + "_input", "!fabric.bits<32>", ""});
      pe.resultNames.push_back("cast" + index + "_result");
      pe.resultTypes.push_back("!fabric.bits<32>");
      fu.inputs.push_back({value, ports[i], "!fabric.bits<32>", ""});
      fu.resultTypes.push_back("!fabric.bits<32>");
      fu.operations.push_back(
          FabricOpSpec{{converted},
                       {"llvm.trunc", "llvm.sext", "llvm.zext"},
                       {value},
                       {"!fabric.bits<32>"},
                       {"!fabric.bits<32>"},
                       {},
                       {}});
      fu.yieldValues.push_back(std::move(converted));
    }
    pe.fus.push_back(std::move(fu));
    module.addPe(std::move(pe));
  };
  addCastBankPe();

  auto addWideExtensionPe = [&](std::string resultName, std::string inputName) {
    PeSpec pe;
    pe.inputs = {
        {"pa", std::move(inputName), "!fabric.bits<32>", "!fabric.bits<64>"}};
    pe.resultNames = {std::move(resultName)};
    pe.resultTypes = {"!fabric.bits<64>"};
    pe.fus.push_back(
        FuSpec{{{"value", "pa", "!fabric.bits<64>", "!fabric.bits<32>"}},
               {"!fabric.bits<64>"},
               {FabricOpSpec{{"wide"},
                             {"llvm.sext", "llvm.zext"},
                             {"value"},
                             {"!fabric.bits<32>"},
                             {"!fabric.bits<64>"},
                             {},
                             {}}},
               {"wide"}});
    module.addPe(std::move(pe));
  };
  addWideExtensionPe("wide_zext0", "wide_zext0_input");
  addWideExtensionPe("wide_zext1", "wide_zext1_input");

  auto addWideBinaryPe = [&](std::string peResultName, std::string lhsInput,
                             std::string rhsInput,
                             std::vector<std::string> opList) {
    PeSpec pe;
    pe.inputs = {{"pa", std::move(lhsInput), "!fabric.bits<64>", ""},
                 {"pb", std::move(rhsInput), "!fabric.bits<64>", ""}};
    pe.resultNames = {std::move(peResultName)};
    pe.resultTypes = {"!fabric.bits<64>"};
    pe.fus.push_back(
        FuSpec{{{"lhs", "pa", "!fabric.bits<64>", ""},
                {"rhs", "pb", "!fabric.bits<64>", ""}},
               {"!fabric.bits<64>"},
               {FabricOpSpec{{"value"},
                             std::move(opList),
                             {"lhs", "rhs"},
                             {"!fabric.bits<64>", "!fabric.bits<64>"},
                             {"!fabric.bits<64>"},
                             {},
                             {}}},
               {"value"}});
    module.addPe(std::move(pe));
  };
  addWideBinaryPe("wide_product", "wide_mul_lhs", "wide_mul_rhs",
                  {"arith.muli"});
  addWideBinaryPe("wide_remainder", "wide_rem_lhs", "wide_rem_rhs",
                  {"arith.divui", "arith.remui"});

  PeSpec wideTruncPe;
  wideTruncPe.inputs = {{"pa", "wide_trunc_input", "!fabric.bits<64>", ""}};
  wideTruncPe.resultNames = {"wide_truncated_wide"};
  wideTruncPe.resultTypes = {"!fabric.bits<64>"};
  wideTruncPe.fus.push_back(FuSpec{{{"value", "pa", "!fabric.bits<64>", ""}},
                                   {"!fabric.bits<64>"},
                                   {FabricOpSpec{{"narrow"},
                                                 {"llvm.trunc"},
                                                 {"value"},
                                                 {"!fabric.bits<64>"},
                                                 {"!fabric.bits<32>"},
                                                 {},
                                                 {}}},
                                   {"narrow"},
                                   {"!fabric.bits<32>"}});
  module.addPe(std::move(wideTruncPe));

  addUnary32YieldPe("fp", "llvm.uitofp");
  addUnary32YieldPe("fp_negated", "llvm.fneg", "fp_negated_input");

  auto addCmpPe = [&](std::string resultName, std::string opName,
                      std::vector<std::string> predicates) {
    PeSpec pe;
    pe.inputs = {{"pa", "cmp_lhs", "!fabric.bits<32>", ""},
                 {"pb", "cmp_rhs", "!fabric.bits<32>", ""}};
    pe.resultNames = {resultName};
    pe.resultTypes = {"!fabric.bits<32>"};
    pe.fus.push_back(
        FuSpec{{{"lhs", "pa", "!fabric.bits<32>", ""},
                {"rhs", "pb", "!fabric.bits<32>", ""}},
               {"!fabric.bits<32>"},
               {FabricOpSpec{{"pred"},
                             {std::move(opName)},
                             {"lhs", "rhs"},
                             {"!fabric.bits<32>", "!fabric.bits<32>"},
                             {"!fabric.bits<1>"},
                             {{"predicate", std::move(predicates)}},
                             {}}},
               {"pred"},
               {"!fabric.bits<1>"}});
    module.addPe(std::move(pe));
  };
  addCmpPe("cmpf_pred", "arith.cmpf", {"oeq", "ogt", "ugt", "ule", "olt"});
  addCmpPe("cmpi_pred", "arith.cmpi", {"eq", "ne", "slt", "sgt", "ult", "ule"});
  addCmpPe("cmpi_pred_aux", "arith.cmpi",
           {"eq", "ne", "slt", "sgt", "ult", "ule"});

  PeSpec selectPe;
  selectPe.inputs = {{"pa", "select_pred", "!fabric.bits<32>", ""},
                     {"pb", "select_true", "!fabric.bits<32>", ""},
                     {"pc", "select_false", "!fabric.bits<32>", ""}};
  selectPe.resultNames = {"selected"};
  selectPe.resultTypes = {"!fabric.bits<32>"};
  auto makeSelectFu = []() {
    return FuSpec{{{"sel", "pa", "!fabric.bits<32>", "!fabric.bits<1>"},
                   {"when_true", "pb", "!fabric.bits<32>", ""},
                   {"when_false", "pc", "!fabric.bits<32>", ""}},
                  {"!fabric.bits<32>"},
                  {FabricOpSpec{{"selected_value"},
                                {"arith.select"},
                                {"sel", "when_true", "when_false"},
                                {"!fabric.bits<1>", "!fabric.bits<32>",
                                 "!fabric.bits<32>"},
                                {"!fabric.bits<32>"},
                                {},
                                {}}},
                  {"selected_value"}};
  };
  selectPe.fus.push_back(makeSelectFu());
  selectPe.fus.push_back(makeSelectFu());
  module.addPe(std::move(selectPe));

  auto addDemuxPe = [&](llvm::StringRef valueInput, llvm::StringRef falseResult,
                        llvm::StringRef trueResult) {
    PeSpec demuxPe;
    demuxPe.inputs = {{"pa", "demux_sel", "!fabric.bits<32>", ""},
                      {"pb", valueInput.str(), "!fabric.bits<32>", ""}};
    demuxPe.resultNames = {falseResult.str(), trueResult.str()};
    demuxPe.resultTypes = {"!fabric.bits<32>", "!fabric.bits<32>"};
    demuxPe.fus.push_back(
        FuSpec{{{"sel", "pa", "!fabric.bits<32>", "!fabric.bits<1>"},
                {"value", "pb", "!fabric.bits<32>", ""}},
               {"!fabric.bits<32>", "!fabric.bits<32>"},
               {FabricOpSpec{{"false_lane", "true_lane"},
                             {"dataflow.demux"},
                             {"sel", "value"},
                             {"!fabric.bits<1>", "!fabric.bits<32>"},
                             {"!fabric.bits<32>", "!fabric.bits<32>"},
                             {},
                             {}}},
               {"false_lane", "true_lane"}});
    module.addPe(std::move(demuxPe));
  };
  addDemuxPe("demux_value", "control_demux_false", "control_demux_true");
  addDemuxPe("demux_then_value", "compute_demux_false", "compute_demux_true");

  PeSpec muxPe;
  muxPe.inputs = {{"pa", "mux_sel", "!fabric.bits<32>", ""},
                  {"pb", "mux_false", "!fabric.bits<32>", ""},
                  {"pc", "mux_true", "!fabric.bits<32>", ""}};
  muxPe.resultNames = {"control_muxed"};
  muxPe.resultTypes = {"!fabric.bits<32>"};
  muxPe.fus.push_back(FuSpec{
      {{"sel", "pa", "!fabric.bits<32>", "!fabric.bits<1>"},
       {"false_lane", "pb", "!fabric.bits<32>", ""},
       {"true_lane", "pc", "!fabric.bits<32>", ""}},
      {"!fabric.bits<32>"},
      {FabricOpSpec{{"selected_lane"},
                    {"dataflow.mux"},
                    {"sel", "false_lane", "true_lane"},
                    {"!fabric.bits<1>", "!fabric.bits<32>", "!fabric.bits<32>"},
                    {"!fabric.bits<32>"},
                    {},
                    {}}},
      {"selected_lane"}});
  module.addPe(std::move(muxPe));

  PeSpec controlDemuxPe;
  controlDemuxPe.inputs = {
      {"pa", "control_token_demux_sel", "!fabric.bits<32>", ""},
      {"pb", "ctrl", "!fabric.bits<0>", "!fabric.bits<32>"}};
  controlDemuxPe.resultNames = {"control_token_demux_false",
                                "control_token_demux_true"};
  controlDemuxPe.resultTypes = {"!fabric.bits<32>", "!fabric.bits<32>"};
  controlDemuxPe.fus.push_back(
      FuSpec{{{"sel", "pa", "!fabric.bits<32>", "!fabric.bits<1>"},
              {"value", "pb", "!fabric.bits<32>", "!fabric.bits<0>"}},
             {"!fabric.bits<32>", "!fabric.bits<32>"},
             {FabricOpSpec{{"false_lane", "true_lane"},
                           {"dataflow.demux"},
                           {"sel", "value"},
                           {"!fabric.bits<1>", "!fabric.bits<0>"},
                           {"!fabric.bits<0>", "!fabric.bits<0>"},
                           {},
                           {}}},
             {"false_lane", "true_lane"},
             {"!fabric.bits<0>", "!fabric.bits<0>"}});
  module.addPe(std::move(controlDemuxPe));

  PeSpec controlMuxPe;
  controlMuxPe.inputs = {
      {"pa", "control_token_mux_sel", "!fabric.bits<32>", ""},
      {"pb", "control_token_mux_false", "!fabric.bits<0>", "!fabric.bits<32>"},
      {"pc", "control_token_mux_true", "!fabric.bits<0>", "!fabric.bits<32>"}};
  controlMuxPe.resultNames = {"control_token_muxed"};
  controlMuxPe.resultTypes = {"!fabric.bits<32>"};
  controlMuxPe.fus.push_back(FuSpec{
      {{"sel", "pa", "!fabric.bits<32>", "!fabric.bits<1>"},
       {"false_lane", "pb", "!fabric.bits<32>", "!fabric.bits<0>"},
       {"true_lane", "pc", "!fabric.bits<32>", "!fabric.bits<0>"}},
      {"!fabric.bits<32>"},
      {FabricOpSpec{{"selected_lane"},
                    {"dataflow.mux"},
                    {"sel", "false_lane", "true_lane"},
                    {"!fabric.bits<1>", "!fabric.bits<0>", "!fabric.bits<0>"},
                    {"!fabric.bits<0>"},
                    {},
                    {}}},
      {"selected_lane"},
      {"!fabric.bits<0>"}});
  module.addPe(std::move(controlMuxPe));

  PeSpec vectorSyncPe;
  vectorSyncPe.inputs = {{"pa", "sync_head", "!fabric.bits<0>", ""},
                         {"pb", "vector_sync_mid", "!fabric.bits<0>", ""},
                         {"pc", "sync_tail", "!fabric.bits<0>", ""},
                         {"pd", "sync_extra", "!fabric.bits<0>", ""},
                         {"pe", "done4", "!fabric.bits<0>", ""},
                         {"pf", "sync_lane5", "!fabric.bits<0>", ""},
                         {"pg", "store_done0", "!fabric.bits<0>", ""}};
  vectorSyncPe.resultTypes = {"!fabric.bits<0>"};
  vectorSyncPe.fus.push_back(FuSpec{
      {{"fa", "pa", "!fabric.bits<0>", ""},
       {"fb", "pb", "!fabric.bits<0>", ""},
       {"fc", "pc", "!fabric.bits<0>", ""},
       {"fd", "pd", "!fabric.bits<0>", ""},
       {"fe", "pe", "!fabric.bits<0>", ""},
       {"ff", "pf", "!fabric.bits<0>", ""},
       {"fg", "pg", "!fabric.bits<0>", ""}},
      {"!fabric.bits<0>"},
      {FabricOpSpec{{"sync_done0", "sync_done1", "sync_done2", "sync_done3",
                     "sync_done4", "sync_done5", "sync_done6"},
                    {"dataflow.sync"},
                    {"fa", "fb", "fc", "fd", "fe", "ff", "fg"},
                    {"!fabric.bits<0>", "!fabric.bits<0>", "!fabric.bits<0>",
                     "!fabric.bits<0>", "!fabric.bits<0>", "!fabric.bits<0>",
                     "!fabric.bits<0>"},
                    {"!fabric.bits<0>", "!fabric.bits<0>", "!fabric.bits<0>",
                     "!fabric.bits<0>", "!fabric.bits<0>", "!fabric.bits<0>",
                     "!fabric.bits<0>"},
                    {},
                    {{"bitmask", "1111111"}}}},
      {"sync_done0"}});
  module.addPe(std::move(vectorSyncPe));

  PeSpec syncPe;
  syncPe.inputs = {{"pc", "done0", "!fabric.bits<0>", ""},
                   {"pd", "sync_aux_done", "!fabric.bits<0>", ""}};
  syncPe.resultTypes = {"!fabric.bits<0>"};
  syncPe.fus.push_back(
      FuSpec{{{"fc", "pc", "!fabric.bits<0>", ""},
              {"fd", "pd", "!fabric.bits<0>", ""}},
             {"!fabric.bits<0>"},
             {FabricOpSpec{{"sync_done0", "sync_done1"},
                           {"dataflow.sync"},
                           {"fc", "fd"},
                           {"!fabric.bits<0>", "!fabric.bits<0>"},
                           {"!fabric.bits<0>", "!fabric.bits<0>"},
                           {},
                           {{"bitmask", "11"}}}},
             {"sync_done0"}});
  module.addPe(std::move(syncPe));

  PeSpec addrAddPe;
  addrAddPe.inputs = {{"pa", "addr_add_lhs", "!fabric.bits<32>", ""},
                      {"pb", "addr_add_rhs", "!fabric.bits<32>", ""}};
  addrAddPe.resultNames = {"addr_sum"};
  addrAddPe.resultTypes = {"!fabric.bits<32>"};
  addrAddPe.fus.push_back(
      FuSpec{{{"lhs", "pa", "!fabric.bits<32>", ""},
              {"rhs", "pb", "!fabric.bits<32>", ""}},
             {"!fabric.bits<32>"},
             {FabricOpSpec{{"sum"},
                           {"arith.addi", "arith.subi"},
                           {"lhs", "rhs"},
                           {"!fabric.bits<32>", "!fabric.bits<32>"},
                           {"!fabric.bits<32>"},
                           {},
                           {}}},
             {"sum"}});
  module.addPe(std::move(addrAddPe));

  PeSpec addrMaskPe;
  addrMaskPe.inputs = {{"pa", "addr_mask_lhs", "!fabric.bits<32>", ""},
                       {"pb", "addr_mask_rhs", "!fabric.bits<32>", ""}};
  addrMaskPe.resultNames = {"addr_masked"};
  addrMaskPe.resultTypes = {"!fabric.bits<32>"};
  addrMaskPe.fus.push_back(
      FuSpec{{{"lhs", "pa", "!fabric.bits<32>", ""},
              {"rhs", "pb", "!fabric.bits<32>", ""}},
             {"!fabric.bits<32>"},
             {FabricOpSpec{{"masked"},
                           {"arith.andi"},
                           {"lhs", "rhs"},
                           {"!fabric.bits<32>", "!fabric.bits<32>"},
                           {"!fabric.bits<32>"},
                           {},
                           {}}},
             {"masked"}});
  module.addPe(std::move(addrMaskPe));

  addBinary32Pe("aux_masked", "aux_mask_lhs", "aux_mask_rhs", "masked",
                {"arith.andi"});
  addBinary32Pe("aux_xor", "aux_xor_lhs", "aux_xor_rhs", "xor_value",
                {"arith.xori"});

  addFmulAddPe("mac_result2", "mac2_lhs", "mac2_rhs", "mac2_acc");
  addFmulAddPe("mac_result3", "mac3_lhs", "mac3_rhs", "mac3_acc");

  PeSpec stateCarryPe;
  stateCarryPe.inputs = {{"pa", "state_carry_cond", "!fabric.bits<32>", ""},
                         {"pb", "state_carry_init", "!fabric.bits<32>", ""},
                         {"pc", "state_carry_next", "!fabric.bits<32>", ""}};
  stateCarryPe.resultNames = {"state_carry"};
  stateCarryPe.resultTypes = {"!fabric.bits<32>"};
  stateCarryPe.fus.push_back(makeCarryFu());
  stateCarryPe.fus.push_back(makeCarryFu());
  module.addPe(std::move(stateCarryPe));

  auto addSingleResultBits32Switch =
      [&](llvm::StringRef result,
          std::initializer_list<llvm::StringRef> inputs) {
        llvm::ArrayRef<llvm::StringRef> inputRefs(inputs.begin(),
                                                  inputs.size());
        module.addExactBodyLine(valueName(result) +
                                " = fabric.switch [spatial] " +
                                valueList(inputRefs));
        module.addExactBodyLine("  [{connectivity_table = [\"" +
                                std::string(inputRefs.size(), '1') + "\"]}]");
        module.addExactBodyLine("  : " + bits32TypeList(inputRefs.size()) +
                                " -> !fabric.bits<32>");
      };
  auto addSingleResultBits64Switch =
      [&](llvm::StringRef result,
          std::initializer_list<llvm::StringRef> inputs) {
        llvm::ArrayRef<llvm::StringRef> inputRefs(inputs.begin(),
                                                  inputs.size());
        module.addExactBodyLine(valueName(result) +
                                " = fabric.switch [spatial] " +
                                valueList(inputRefs));
        module.addExactBodyLine("  [{connectivity_table = [\"" +
                                std::string(inputRefs.size(), '1') + "\"]}]");
        module.addExactBodyLine(
            "  : " + uniformTypeList(inputRefs.size(), "!fabric.bits<64>") +
            " -> !fabric.bits<64>");
      };
  addSingleResultBits32Switch("logic_mask_lhs",
                              {"i32a", "data0", "data1", "bit_carry",
                               "addr_unscaled", "cmpi_pred", "cmpi_pred_aux"});
  addSingleResultBits32Switch(
      "logic_mask_rhs", {"i32b", "i32c", "reduction_scale", "fp_invariant",
                         "bit_invariant", "bit_invariant_aux0",
                         "bit_invariant_aux1", "cmpi_pred", "cmpi_pred_aux"});
  const std::initializer_list<llvm::StringRef> auxMaskLhsSources = {
      "carried_scan", "bit_carry", "state_carry", "selected", "addr_shifted",
      "running",      "idx",       "data0",       "data1"};
  const std::initializer_list<llvm::StringRef> auxMaskRhsSources = {
      "i32a",
      "i32b",
      "i32c",
      "i32d",
      "reduction_scale",
      "fp_invariant",
      "bit_invariant",
      "bit_invariant_aux0",
      "bit_invariant_aux1",
      "addr_shift_const",
      "addr_aux_const",
      "addr_bias_const"};
  const std::initializer_list<llvm::StringRef> auxXorLhsSources = {
      "selected",    "addr_shifted", "carried_scan",
      "bit_carry",   "state_carry",  "logic_masked",
      "addr_masked", "int_xor",      "aux_masked"};
  const std::initializer_list<llvm::StringRef> auxXorRhsSources = {
      "carried_scan",
      "bit_carry",
      "state_carry",
      "i32a",
      "i32b",
      "i32c",
      "i32d",
      "reduction_scale",
      "fp_invariant",
      "bit_invariant",
      "bit_invariant_aux0",
      "bit_invariant_aux1"};
  addSingleResultBits32Switch("aux_mask_lhs", auxMaskLhsSources);
  addSingleResultBits32Switch("aux_mask_rhs", auxMaskRhsSources);
  addSingleResultBits32Switch("aux_xor_lhs", auxXorLhsSources);
  addSingleResultBits32Switch("aux_xor_rhs", auxXorRhsSources);
  addSingleResultBits32Switch("int_add_lhs", {"i32a",
                                              "data1",
                                              "data0",
                                              "carried_scan",
                                              "squared_data",
                                              "bit_carry",
                                              "reduction_scale",
                                              "int_product",
                                              "int_product_aux",
                                              "fp_invariant",
                                              "bit_invariant",
                                              "bit_invariant_aux0",
                                              "bit_invariant_aux1",
                                              "aux_invariant0",
                                              "aux_invariant1",
                                              "aux_invariant2",
                                              "cast0_result",
                                              "cast1_result",
                                              "cast2_result",
                                              "cast3_result"});
  addSingleResultBits32Switch(
      "int_add_rhs",
      {"i32b", "data0", "data1", "fp_invariant", "idx", "reduction_scale",
       "bit_invariant", "bit_invariant_aux0", "bit_invariant_aux1", "int_rem",
       "aux_idx", "aux_active_idx", "cast0_result", "cast1_result",
       "cast2_result", "cast3_result"});
  addSingleResultBits32Switch("int_mul_lhs",
                              {"i32a", "int_xor", "data0", "data1", "int_div0",
                               "int_div1", "aux_idx", "aux_active_idx",
                               "bit_invariant", "bit_invariant_aux0",
                               "bit_invariant_aux1", "cast0_result",
                               "cast1_result", "cast2_result", "cast3_result"});
  addSingleResultBits32Switch(
      "int_mul_rhs",
      {"i32b", "data0", "data1", "reduction_scale", "bit_invariant",
       "bit_invariant_aux0", "bit_invariant_aux1", "aux_invariant0",
       "aux_invariant1", "aux_invariant2", "fp_invariant", "cast0_result",
       "cast1_result", "cast2_result", "cast3_result"});
  addSingleResultBits32Switch("int_mul_aux_lhs",
                              {"i32a", "int_xor", "data0", "data1", "int_div0",
                               "int_div1", "aux_idx", "aux_active_idx"});
  addSingleResultBits32Switch(
      "int_mul_aux_rhs",
      {"i32b", "data0", "data1", "reduction_scale", "bit_invariant",
       "bit_invariant_aux0", "bit_invariant_aux1", "aux_invariant0",
       "aux_invariant1", "aux_invariant2", "fp_invariant"});
  addSingleResultBits32Switch(
      "int_div0_lhs",
      {"int_sum", "addr_sum", "aux_idx", "aux_active_idx", "i32b", "i32c"});
  addSingleResultBits32Switch("int_div0_rhs",
                              {"i32c", "reduction_scale", "fp_invariant",
                               "bit_invariant", "bit_invariant_aux0",
                               "bit_invariant_aux1", "aux_invariant0",
                               "aux_invariant1", "aux_invariant2"});
  addSingleResultBits32Switch(
      "int_div1_lhs",
      {"int_sum", "addr_sum", "aux_idx", "aux_active_idx", "i32b", "i32c"});
  addSingleResultBits32Switch("int_div1_rhs",
                              {"i32c", "reduction_scale", "fp_invariant",
                               "bit_invariant", "bit_invariant_aux0",
                               "bit_invariant_aux1", "aux_invariant0",
                               "aux_invariant1", "aux_invariant2"});
  addSingleResultBits32Switch("int_rem_lhs",
                              {"aux_idx", "aux_active_idx", "i32a", "i32b"});
  addSingleResultBits32Switch(
      "int_rem_rhs", {"reduction_scale", "bit_invariant", "bit_invariant_aux0",
                      "bit_invariant_aux1", "aux_invariant0", "aux_invariant1",
                      "aux_invariant2", "i32d"});
  addSingleResultBits32Switch(
      "uint_rem_lhs", {"int_product", "aux_idx", "aux_active_idx", "i32b"});
  addSingleResultBits32Switch(
      "uint_rem_rhs", {"i32c", "reduction_scale", "bit_invariant",
                       "bit_invariant_aux0", "bit_invariant_aux1",
                       "aux_invariant0", "aux_invariant1", "aux_invariant2"});
  addSingleResultBits32Switch(
      "int_or_lhs", {"i32a", "logic_masked", "data0", "data1", "addr_shifted"});
  addSingleResultBits32Switch("int_or_rhs",
                              {"i32b", "logic_masked", "data0", "data1"});
  addSingleResultBits32Switch("int_xor_lhs", {"i32a", "rotated", "logic_masked",
                                              "data0", "packed_sat", "selected",
                                              "addr_masked", "aux_masked"});
  addSingleResultBits32Switch(
      "int_xor_rhs",
      {"i32b", "data1", "data0", "logic_masked", "reduction_scale",
       "fp_invariant", "bit_invariant", "bit_invariant_aux0",
       "bit_invariant_aux1", "carried_scan", "bit_carry", "state_carry",
       "addr_masked", "selected", "aux_masked"});
  addSingleResultBits32Switch("packed_sat_lhs",
                              {"i32a", "reduction_scale", "fp_invariant",
                               "bit_invariant", "bit_invariant_aux0",
                               "bit_invariant_aux1", "cast0_result",
                               "cast1_result", "cast2_result", "cast3_result"});
  addSingleResultBits32Switch("packed_sat_rhs",
                              {"logic_masked", "addr_masked", "data0", "data1",
                               "i32b", "reduction_scale", "fp_invariant",
                               "bit_invariant", "bit_invariant_aux0",
                               "bit_invariant_aux1", "cast0_result",
                               "cast1_result", "cast2_result", "cast3_result"});
  module.addExactBodyLine(
      "%rotate_lhs = fabric.switch [spatial] %i32a, %data1, %data0, "
      "%logic_masked, %int_sum, %int_product");
  module.addExactBodyLine("  [{connectivity_table = [\"111111\"]}]");
  module.addExactBodyLine(
      "  : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, "
      "!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> "
      "!fabric.bits<32>");
  module.addExactBodyLine(
      "%rotate_rhs = fabric.switch [spatial] %i32b, %data1, %data0, "
      "%logic_masked, %int_sum, %int_product");
  module.addExactBodyLine("  [{connectivity_table = [\"111111\"]}]");
  module.addExactBodyLine(
      "  : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, "
      "!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> "
      "!fabric.bits<32>");
  module.addExactBodyLine(
      "%rotate_amount = fabric.switch [spatial] %i32c, %data0, "
      "%reduction_scale, %addr_shift_const");
  module.addExactBodyLine("  [{connectivity_table = [\"1111\"]}]");
  module.addExactBodyLine(
      "  : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, "
      "!fabric.bits<32>) -> !fabric.bits<32>");
  addSingleResultBits32Switch(
      "cmp_lhs", {"i32a", "logic_masked", "data0", "data1", "bit_carry",
                  "running", "addr_sum", "addr_masked", "aux_masked"});
  addSingleResultBits32Switch("cmp_rhs", {"i32b", "i32c", "reduction_scale",
                                          "data1", "data0", "fp_invariant",
                                          "bit_invariant", "bit_invariant_aux0",
                                          "bit_invariant_aux1", "aux_masked"});
  addSingleResultBits32Switch(
      "select_pred", {"i32a", "logic_masked", "addr_masked", "cmpi_pred",
                      "cmpi_pred_aux", "cmpf_pred", "aux_masked"});
  addSingleResultBits32Switch(
      "select_true",
      {"i32b", "idx", "data1", "rotated", "data0", "int_sum", "addr_sum",
       "addr_shifted", "reduction_scale", "bit_invariant", "bit_invariant_aux0",
       "bit_invariant_aux1", "cast0_result", "cast1_result", "cast2_result",
       "cast3_result", "aux_xor", "carried_scan"});
  addSingleResultBits32Switch(
      "select_false", {"i32c", "rotated", "data0", "data1", "carried_scan",
                       "bit_carry", "addr_shift_const", "addr_aux_const",
                       "addr_bias_const", "aux_xor", "running"});
  addSingleResultBits32Switch(
      "gate_cond", {"aux_rwc", "logic_masked", "addr_masked", "cmpi_pred",
                    "cmpi_pred_aux", "cmpf_pred", "fp_gate", "i32a"});
  addSingleResultBits32Switch(
      "gate_value", {"aux_idx", "idx", "running", "addr_sum", "int_sum",
                     "squared_data", "carried_scan", "cast0_result",
                     "cast1_result", "cast2_result", "cast3_result"});
  addSingleResultBits32Switch("demux_sel", {"logic_masked", "addr_masked",
                                            "cmpi_pred", "cmpi_pred_aux",
                                            "cmpf_pred", "fp_gate", "i32a"});
  addSingleResultBits32Switch("demux_value",
                              {"carried_scan", "bit_carry", "state_carry",
                               "fp_invariant", "reduction_scale", "running"});
  addSingleResultBits32Switch(
      "demux_then_value",
      {"mac_result", "mac_result1", "mac_result2", "mac_result3", "fp_running",
       "fp_running_aux", "scaled_reduction", "data0", "data1", "int_sum",
       "addr_sum", "int_product", "int_product_aux", "selected"});
  addSingleResultBits32Switch("mux_sel", {"logic_masked", "addr_masked",
                                          "cmpi_pred", "cmpi_pred_aux",
                                          "cmpf_pred", "fp_gate", "i32a"});
  addSingleResultBits32Switch("mux_false",
                              {"control_demux_false", "carried_scan",
                               "bit_carry", "state_carry", "fp_invariant"});
  addSingleResultBits32Switch(
      "mux_true", {"compute_demux_true", "mac_result", "mac_result1",
                   "mac_result2", "mac_result3", "fp_running", "fp_running_aux",
                   "scaled_reduction", "data0", "data1"});
  addSingleResultBits32Switch("control_token_demux_sel",
                              {"logic_masked", "addr_masked", "cmpi_pred",
                               "cmpi_pred_aux", "cmpf_pred", "fp_gate",
                               "i32a"});
  module.addExactBodyLine(
      "%control_token_demux_false_token = fabric.fifo "
      "%control_token_demux_false [max_depth = 1, bypassable = true] "
      "{bypassed = true}");
  module.addExactBodyLine("  : !fabric.bits<32> to !fabric.bits<0>");
  module.addExactBodyLine(
      "%control_token_demux_true_token = fabric.fifo "
      "%control_token_demux_true [max_depth = 1, bypassable = true] "
      "{bypassed = true}");
  module.addExactBodyLine("  : !fabric.bits<32> to !fabric.bits<0>");
  module.addExactBodyLine(
      "%control_token_muxed_token = fabric.fifo %control_token_muxed "
      "[max_depth = 1, bypassable = true] {bypassed = true}");
  module.addExactBodyLine("  : !fabric.bits<32> to !fabric.bits<0>");
  addSingleResultBits32Switch("control_token_mux_sel",
                              {"logic_masked", "addr_masked", "cmpi_pred",
                               "cmpi_pred_aux", "cmpf_pred", "fp_gate",
                               "i32a"});
  module.addExactBodyLine(
      "%control_token_mux_false = fabric.switch [spatial] "
      "%control_token_demux_false_token, %store_done0, %ctrl");
  module.addExactBodyLine("  [{connectivity_table = [\"111\"]}]");
  module.addExactBodyLine(
      "  : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>) -> "
      "!fabric.bits<0>");
  module.addExactBodyLine(
      "%control_token_mux_true = fabric.switch [spatial] "
      "%store_done0, %control_token_demux_true_token, %ctrl");
  module.addExactBodyLine("  [{connectivity_table = [\"111\"]}]");
  module.addExactBodyLine(
      "  : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>) -> "
      "!fabric.bits<0>");
  addSingleResultBits32Switch("load1_addr",
                              {"idx", "i32b", "addr_unscaled", "cast0_result",
                               "cast1_result", "cast2_result", "cast3_result",
                               "running", "addr_sum", "squared_data", "int_sum",
                               "carried_scan", "aux_idx", "aux_active_idx",
                               "selected", "logic_masked", "addr_masked"});
  addSingleResultBits32Switch(
      "cast0_input", {"i32a", "data0", "data1", "logic_masked", "packed_sat",
                      "idx", "running", "int_sum", "addr_sum", "uint_rem"});
  addSingleResultBits32Switch("cast1_input",
                              {"i32a", "data0", "data1", "logic_masked",
                               "packed_sat", "idx", "running", "int_sum",
                               "addr_sum", "uint_rem", "cast0_result"});
  addSingleResultBits32Switch(
      "cast2_input",
      {"i32a", "data0", "data1", "logic_masked", "packed_sat", "idx", "running",
       "int_sum", "addr_sum", "uint_rem", "cast0_result", "cast1_result"});
  addSingleResultBits32Switch(
      "cast3_input", {"i32a", "data0", "data1", "logic_masked", "packed_sat",
                      "idx", "running", "int_sum", "addr_sum", "uint_rem",
                      "cast0_result", "cast1_result", "cast2_result"});
  addSingleResultBits32Switch(
      "wide_zext0_input",
      {"data0", "data1", "i32a", "cast0_result", "cast1_result"});
  addSingleResultBits32Switch(
      "wide_zext1_input",
      {"data1", "data0", "i32b", "cast0_result", "cast1_result"});
  addSingleResultBits64Switch("wide_mul_lhs",
                              {"wide_zext1", "wide_zext0", "i64a", "i64b"});
  addSingleResultBits64Switch("wide_mul_rhs",
                              {"wide_zext0", "wide_zext1", "i64a", "i64c"});
  addSingleResultBits64Switch(
      "wide_rem_lhs", {"wide_product", "wide_zext0", "wide_zext1", "i64a"});
  addSingleResultBits64Switch(
      "wide_rem_rhs", {"i64a", "i64b", "i64c", "wide_zext0", "wide_zext1"});
  addSingleResultBits64Switch(
      "wide_trunc_input",
      {"wide_remainder", "wide_product", "wide_zext0", "wide_zext1"});
  addSingleResultBits32Switch("fp_negated_input",
                              {"data0", "data1", "data2", "data3", "data4",
                               "data5", "fp_running", "fp_running_aux",
                               "fp_diff", "fp_diff_aux", "scaled_reduction"});
  addSingleResultBits32Switch("load2_addr",
                              {"i32c", "cast0_result", "cast1_result",
                               "cast2_result", "cast3_result", "idx",
                               "addr_sum", "running", "squared_data", "int_sum",
                               "aux_idx", "aux_active_idx", "data0", "data1"});
  module.addExactBodyLine(
      "%wide_truncated = fabric.fifo %wide_truncated_wide "
      "[max_depth = 1, bypassable = true] {bypassed = true}");
  module.addExactBodyLine("  : !fabric.bits<64> to !fabric.bits<32>");
  addSingleResultBits32Switch("store0_value", {"scan_store_value",
                                               "fp_running",
                                               "fp_running_aux",
                                               "running",
                                               "mac_result",
                                               "mac_result1",
                                               "mac_result2",
                                               "mac_result3",
                                               "data0",
                                               "data1",
                                               "selected",
                                               "rotated",
                                               "addr_masked",
                                               "logic_masked",
                                               "int_xor",
                                               "packed_sat",
                                               "cast0_result",
                                               "cast1_result",
                                               "cast2_result",
                                               "cast3_result",
                                               "abs_data",
                                               "scaled_reduction",
                                               "int_product",
                                               "reduction_scale",
                                               "int_sum",
                                               "fp_diff",
                                               "fp_diff_aux",
                                               "compute_demux_false",
                                               "compute_demux_true",
                                               "wide_truncated",
                                               "fp_negated"});
  module.addExactBodyLine(
      "%store1_value = fabric.switch [spatial] %i32d, %selected");
  module.addExactBodyLine("  [{connectivity_table = [\"11\"]}]");
  module.addExactBodyLine(
      "  : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>");
  module.addExactBodyLine(
      "%vector_sync_mid = fabric.switch [spatial] %done1, %store_done0, "
      "%control_token_muxed_token");
  module.addExactBodyLine("  [{connectivity_table = [\"111\"]}]");
  module.addExactBodyLine(
      "  : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>) -> "
      "!fabric.bits<0>");
  module.addExactBodyLine(
      "%sync_head = fabric.switch [spatial] %done0, %store_done0");
  module.addExactBodyLine("  [{connectivity_table = [\"11\"]}]");
  module.addExactBodyLine(
      "  : (!fabric.bits<0>, !fabric.bits<0>) -> !fabric.bits<0>");
  module.addExactBodyLine(
      "%sync_tail = fabric.switch [spatial] %store_done0, %done2");
  module.addExactBodyLine("  [{connectivity_table = [\"11\"]}]");
  module.addExactBodyLine(
      "  : (!fabric.bits<0>, !fabric.bits<0>) -> !fabric.bits<0>");
  module.addExactBodyLine(
      "%sync_extra = fabric.switch [spatial] %store_done1, %done3, "
      "%store_done0");
  module.addExactBodyLine("  [{connectivity_table = [\"111\"]}]");
  module.addExactBodyLine(
      "  : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>) -> "
      "!fabric.bits<0>");
  module.addExactBodyLine(
      "%sync_lane5 = fabric.switch [spatial] %done5, %store_done0");
  module.addExactBodyLine("  [{connectivity_table = [\"11\"]}]");
  module.addExactBodyLine(
      "  : (!fabric.bits<0>, !fabric.bits<0>) -> !fabric.bits<0>");
  addSingleResultBits32Switch("addr_add_lhs",
                              {"idx", "i32a", "i32b", "i32c", "squared_data",
                               "int_product", "running", "reduction_scale",
                               "int_product_aux"});
  module.addExactBodyLine(
      "%addr_add_rhs = fabric.switch [spatial] %fp_invariant, "
      "%reduction_scale, %i32a, %i32b, %idx, %int_rem, %aux_idx, "
      "%aux_active_idx, %carried_scan");
  module.addExactBodyLine("  [{connectivity_table = [\"111111111\"]}]");
  module.addExactBodyLine(
      "  : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, "
      "!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, "
      "!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)");
  module.addExactBodyLine("  -> !fabric.bits<32>");
  addSingleResultBits32Switch(
      "addr_mask_lhs",
      {"addr_sum", "idx", "data0", "data1", "logic_masked", "carried_scan",
       "bit_carry", "state_carry", "selected", "aux_masked", "aux_xor"});
  addSingleResultBits32Switch("addr_mask_rhs",
                              {"reduction_scale", "fp_invariant", "i32b",
                               "i32c", "int_xor", "packed_sat", "logic_masked",
                               "bit_invariant", "bit_invariant_aux0",
                               "bit_invariant_aux1", "aux_masked", "aux_xor"});
  module.addExactBodyLine(
      "%addr_unscale_lhs = fabric.switch [spatial] %i32a, %addr_shifted, "
      "%bit_carry, %data0, %squared_data, %int_product, %int_product_aux");
  module.addExactBodyLine("  [{connectivity_table = [\"1111111\"]}]");
  module.addExactBodyLine(
      "  : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, "
      "!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, "
      "!fabric.bits<32>) -> !fabric.bits<32>");
  module.addExactBodyLine(
      "%addr_unscale_rhs = fabric.switch [spatial] %i32b, %addr_shift_const, "
      "%reduction_scale, %fp_invariant, %bit_invariant, "
      "%bit_invariant_aux0, %bit_invariant_aux1");
  module.addExactBodyLine("  [{connectivity_table = [\"1111111\"]}]");
  module.addExactBodyLine(
      "  : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, "
      "!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, "
      "!fabric.bits<32>) -> !fabric.bits<32>");
  addSingleResultBits32Switch(
      "addr_shift_lhs", {"i32a", "carried_scan", "idx", "bit_carry",
                         "state_carry", "selected", "aux_masked", "aux_xor"});
  addSingleResultBits32Switch("addr_shift_rhs",
                              {"i32b", "reduction_scale", "bit_invariant",
                               "bit_invariant_aux0", "bit_invariant_aux1",
                               "addr_shift_const", "addr_aux_const",
                               "addr_bias_const", "aux_masked", "aux_xor"});
  addSingleResultBits32Switch("load0_addr", {"idx",
                                             "addr_masked",
                                             "addr_shifted",
                                             "addr_unscaled",
                                             "carried_scan",
                                             "bit_carry",
                                             "state_carry",
                                             "squared_data",
                                             "running",
                                             "addr_sum",
                                             "int_product",
                                             "int_sum",
                                             "aux_idx",
                                             "aux_active_idx",
                                             "cast0_result",
                                             "cast1_result",
                                             "cast2_result",
                                             "cast3_result",
                                             "selected",
                                             "addr_shift_const",
                                             "addr_aux_const",
                                             "addr_bias_const"});
  addSingleResultBits32Switch(
      "load3_addr", {"i32d", "carried_scan", "idx", "squared_data", "running",
                     "addr_sum", "int_sum", "aux_idx", "aux_active_idx"});
  addSingleResultBits32Switch("load4_addr",
                              {"idx", "squared_data", "running", "addr_sum",
                               "int_product", "int_sum", "addr_unscaled",
                               "addr_shifted", "aux_idx", "aux_active_idx"});
  addSingleResultBits32Switch("load5_addr",
                              {"idx", "squared_data", "running", "addr_sum",
                               "int_product", "int_sum", "addr_unscaled",
                               "addr_shifted", "aux_idx", "aux_active_idx"});
  addSingleResultBits32Switch("store0_addr",
                              {"idx", "addr_unscaled", "carried_scan",
                               "addr_shift_const", "state_carry",
                               "addr_bias_const", "int_sum", "addr_sum",
                               "aux_idx", "running", "aux_active_idx",
                               "control_demux_false", "control_demux_true"});
  addSingleResultBits32Switch(
      "aux_stream_lb",
      {"addr_shift_const", "addr_aux_const", "addr_bias_const"});
  addSingleResultBits32Switch(
      "aux_stream_ub", {"int_product", "int_product_aux", "squared_data"});
  addSingleResultBits32Switch(
      "aux_stream_step",
      {"addr_shift_const", "addr_aux_const", "addr_bias_const"});
  addSingleResultBits32Switch("aux_invariant_cond", {"aux_rwc", "fp_gate"});
  addSingleResultBits32Switch(
      "aux_invariant0_value",
      {"i32a", "i32b", "i32c", "i32d", "reduction_scale", "bit_invariant",
       "bit_invariant_aux0", "bit_invariant_aux1", "fp_invariant",
       "addr_shift_const", "addr_aux_const", "addr_bias_const"});
  addSingleResultBits32Switch(
      "aux_invariant1_value",
      {"i32a", "i32b", "i32c", "i32d", "reduction_scale", "bit_invariant",
       "bit_invariant_aux0", "bit_invariant_aux1", "fp_invariant",
       "aux_invariant0", "addr_shift_const", "addr_aux_const",
       "addr_bias_const"});
  module.addExactBodyLine(
      "%store0_ctrl = fabric.switch [spatial] %ctrl, %done0, %done1, %done2, "
      "%done3, %done4, %done5, %control_token_demux_false_token, "
      "%control_token_demux_true_token");
  module.addExactBodyLine("  [{connectivity_table = [\"111111111\"]}]");
  module.addExactBodyLine(
      "  : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, "
      "!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, "
      "!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>) -> "
      "!fabric.bits<0>");
  module.addExactBodyLine(
      "%store1_ctrl = fabric.switch [spatial] %ctrl, %done0, %done1, %done2, "
      "%done3, %done4, %done5, %control_token_demux_false_token, "
      "%control_token_demux_true_token");
  module.addExactBodyLine("  [{connectivity_table = [\"111111111\"]}]");
  module.addExactBodyLine(
      "  : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, "
      "!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, "
      "!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>) -> "
      "!fabric.bits<0>");
  module.addExactBodyLine(
      "%data0, %done0, %data1, %done1, %data2, %done2, %data3, %done3, "
      "%data4, %done4, %data5, %done5, %store_done0, %store_done1 =");
  module.addExactBodyLine(
      "    fabric.mem [spatial] mgr(%mgr) load(%load0_addr, %ctrl, "
      "%load1_addr, "
      "%ctrl, %load2_addr, %ctrl, %load3_addr, %ctrl, %load4_addr, %ctrl, "
      "%load5_addr, %ctrl)");
  module.addExactBodyLine(
      "                              store(%store0_addr, %store0_value, "
      "%store0_ctrl, %i32c, %store1_value, %store1_ctrl)");
  module.addExactBodyLine(
      "      [{load_group_size = 6 : i32, store_group_size = 2 : i32}]");
  module.addExactBodyLine(
      "      : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, "
      "!fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, "
      "!fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, "
      "!fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, "
      "!fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, "
      "!fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, "
      "!fabric.bits<32>, !fabric.bits<0>)");
  module.addExactBodyLine(
      "      -> (!fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, "
      "!fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, "
      "!fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, "
      "!fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, "
      "!fabric.bits<0>, !fabric.bits<0>)");
  addSingleResultBits32Switch(
      "mul_lhs_input",
      {"data0", "data1", "data2", "idx", "data4", "int_div0", "int_div1",
       "aux_idx", "aux_active_idx", "bit_invariant", "bit_invariant_aux0",
       "bit_invariant_aux1", "cast0_result", "cast1_result", "cast2_result",
       "cast3_result", "aux_invariant0", "aux_invariant1", "aux_invariant2"});
  addSingleResultBits32Switch(
      "mul_rhs_input",
      {"data0", "data1", "data2", "data4", "reduction_scale", "bit_invariant",
       "bit_invariant_aux0", "bit_invariant_aux1", "aux_invariant0",
       "aux_invariant1", "aux_invariant2", "fp_invariant", "cast0_result",
       "cast1_result", "cast2_result", "cast3_result"});
  module.addExactBodyLine(
      "%reduction_input = fabric.switch [spatial] %data0, %abs_data, "
      "%squared_data");
  module.addExactBodyLine("  [{connectivity_table = [\"111\"]}]");
  module.addExactBodyLine(
      "  : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)");
  module.addExactBodyLine("  -> !fabric.bits<32>");
  addSingleResultBits32Switch("stream_sum_lhs",
                              {"reduction_input", "carried_scan", "bit_carry",
                               "state_carry", "int_product", "int_product_aux",
                               "bit_invariant", "bit_invariant_aux0",
                               "bit_invariant_aux1", "cast0_result",
                               "cast1_result", "cast2_result", "cast3_result"});
  addSingleResultBits32Switch(
      "stream_sum_rhs",
      {"carried_scan", "fp_invariant", "reduction_scale", "bit_invariant_aux1",
       "int_rem", "aux_idx", "aux_invariant0", "aux_invariant1",
       "aux_invariant2", "bit_invariant", "bit_invariant_aux0", "cast0_result",
       "cast1_result", "cast2_result", "cast3_result"});
  addSingleResultBits32Switch(
      "scan_init",
      {"i32a", "addr_shift_const", "addr_aux_const", "addr_bias_const"});
  addSingleResultBits32Switch(
      "scan_scale",
      {"i32b", "addr_shift_const", "addr_aux_const", "addr_bias_const"});
  addSingleResultBits32Switch("fp_lhs",
                              {"carried_scan", "data0", "data2", "data4",
                               "reduction_scale", "mac_result1"});
  addSingleResultBits32Switch(
      "fp_rhs", {"data0", "data1", "data3", "data5", "reduction_scale"});
  addSingleResultBits32Switch("fp_lhs_aux",
                              {"bit_carry", "state_carry", "carried_scan",
                               "data0", "data1", "data2", "data4",
                               "reduction_scale", "mac_result1"});
  addSingleResultBits32Switch("fp_rhs_aux", {"data1", "data0", "data3", "data5",
                                             "reduction_scale", "fp_invariant",
                                             "bit_invariant"});
  module.addExactBodyLine(
      "%fp_diff_lhs = fabric.switch [spatial] %i32a, %data0");
  module.addExactBodyLine("  [{connectivity_table = [\"11\"]}]");
  module.addExactBodyLine(
      "  : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>");
  addSingleResultBits32Switch("fp_diff_rhs",
                              {"i32b", "fp_invariant", "data1", "fp_div"});
  addSingleResultBits32Switch("fp_diff_aux_lhs", {"data1", "data0", "i32a"});
  addSingleResultBits32Switch("fp_diff_aux_rhs",
                              {"bit_invariant", "fp_invariant",
                               "bit_invariant_aux0", "bit_invariant_aux1",
                               "aux_invariant0", "aux_invariant1",
                               "aux_invariant2", "i32b", "data1", "fp_div"});
  addSingleResultBits32Switch("fp_div_lhs", {"data1", "data0"});
  addSingleResultBits32Switch("fp_div_rhs",
                              {"data2", "fp_invariant", "reduction_scale"});
  addSingleResultBits32Switch(
      "fp_invariant_value",
      {"i32b", "addr_shift_const", "addr_aux_const", "addr_bias_const"});
  addSingleResultBits32Switch("bit_invariant_aux1_value",
                              {"i32b", "reduction_scale", "addr_shift_const",
                               "addr_aux_const", "addr_bias_const"});
  addSingleResultBits32Switch(
      "scaled_reduction_lhs",
      {"carried_scan", "fp_running", "fp_running_aux", "data1", "data3",
       "data5", "data0", "bit_invariant", "bit_invariant_aux0",
       "bit_invariant_aux1", "aux_invariant0", "aux_invariant1",
       "aux_invariant2", "fp_negated"});
  addSingleResultBits32Switch("scaled_reduction_rhs",
                              {"reduction_scale", "data4", "data5", "data1",
                               "data3", "state_carry", "bit_carry",
                               "aux_invariant0", "aux_invariant1",
                               "aux_invariant2", "fp_negated"});
  addSingleResultBits32Switch(
      "mac_lhs",
      {"i32a", "data0", "data2", "data4", "fp_diff", "fp_diff_aux",
       "scaled_reduction", "fp_invariant", "bit_invariant",
       "bit_invariant_aux0", "data1", "bit_invariant_aux1", "reduction_scale"});
  addSingleResultBits32Switch("mac_rhs", {"i32b", "data1", "data3", "data5",
                                          "fp_diff", "fp_diff_aux", "data0",
                                          "bit_carry", "state_carry"});
  addSingleResultBits32Switch("mac_acc",
                              {"i32c", "carried_scan", "bit_carry",
                               "scaled_reduction", "state_carry", "data0"});
  addSingleResultBits32Switch(
      "mac1_lhs", {"i32a", "data2", "data4", "data0", "fp_diff", "fp_diff_aux",
                   "fp_invariant", "bit_invariant", "bit_invariant_aux0",
                   "bit_invariant_aux1", "reduction_scale"});
  addSingleResultBits32Switch(
      "mac1_rhs", {"i32b", "data3", "data5", "data1", "fp_diff", "fp_diff_aux",
                   "bit_carry", "state_carry", "carried_scan"});
  addSingleResultBits32Switch("mac1_acc",
                              {"i32c", "mac_result", "scaled_reduction",
                               "carried_scan", "bit_carry", "state_carry"});
  addSingleResultBits32Switch(
      "mac2_lhs",
      {"i32a", "data0", "data2", "data4", "fp_invariant", "bit_invariant",
       "bit_invariant_aux0", "bit_invariant_aux1", "reduction_scale"});
  addSingleResultBits32Switch("mac2_rhs",
                              {"i32b", "data0", "data1", "data3", "data5",
                               "bit_carry", "state_carry", "carried_scan"});
  addSingleResultBits32Switch("mac2_acc",
                              {"mac_result1", "mac_result", "scaled_reduction",
                               "bit_carry", "state_carry"});
  addSingleResultBits32Switch(
      "mac3_lhs",
      {"i32a", "data0", "data2", "data4", "fp_invariant", "bit_invariant",
       "bit_invariant_aux0", "bit_invariant_aux1", "reduction_scale"});
  addSingleResultBits32Switch("mac3_rhs",
                              {"i32b", "data0", "data1", "data3", "data5",
                               "bit_carry", "state_carry", "carried_scan",
                               "fp_running", "fp_running_aux"});
  addSingleResultBits32Switch(
      "mac3_acc", {"mac_result2", "mac_result1", "mac_result",
                   "scaled_reduction", "bit_carry", "state_carry", "data4"});
  module.addExactBodyLine(
      "%bit_carry_cond = fabric.switch [spatial] %i32a, %fp_gate");
  module.addExactBodyLine("  [{connectivity_table = [\"11\"]}]");
  module.addExactBodyLine(
      "  : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>");
  addSingleResultBits32Switch(
      "bit_carry_init",
      {"i32b", "i32c", "addr_shift_const", "addr_bias_const"});
  addSingleResultBits32Switch(
      "bit_carry_next",
      {"i32c", "addr_unscaled", "mac_result", "mac_result1", "int_sum",
       "selected", "running", "mac_result2", "mac_result3", "data0",
       "state_carry", "aux_masked", "aux_xor", "fp_running_aux"});
  addSingleResultBits32Switch("state_carry_cond", {"fp_gate", "i32a"});
  addSingleResultBits32Switch("state_carry_init",
                              {"i32a", "i32b", "i32c", "i32d",
                               "addr_shift_const", "addr_aux_const",
                               "addr_bias_const", "data0", "data1"});
  addSingleResultBits32Switch("state_carry_next",
                              {"mac_result", "mac_result1", "mac_result2",
                               "mac_result3", "bit_carry", "carried_scan",
                               "int_sum", "data0", "running", "aux_masked",
                               "aux_xor", "fp_running_aux"});
  module.addExactBodyLine(
      "%scan_feedback, %scan_store_value = fabric.switch [spatial] "
      "%running, %fp_running, %mac_result, %mac_result1, %mac_result2, "
      "%mac_result3, %bit_carry, %state_carry, %int_or, %selected, "
      "%int_sum, %addr_sum, %int_product, %int_product_aux, "
      "%control_muxed, %int_xor, %aux_masked, %aux_xor, %fp_running_aux");
  module.addExactBodyLine("  [{connectivity_table = [\"1111111111111111111\", "
                          "\"0011110000000000000\"]}]");
  module.addExactBodyLine(
      "  : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, "
      "!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, "
      "!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, "
      "!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, "
      "!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, "
      "!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, "
      "!fabric.bits<32>)");
  module.addExactBodyLine("  -> (!fabric.bits<32>, !fabric.bits<32>)");
  module.addExactBodyLine(
      "%sync_aux_done = fabric.switch [spatial] %store_done0, %done1, "
      "%done2, %done3, %done4, %done5, %control_token_muxed_token");
  module.addExactBodyLine("  [{connectivity_table = [\"1111111\"]}]");
  module.addExactBodyLine(
      "  : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, "
      "!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, "
      "!fabric.bits<0>) -> !fabric.bits<0>");
  return module;
}

ModuleBuilder loom::adg::buildSharedVectorAluAdg() {
  ModuleBuilder module("shared_vector_alu_adg");
  module.addInput("mgr", "memref<?x!fabric.bits<32>>")
      .addInput("idx0", "!fabric.bits<32>")
      .addInput("idx1", "!fabric.bits<32>")
      .addInput("store_idx", "!fabric.bits<32>")
      .addInput("ctrl", "!fabric.bits<0>")
      .addInput("i32a", "!fabric.bits<32>")
      .addInput("i32b", "!fabric.bits<32>");

  PeSpec xorPe;
  xorPe.inputs = {{"lhs", "bin0", "!fabric.bits<32>", ""},
                  {"rhs", "bin1", "!fabric.bits<32>", ""}};
  xorPe.resultNames = {"xored"};
  xorPe.resultTypes = {"!fabric.bits<32>"};
  xorPe.fus.push_back(
      FuSpec{{{"a", "lhs", "!fabric.bits<32>", ""},
              {"b", "rhs", "!fabric.bits<32>", ""}},
             {"!fabric.bits<32>"},
             {FabricOpSpec{{"value"},
                           {"arith.xori"},
                           {"a", "b"},
                           {"!fabric.bits<32>", "!fabric.bits<32>"},
                           {"!fabric.bits<32>"},
                           {},
                           {}}},
             {"value"}});
  module.addPe(std::move(xorPe));

  PeSpec bswapPe;
  bswapPe.inputs = {{"value", "unary", "!fabric.bits<32>", ""}};
  bswapPe.resultNames = {"swapped"};
  bswapPe.resultTypes = {"!fabric.bits<32>"};
  bswapPe.fus.push_back(FuSpec{{{"input", "value", "!fabric.bits<32>", ""}},
                               {"!fabric.bits<32>"},
                               {FabricOpSpec{{"result"},
                                             {"llvm.intr.bswap"},
                                             {"input"},
                                             {"!fabric.bits<32>"},
                                             {"!fabric.bits<32>"},
                                             {},
                                             {}}},
                               {"result"}});
  module.addPe(std::move(bswapPe));

  PeSpec floatMulPe;
  floatMulPe.inputs = {{"lhs", "bin0", "!fabric.bits<32>", ""},
                       {"rhs", "bin1", "!fabric.bits<32>", ""}};
  floatMulPe.resultNames = {"product"};
  floatMulPe.resultTypes = {"!fabric.bits<32>"};
  floatMulPe.fus.push_back(
      FuSpec{{{"a", "lhs", "!fabric.bits<32>", ""},
              {"b", "rhs", "!fabric.bits<32>", ""}},
             {"!fabric.bits<32>"},
             {FabricOpSpec{{"value"},
                           {"arith.mulf"},
                           {"a", "b"},
                           {"!fabric.bits<32>", "!fabric.bits<32>"},
                           {"!fabric.bits<32>"},
                           {},
                           {}}},
             {"value"}});
  module.addPe(std::move(floatMulPe));

  PeSpec intMulPe;
  intMulPe.inputs = {{"lhs", "bin0", "!fabric.bits<32>", ""},
                     {"rhs", "i32b", "!fabric.bits<32>", ""}};
  intMulPe.resultNames = {"int_product"};
  intMulPe.resultTypes = {"!fabric.bits<32>"};
  intMulPe.fus.push_back(
      FuSpec{{{"a", "lhs", "!fabric.bits<32>", ""},
              {"b", "rhs", "!fabric.bits<32>", ""}},
             {"!fabric.bits<32>"},
             {FabricOpSpec{{"value"},
                           {"arith.muli"},
                           {"a", "b"},
                           {"!fabric.bits<32>", "!fabric.bits<32>"},
                           {"!fabric.bits<32>"},
                           {},
                           {}}},
             {"value"}});
  module.addPe(std::move(intMulPe));

  PeSpec intAddPe;
  intAddPe.inputs = {{"lhs", "int_product", "!fabric.bits<32>", ""},
                     {"rhs", "bin1", "!fabric.bits<32>", ""}};
  intAddPe.resultNames = {"int_sum"};
  intAddPe.resultTypes = {"!fabric.bits<32>"};
  intAddPe.fus.push_back(
      FuSpec{{{"a", "lhs", "!fabric.bits<32>", ""},
              {"b", "rhs", "!fabric.bits<32>", ""}},
             {"!fabric.bits<32>"},
             {FabricOpSpec{{"value"},
                           {"arith.addi"},
                           {"a", "b"},
                           {"!fabric.bits<32>", "!fabric.bits<32>"},
                           {"!fabric.bits<32>"},
                           {},
                           {}}},
             {"value"}});
  module.addPe(std::move(intAddPe));

  PeSpec qsub16Pe;
  qsub16Pe.inputs = {{"lhs", "bin0", "!fabric.bits<32>", ""},
                     {"rhs", "bin1", "!fabric.bits<32>", ""}};
  qsub16Pe.resultNames = {"qsub16"};
  qsub16Pe.resultTypes = {"!fabric.bits<32>"};
  qsub16Pe.fus.push_back(
      FuSpec{{{"a", "lhs", "!fabric.bits<32>", ""},
              {"b", "rhs", "!fabric.bits<32>", ""}},
             {"!fabric.bits<32>"},
             {FabricOpSpec{{"value"},
                           {"llvm.arm.qsub16"},
                           {"a", "b"},
                           {"!fabric.bits<32>", "!fabric.bits<32>"},
                           {"!fabric.bits<32>"},
                           {},
                           {}}},
             {"value"}});
  module.addPe(std::move(qsub16Pe));

  PeSpec syncPe;
  syncPe.inputs = {{"pa", "sync0", "!fabric.bits<0>", ""},
                   {"pb", "sync1", "!fabric.bits<0>", ""},
                   {"pc", "sync2", "!fabric.bits<0>", ""}};
  syncPe.resultTypes = {"!fabric.bits<0>"};
  syncPe.fus.push_back(FuSpec{
      {{"fa", "pa", "!fabric.bits<0>", ""},
       {"fb", "pb", "!fabric.bits<0>", ""},
       {"fc", "pc", "!fabric.bits<0>", ""}},
      {"!fabric.bits<0>"},
      {FabricOpSpec{{"sa", "sb", "sc"},
                    {"dataflow.sync"},
                    {"fa", "fb", "fc"},
                    {"!fabric.bits<0>", "!fabric.bits<0>", "!fabric.bits<0>"},
                    {"!fabric.bits<0>", "!fabric.bits<0>", "!fabric.bits<0>"},
                    {},
                    {{"bitmask", "111"}}}},
      {"sa"}});
  module.addPe(std::move(syncPe));

  module.addExactBodyLine("%data0, %done0, %data1, %done1, %store_done =");
  module.addExactBodyLine("    fabric.mem [spatial] mgr(%mgr)");
  module.addExactBodyLine("      load(%idx0, %ctrl, %idx1, %ctrl)");
  module.addExactBodyLine("      store(%store_idx, %store_value, %ctrl)");
  module.addExactBodyLine(
      "      [{load_group_size = 2 : i32, store_group_size = 1 : i32}]");
  module.addExactBodyLine(
      "      : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, "
      "!fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, "
      "!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<0>)");
  module.addExactBodyLine(
      "      -> (!fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, "
      "!fabric.bits<0>, !fabric.bits<0>)");
  module.addExactBodyLine(
      "%bin0, %bin1, %unary = fabric.switch [spatial] %data0, %data1, "
      "%i32a");
  module.addExactBodyLine(
      "  [{connectivity_table = [\"111\", \"111\", \"111\"]}]");
  module.addExactBodyLine(
      "  : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)");
  module.addExactBodyLine(
      "  -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)");
  module.addExactBodyLine(
      "%store_value = fabric.switch [spatial] %xored, %swapped, %product, "
      "%int_product, %int_sum, %qsub16, %i32b");
  module.addExactBodyLine("  [{connectivity_table = [\"1111111\"]}]");
  module.addExactBodyLine(
      "  : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, "
      "!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, "
      "!fabric.bits<32>)");
  module.addExactBodyLine("  -> !fabric.bits<32>");
  module.addExactBodyLine(
      "%sync0, %sync1, %sync2 = fabric.switch [spatial] %done0, %done1, "
      "%store_done");
  module.addExactBodyLine(
      "  [{connectivity_table = [\"111\", \"111\", \"111\"]}]");
  module.addExactBodyLine(
      "  : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)");
  module.addExactBodyLine(
      "  -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)");
  return module;
}

ModuleBuilder loom::adg::buildSharedVectorMeshAdg() {
  ModuleBuilder module("shared_vector_mesh_adg");
  module.addInput("mgr", "memref<?x!fabric.bits<32>>")
      .addInput("idx0", "!fabric.bits<32>")
      .addInput("idx1", "!fabric.bits<32>")
      .addInput("store_idx", "!fabric.bits<32>")
      .addInput("ctrl", "!fabric.bits<0>")
      .addInput("i32a", "!fabric.bits<32>");

  PeSpec xorPe;
  xorPe.inputs = {{"lhs", "mesh_lhs", "!fabric.bits<32>", ""},
                  {"rhs", "mesh_rhs", "!fabric.bits<32>", ""}};
  xorPe.resultNames = {"xored"};
  xorPe.resultTypes = {"!fabric.bits<32>"};
  xorPe.fus.push_back(
      FuSpec{{{"a", "lhs", "!fabric.bits<32>", ""},
              {"b", "rhs", "!fabric.bits<32>", ""}},
             {"!fabric.bits<32>"},
             {FabricOpSpec{{"value"},
                           {"arith.xori"},
                           {"a", "b"},
                           {"!fabric.bits<32>", "!fabric.bits<32>"},
                           {"!fabric.bits<32>"},
                           {},
                           {}}},
             {"value"}});
  module.addPe(std::move(xorPe));

  PeSpec bswapPe;
  bswapPe.inputs = {{"value", "mesh_unary", "!fabric.bits<32>", ""}};
  bswapPe.resultNames = {"swapped"};
  bswapPe.resultTypes = {"!fabric.bits<32>"};
  bswapPe.fus.push_back(FuSpec{{{"input", "value", "!fabric.bits<32>", ""}},
                               {"!fabric.bits<32>"},
                               {FabricOpSpec{{"result"},
                                             {"llvm.intr.bswap"},
                                             {"input"},
                                             {"!fabric.bits<32>"},
                                             {"!fabric.bits<32>"},
                                             {},
                                             {}}},
                               {"result"}});
  module.addPe(std::move(bswapPe));

  PeSpec syncPe;
  syncPe.inputs = {{"pa", "sync0", "!fabric.bits<0>", ""},
                   {"pb", "sync1", "!fabric.bits<0>", ""},
                   {"pc", "sync2", "!fabric.bits<0>", ""}};
  syncPe.resultTypes = {"!fabric.bits<0>"};
  syncPe.fus.push_back(FuSpec{
      {{"fa", "pa", "!fabric.bits<0>", ""},
       {"fb", "pb", "!fabric.bits<0>", ""},
       {"fc", "pc", "!fabric.bits<0>", ""}},
      {"!fabric.bits<0>"},
      {FabricOpSpec{{"sa", "sb", "sc"},
                    {"dataflow.sync"},
                    {"fa", "fb", "fc"},
                    {"!fabric.bits<0>", "!fabric.bits<0>", "!fabric.bits<0>"},
                    {"!fabric.bits<0>", "!fabric.bits<0>", "!fabric.bits<0>"},
                    {},
                    {{"bitmask", "111"}}}},
      {"sa"}});
  module.addPe(std::move(syncPe));

  module.addExactBodyLine("%data0, %done0, %data1, %done1, %store_done =");
  module.addExactBodyLine("    fabric.mem [spatial] mgr(%mgr)");
  module.addExactBodyLine("      load(%idx0, %ctrl, %idx1, %ctrl)");
  module.addExactBodyLine("      store(%store_idx, %store_value, %ctrl)");
  module.addExactBodyLine(
      "      [{load_group_size = 2 : i32, store_group_size = 1 : i32}]");
  module.addExactBodyLine(
      "      : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, "
      "!fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, "
      "!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<0>)");
  module.addExactBodyLine(
      "      -> (!fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, "
      "!fabric.bits<0>, !fabric.bits<0>)");
  module.addExactBodyLine(
      "%west0, %west1, %west_unary = fabric.switch [spatial] %data0, "
      "%data1, %i32a");
  module.addExactBodyLine(
      "  [{connectivity_table = [\"111\", \"111\", \"101\"]}]");
  module.addExactBodyLine(
      "  : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)");
  module.addExactBodyLine(
      "  -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)");
  module.addExactBodyLine(
      "%mesh_lhs, %mesh_rhs_pre = fabric.switch [spatial] %west0, %west1");
  module.addExactBodyLine("  [{connectivity_table = [\"11\", \"11\"]}]");
  module.addExactBodyLine("  : (!fabric.bits<32>, !fabric.bits<32>)");
  module.addExactBodyLine("  -> (!fabric.bits<32>, !fabric.bits<32>)");
  module.addExactBodyLine(
      "%mesh_rhs, %mesh_unary = fabric.switch [spatial] %mesh_rhs_pre, "
      "%west0, %west_unary");
  module.addExactBodyLine("  [{connectivity_table = [\"111\", \"011\"]}]");
  module.addExactBodyLine(
      "  : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)");
  module.addExactBodyLine("  -> (!fabric.bits<32>, !fabric.bits<32>)");
  module.addExactBodyLine(
      "%store_value = fabric.switch [spatial] %xored, %swapped, %i32a");
  module.addExactBodyLine("  [{connectivity_table = [\"111\"]}]");
  module.addExactBodyLine(
      "  : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)");
  module.addExactBodyLine("  -> !fabric.bits<32>");
  module.addExactBodyLine(
      "%sync0, %sync1, %sync2 = fabric.switch [spatial] %done0, %done1, "
      "%store_done");
  module.addExactBodyLine(
      "  [{connectivity_table = [\"111\", \"111\", \"111\"]}]");
  module.addExactBodyLine(
      "  : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)");
  module.addExactBodyLine(
      "  -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)");
  return module;
}

ModuleBuilder loom::adg::buildFullSpatialCoreAdg() {
  ModuleBuilder module("full_spatialcore_adg");
  module.addInput("mgr", "memref<?x!fabric.bits<32>>")
      .addInput("lhs", "!fabric.bits<32>")
      .addInput("rhs", "!fabric.bits<32>")
      .addInput("addr", "!fabric.bits<32>")
      .addInput("ctrl", "!fabric.bits<0>")
      .addInput("tag", "!fabric.bits<4>")
      .addInput("lhs_t", "!fabric.bits_tag<32, 4>")
      .addInput("rhs_t", "!fabric.bits_tag<32, 4>")
      .addInput("addr_t", "!fabric.bits_tag<32, 4>")
      .addInput("ctrl_t", "!fabric.bits_tag<0, 4>");

  module.addPe(makeMinimalAddPe(Schedule::Spatial, "!fabric.bits<32>",
                                "!fabric.bits<32>"));

  TemporalPeConfig temporal;
  temporal.tagWidth = 4;
  temporal.numInstruction = 2;
  temporal.fuConfigMode = "per_fu_config";
  temporal.operandBufferMode = "per_input_port";
  temporal.operandBufferSize = 2;
  temporal.numRegFifo = 2;
  temporal.regFifoDepth = 4;
  temporal.regFifoPorts = 1;
  module.addPe(makeMinimalAddPe(Schedule::Temporal, "lhs_t", "rhs_t",
                                "!fabric.bits_tag<32, 4>", "!fabric.bits<32>",
                                std::move(temporal)));

  module.addSwitch(SwitchSpec{Schedule::Spatial,
                              {"lhs", "rhs"},
                              {"!fabric.bits<32>", "!fabric.bits<32>"},
                              {"11", "11"},
                              0});
  module.addSwitch(
      SwitchSpec{Schedule::Temporal,
                 {"lhs_t", "rhs_t"},
                 {"!fabric.bits_tag<32, 4>", "!fabric.bits_tag<32, 4>"},
                 {"11", "11"},
                 2});

  module.addMem(MemSpec{
      Schedule::Spatial, "mgr", {{"addr", "ctrl"}}, {{"addr", "lhs", "ctrl"}}});

  MemSpec temporalMem;
  temporalMem.schedule = Schedule::Temporal;
  temporalMem.manager = "mgr";
  temporalMem.loads = {{"addr_t", "ctrl_t"}};
  temporalMem.stores = {{"addr_t", "lhs_t", "ctrl_t"}};
  temporalMem.temporalTagWidth = 4;
  temporalMem.temporalAddrTableSize = 2;
  module.addMem(std::move(temporalMem));

  module.addExactBodyLine(
      "%tagged = fabric.boundary [s2t] %lhs, %tag : (!fabric.bits<32>, "
      "!fabric.bits<4>) -> !fabric.bits_tag<32, 4>");
  module.addExactBodyLine(
      "%queued = fabric.fifo %tagged [max_depth = 4, bypassable = true] : "
      "!fabric.bits_tag<32, 4>");
  module.addExactBodyLine(
      "fabric.pe @ALU [spatial] (!fabric.bits<32>) -> (!fabric.bits<32>) {");
  module.addExactBodyLine("^bb0(%pa: !fabric.bits<32>):");
  module.addExactBodyLine(
      "  fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {");
  module.addExactBodyLine(
      "    %v = fabric.op [@arith.addi] (%fa, %fa) : (!fabric.bits<32>, "
      "!fabric.bits<32>) -> !fabric.bits<32>");
  module.addExactBodyLine("    fabric.yield %v : !fabric.bits<32>");
  module.addExactBodyLine("  }");
  module.addExactBodyLine("  fabric.yield %pa : !fabric.bits<32>");
  module.addExactBodyLine("}");
  module.addExactBodyLine(
      "%inst = fabric.instantiate @ALU(%lhs : !fabric.bits<32>) -> "
      "(!fabric.bits<32>)");
  return module;
}

SystemBuilder loom::adg::buildHeterogeneousSocAdg() {
  SystemBuilder system("heterogeneous_dual_accel_soc", "sequential");
  system.addHostCore("host0", "rv64gc", axiManagerPort("mem"));
  system.addSpatialAccelerator("acc0", "shared_reduction_adg", "rv32im",
                               axiManagerPort("mem"));
  system.addFixedAccelerator("fft0", "fft", axiManagerPort("mem"));

  std::vector<std::string> cachePorts;
  appendPorts(cachePorts, axiSubordinatePort("host"));
  appendPorts(cachePorts, axiManagerPort("mem"));
  system.addCache("l1d0", 64, 32 * 1024, std::move(cachePorts));

  std::vector<std::string> dmaPorts;
  appendPorts(dmaPorts, axiSubordinatePort("ctrl"));
  appendPorts(dmaPorts, axiManagerPort("mem"));
  system.addDmaEngine("dma0", 4, std::move(dmaPorts));

  std::vector<std::string> dramPorts;
  appendPorts(dramPorts, axiSubordinatePort("cache"));
  appendPorts(dramPorts, axiSubordinatePort("acc0"));
  appendPorts(dramPorts, axiSubordinatePort("fft0"));
  appendPorts(dramPorts, axiSubordinatePort("dma0"));
  system.addMemory("dram0", 1024 * 1024, std::move(dramPorts));

  connectAxiMemoryPort(system, "host0", "mem", "l1d0", "host");
  connectAxiMemoryPort(system, "l1d0", "mem", "dram0", "cache");
  connectAxiMemoryPort(system, "acc0", "mem", "dram0", "acc0");
  connectAxiMemoryPort(system, "fft0", "mem", "dram0", "fft0");
  connectAxiMemoryPort(system, "dma0", "mem", "dram0", "dma0");
  return system;
}

llvm::Error loom::adg::writeMinimalSpatialAdg(llvm::raw_ostream &os) {
  return buildMinimalSpatialAdg().print(os);
}

llvm::Error loom::adg::writeMinimalTemporalAdg(llvm::raw_ostream &os) {
  return buildMinimalTemporalAdg().print(os);
}

llvm::Error loom::adg::writeSharedReductionAdg(llvm::raw_ostream &os) {
  return buildSharedReductionAdg().print(os);
}

llvm::Error loom::adg::writeSharedVectorAluAdg(llvm::raw_ostream &os) {
  return buildSharedVectorAluAdg().print(os);
}

llvm::Error loom::adg::writeSharedVectorMeshAdg(llvm::raw_ostream &os) {
  return buildSharedVectorMeshAdg().print(os);
}

llvm::Error loom::adg::writeFullSpatialCoreAdg(llvm::raw_ostream &os) {
  return buildFullSpatialCoreAdg().print(os);
}

llvm::Error loom::adg::writeHeterogeneousSocAdg(llvm::raw_ostream &os) {
  if (llvm::Error err = buildSharedReductionAdg().print(os))
    return err;
  os << '\n';
  return buildHeterogeneousSocAdg().print(os);
}

llvm::Error loom::adg::writeSpatialTopologyMatrixAdg(llvm::raw_ostream &os,
                                                     llvm::StringRef family) {
  if (family == "chain-1d")
    return buildChain1DAdg().print(os);
  if (family == "mesh-2d")
    return buildMesh2DAdg().print(os);
  if (family == "torus-edge")
    return buildTorusEdgeAdg().print(os);
  if (family == "systolic-array")
    return buildSystolicArrayAdg().print(os);
  if (family == "clustered-array")
    return buildClusteredArrayAdg().print(os);
  if (family == "reduction-tree")
    return buildReductionTreeAdg().print(os);
  if (family == "cross-coupled-switch")
    return buildCrossCoupledSwitchAdg().print(os);
  if (family == "sparse-long-link")
    return buildSparseLongLinkAdg().print(os);
  if (family == "heterogeneous-islands")
    return buildHeterogeneousIslandsAdg().print(os);
  return llvm::createStringError(std::errc::invalid_argument,
                                 "unknown topology matrix case %s",
                                 family.str().c_str());
}
