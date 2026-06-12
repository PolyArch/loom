#include "ADG/Builder.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

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
    for (std::size_t i = 0; i < fu.yieldValues.size(); ++i) {
      if (i)
        os << ", ";
      os << valueName(fu.yieldValues[i]);
    }
    os << " : ";
    printTypeSequence(os, fu.resultTypes);
  }
  os << "\n";
  os << "    }\n";
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
  os << "  fabric.pe [" << scheduleName(pe.schedule) << "] (";
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

SystemBuilder &SystemBuilder::addHostCore(std::string nodeName,
                                          std::string scalar,
                                          std::vector<std::string> ports) {
  nodes.push_back(SystemNodeSpec{std::move(nodeName), "host_core",
                                 std::move(ports), "", std::move(scalar), "",
                                 std::nullopt, {}});
  return *this;
}

SystemBuilder &SystemBuilder::addSpatialAccelerator(
    std::string nodeName, std::string spatialModule, std::string scalar,
    std::vector<std::string> ports) {
  nodes.push_back(SystemNodeSpec{std::move(nodeName), "acc_core",
                                 std::move(ports), std::move(spatialModule),
                                 std::move(scalar), "", std::nullopt, {}});
  return *this;
}

SystemBuilder &
SystemBuilder::addFixedAccelerator(std::string nodeName, std::string function,
                                   std::vector<std::string> ports) {
  nodes.push_back(SystemNodeSpec{std::move(nodeName), "fixed_accelerator",
                                 std::move(ports), "", "", std::move(function),
                                 std::nullopt, {}});
  return *this;
}

SystemBuilder &SystemBuilder::addCache(std::string nodeName,
                                       std::uint64_t lineBytes,
                                       std::uint64_t capacityBytes,
                                       std::vector<std::string> ports) {
  nodes.push_back(SystemNodeSpec{
      std::move(nodeName), "cache", std::move(ports), "", "", "",
      std::nullopt,
      {{"capacity_bytes", capacityBytes}, {"line_bytes", lineBytes}}});
  return *this;
}

SystemBuilder &SystemBuilder::addMemory(std::string nodeName,
                                        std::uint64_t bytes,
                                        std::vector<std::string> ports) {
  nodes.push_back(SystemNodeSpec{std::move(nodeName), "memory",
                                 std::move(ports), "", "", "", bytes, {}});
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
  streamPe.inputs = {{"pa", "i64a", "!fabric.bits<64>", ""},
                     {"pb", "i64b", "!fabric.bits<64>", ""},
                     {"pc", "i64c", "!fabric.bits<64>", ""}};
  streamPe.resultTypes = {"!fabric.bits<64>"};
  FuSpec streamFu;
  streamFu.inputs = {{"fa", "pa", "!fabric.bits<64>", ""},
                     {"fb", "pb", "!fabric.bits<64>", ""},
                     {"fc", "pc", "!fabric.bits<64>", ""}};
  streamFu.operations.push_back(
      FabricOpSpec{{"idx", "rwc"},
                   {"dataflow.stream"},
                   {"fa", "fb", "fc"},
                   {"!fabric.bits<64>", "!fabric.bits<64>", "!fabric.bits<64>"},
                   {"!fabric.bits<64>", "!fabric.bits<1>"},
                   {{"cont_cond", {"<"}}, {"step_op", {"+="}}},
                   {{"cont_cond", "<"}, {"step_op", "+="}}});
  streamPe.fus.push_back(std::move(streamFu));
  module.addPe(std::move(streamPe));

  PeSpec reductionPe;
  reductionPe.inputs = {{"pa", "i32a", "!fabric.bits<32>", ""},
                        {"pb", "i32b", "!fabric.bits<32>", ""},
                        {"pc", "i32c", "!fabric.bits<32>", ""}};
  reductionPe.resultTypes = {"!fabric.bits<32>"};
  auto makeCarryFu = []() {
    return FuSpec{{{"cond", "pa", "!fabric.bits<32>", "!fabric.bits<1>"},
                   {"init", "pb", "!fabric.bits<32>", ""},
                   {"next", "pc", "!fabric.bits<32>", ""}},
                  {},
                  {FabricOpSpec{{"carried"},
                                {"dataflow.carry"},
                                {"cond", "init", "next"},
                                {"!fabric.bits<1>", "!fabric.bits<32>",
                                 "!fabric.bits<32>"},
                                {"!fabric.bits<32>"},
                                {},
                                {}}},
                  {}};
  };
  auto makeInvariantFu = []() {
    return FuSpec{{{"cond", "pa", "!fabric.bits<32>", "!fabric.bits<1>"},
                   {"value", "pb", "!fabric.bits<32>", ""}},
                  {},
                  {FabricOpSpec{{"stable"},
                                {"dataflow.invariant"},
                                {"cond", "value"},
                                {"!fabric.bits<1>", "!fabric.bits<32>"},
                                {"!fabric.bits<32>"},
                                {},
                                {}}},
                  {}};
  };
  auto makeBinary32Fu = [](std::string resultName, std::string opName) {
    return FuSpec{{{"lhs", "pa", "!fabric.bits<32>", ""},
                   {"rhs", "pb", "!fabric.bits<32>", ""}},
                  {},
                  {FabricOpSpec{{std::move(resultName)},
                                {std::move(opName)},
                                {"lhs", "rhs"},
                                {"!fabric.bits<32>", "!fabric.bits<32>"},
                                {"!fabric.bits<32>"},
                                {},
                                {}}},
                  {}};
  };
  reductionPe.fus.push_back(makeCarryFu());
  reductionPe.fus.push_back(makeCarryFu());
  reductionPe.fus.push_back(makeInvariantFu());
  reductionPe.fus.push_back(makeInvariantFu());
  reductionPe.fus.push_back(makeBinary32Fu("sum", "arith.addi"));
  reductionPe.fus.push_back(makeBinary32Fu("diff", "arith.subf"));
  reductionPe.fus.push_back(makeBinary32Fu("sum", "arith.addf"));
  reductionPe.fus.push_back(makeBinary32Fu("product", "arith.mulf"));
  reductionPe.fus.push_back(makeBinary32Fu("shifted", "arith.shrui"));
  reductionPe.fus.push_back(makeBinary32Fu("shifted", "arith.shli"));
  reductionPe.fus.push_back(makeBinary32Fu("masked", "arith.andi"));
  reductionPe.fus.push_back(makeBinary32Fu("combined", "arith.ori"));
  reductionPe.fus.push_back(makeBinary32Fu("combined", "arith.xori"));
  module.addPe(std::move(reductionPe));

  PeSpec syncPe;
  syncPe.inputs = {{"pc", "ctrl", "!fabric.bits<0>", ""}};
  syncPe.resultTypes = {"!fabric.bits<0>"};
  syncPe.fus.push_back(FuSpec{{{"fc", "pc", "!fabric.bits<0>", ""}},
                              {},
                              {FabricOpSpec{{"done"},
                                            {"dataflow.sync"},
                                            {"fc"},
                                            {"!fabric.bits<0>"},
                                            {"!fabric.bits<0>"},
                                            {},
                                            {{"bitmask", "1"}}}},
                              {}});
  module.addPe(std::move(syncPe));

  module.addMem(MemSpec{
      Schedule::Spatial,
      "mgr",
      {{"i32a", "ctrl"}, {"i32b", "ctrl"}, {"i32c", "ctrl"}, {"i32d", "ctrl"}},
      {{"i32a", "i32b", "ctrl"}, {"i32c", "i32d", "ctrl"}}});
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
                                "!fabric.bits_tag<32, 4>",
                                "!fabric.bits<32>", std::move(temporal)));

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
      Schedule::Spatial,
      "mgr",
      {{"addr", "ctrl"}},
      {{"addr", "lhs", "ctrl"}}});

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

  std::vector<std::string> dramPorts;
  appendPorts(dramPorts, axiSubordinatePort("cache"));
  appendPorts(dramPorts, axiSubordinatePort("acc0"));
  appendPorts(dramPorts, axiSubordinatePort("fft0"));
  system.addMemory("dram0", 1024 * 1024, std::move(dramPorts));

  connectAxiMemoryPort(system, "host0", "mem", "l1d0", "host");
  connectAxiMemoryPort(system, "l1d0", "mem", "dram0", "cache");
  connectAxiMemoryPort(system, "acc0", "mem", "dram0", "acc0");
  connectAxiMemoryPort(system, "fft0", "mem", "dram0", "fft0");
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

llvm::Error loom::adg::writeFullSpatialCoreAdg(llvm::raw_ostream &os) {
  return buildFullSpatialCoreAdg().print(os);
}

llvm::Error loom::adg::writeHeterogeneousSocAdg(llvm::raw_ostream &os) {
  if (llvm::Error err = buildSharedReductionAdg().print(os))
    return err;
  os << '\n';
  return buildHeterogeneousSocAdg().print(os);
}
