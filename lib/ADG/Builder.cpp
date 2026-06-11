#include "ADG/Builder.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

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
    os << valueName(binding.localName) << " = "
       << valueName(binding.sourceName) << " : " << binding.type;
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

void printHwParams(llvm::raw_ostream &os,
                   const std::map<std::string, std::vector<std::string>>
                       &hwParams) {
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
    os << "  ";
    for (std::size_t i = 0; i < mem.loads.size(); ++i) {
      if (i)
        os << ", ";
      os << "%mem" << memIndex << "_data" << i << ", %mem" << memIndex
         << "_done" << i;
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
    os << "\n        [{load_group_size = "
       << static_cast<unsigned>(mem.loads.size())
       << " : i32, store_group_size = " << mem.storePorts << " : i32";
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
    llvm::SmallVector<std::string> resultTypes;
    for (const MemLoadPort &load : mem.loads) {
      resultTypes.push_back(inputTypes.lookup(load.address));
      resultTypes.push_back(inputTypes.lookup(load.control));
    }
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

ModuleBuilder loom::adg::buildMinimalSpatialAdg() {
  ModuleBuilder module("minimal_spatial_adg");
  module.addInput("mgr", "memref<?x!fabric.bits<32>>")
      .addInput("lhs", "!fabric.bits<32>")
      .addInput("rhs", "!fabric.bits<32>")
      .addInput("addr", "!fabric.bits<32>")
      .addInput("ctrl", "!fabric.bits<0>");

  PeSpec aluPe;
  aluPe.inputs = {{"pa", "lhs", "!fabric.bits<32>", ""},
                  {"pb", "rhs", "!fabric.bits<32>", ""}};
  aluPe.resultTypes = {"!fabric.bits<32>"};
  FuSpec addFu;
  addFu.inputs = {{"fa", "pa", "!fabric.bits<32>", ""},
                  {"fb", "pb", "!fabric.bits<32>", ""}};
  addFu.resultTypes = {"!fabric.bits<32>"};
  addFu.operations.push_back(
      FabricOpSpec{{"sum"},
                   {"arith.addi"},
                   {"fa", "fb"},
                   {"!fabric.bits<32>", "!fabric.bits<32>"},
                   {"!fabric.bits<32>"},
                   {},
                   {}});
  addFu.yieldValues = {"sum"};
  aluPe.fus.push_back(std::move(addFu));
  module.addPe(std::move(aluPe));

  module.addSwitch(SwitchSpec{Schedule::Spatial,
                              {"lhs", "rhs"},
                              {"!fabric.bits<32>", "!fabric.bits<32>"},
                              {"11", "11"},
                              0});
  module.addMem(MemSpec{Schedule::Spatial, "mgr", {{"addr", "ctrl"}}, 0});
  return module;
}

ModuleBuilder loom::adg::buildMinimalTemporalAdg() {
  ModuleBuilder module("minimal_temporal_adg");
  module.addInput("mgr", "memref<?x!fabric.bits<32>>")
      .addInput("lhs", "!fabric.bits_tag<32, 4>")
      .addInput("rhs", "!fabric.bits_tag<32, 4>")
      .addInput("addr", "!fabric.bits_tag<32, 4>")
      .addInput("ctrl", "!fabric.bits_tag<0, 4>");

  PeSpec aluPe;
  aluPe.schedule = Schedule::Temporal;
  aluPe.inputs = {{"pa", "lhs", "!fabric.bits_tag<32, 4>", ""},
                  {"pb", "rhs", "!fabric.bits_tag<32, 4>", ""}};
  aluPe.resultTypes = {"!fabric.bits_tag<32, 4>"};
  aluPe.temporal.tagWidth = 4;
  aluPe.temporal.numInstruction = 1;
  aluPe.temporal.fuConfigMode = "per_fu_config";
  aluPe.temporal.operandBufferMode = "per_instruction";

  FuSpec addFu;
  addFu.inputs = {{"fa", "pa", "!fabric.bits<32>", ""},
                  {"fb", "pb", "!fabric.bits<32>", ""}};
  addFu.resultTypes = {"!fabric.bits<32>"};
  addFu.operations.push_back(
      FabricOpSpec{{"sum"},
                   {"arith.addi"},
                   {"fa", "fb"},
                   {"!fabric.bits<32>", "!fabric.bits<32>"},
                   {"!fabric.bits<32>"},
                   {},
                   {}});
  addFu.yieldValues = {"sum"};
  aluPe.fus.push_back(std::move(addFu));
  module.addPe(std::move(aluPe));

  module.addSwitch(SwitchSpec{Schedule::Temporal,
                              {"lhs", "rhs"},
                              {"!fabric.bits_tag<32, 4>",
                               "!fabric.bits_tag<32, 4>"},
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
  streamFu.operations.push_back(FabricOpSpec{
      {"idx", "rwc"},
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
    return FuSpec{
        {{"cond", "pa", "!fabric.bits<32>", "!fabric.bits<1>"},
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
    return FuSpec{
        {{"cond", "pa", "!fabric.bits<32>", "!fabric.bits<1>"},
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
    return FuSpec{
        {{"lhs", "pa", "!fabric.bits<32>", ""},
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
  module.addPe(std::move(reductionPe));

  PeSpec syncPe;
  syncPe.inputs = {{"pc", "ctrl", "!fabric.bits<0>", ""}};
  syncPe.resultTypes = {"!fabric.bits<0>"};
  syncPe.fus.push_back(FuSpec{
      {{"fc", "pc", "!fabric.bits<0>", ""}},
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

  module.addMem(MemSpec{Schedule::Spatial,
                        "mgr",
                        {{"i32a", "ctrl"},
                         {"i32b", "ctrl"},
                         {"i32c", "ctrl"},
                         {"i32d", "ctrl"}},
                        0});
  return module;
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
