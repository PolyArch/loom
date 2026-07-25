#include "BuilderInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <iterator>

using namespace loom::adg;
using namespace loom::adg::detail;

namespace loom::adg::detail {

PeSpec makeMinimalAddPe(Schedule schedule, std::string lhsSource,
                        std::string rhsSource, std::string boundaryType,
                        std::string fuType, TemporalPeConfig temporal) {
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
                        std::string fuType, TemporalPeConfig temporal) {
  return makeMinimalAddPe(schedule, "lhs", "rhs", std::move(boundaryType),
                          std::move(fuType), std::move(temporal));
}

static std::string visualLayoutAttr(llvm::ArrayRef<VisualPoint> points) {
  std::string text = "[";
  for (const VisualPoint &point : points) {
    if (text.size() > 1)
      text += ", ";
    text += "{node = \"";
    text += point.node.str();
    text += "\", x = ";
    text += std::to_string(point.x);
    text += " : i32, y = ";
    text += std::to_string(point.y);
    text += " : i32}";
  }
  text += "]";
  return text;
}

void addVisualLayout(ModuleBuilder &module,
                     llvm::ArrayRef<VisualPoint> points) {
  module.addAttribute("coordinates_semantic", "false");
  module.addAttribute("visual_layout", visualLayoutAttr(points));
}

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

ModuleBuilder
makeTopologyMatrixModule(llvm::StringRef name, bool includeTemporal,
                         llvm::ArrayRef<VisualPoint> visualPoints) {
  ModuleBuilder module(name.str());
  if (!visualPoints.empty())
    addVisualLayout(module, visualPoints);
  module.addInput("mgr", "memref<?x!fabric.bits<32>>")
      .addInput("a", "!fabric.bits<32>")
      .addInput("b", "!fabric.bits<32>")
      .addInput("c", "!fabric.bits<32>")
      .addInput("d", "!fabric.bits<32>")
      .addInput("addr", "!fabric.bits<32>")
      .addInput("ctrl", "!fabric.bits<0>");
  if (includeTemporal) {
    module.addInput("lhs_t", "!fabric.bits_tag<32, 4>")
        .addInput("rhs_t", "!fabric.bits_tag<32, 4>")
        .addInput("tag", "!fabric.bits<4>");
  }
  return module;
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

std::vector<std::string> stringList(llvm::ArrayRef<llvm::StringRef> values) {
  std::vector<std::string> result;
  result.reserve(values.size());
  for (llvm::StringRef value : values)
    result.push_back(value.str());
  return result;
}

BodyLineSpec exactBodyLine(std::string text) {
  return BodyLineSpec{{std::move(text)}, {}};
}

BodyLineSpec nestedBodyLine(std::string text) {
  return BodyLineSpec{{std::move(text)}, {}, false};
}

BodyLineSpec directBodyLine(std::vector<std::string> fragments,
                            std::vector<std::string> operands) {
  return BodyLineSpec{std::move(fragments), std::move(operands)};
}

ModuleBuilder &appendBodyOp(ModuleBuilder &module, BodyOpSpec op) {
  return ModuleBuilderInternals::addBodyOp(module, std::move(op));
}

std::vector<BodyResultSpec>
uniformBodyResults(llvm::ArrayRef<std::string> names, llvm::StringRef type) {
  std::vector<BodyResultSpec> results;
  results.reserve(names.size());
  for (const std::string &name : names)
    results.push_back(BodyResultSpec{name, type.str()});
  return results;
}

std::string bodyResultTypes(llvm::ArrayRef<BodyResultSpec> results) {
  if (results.size() == 1)
    return results.front().type;
  std::string text = "(";
  for (auto [index, result] : llvm::enumerate(results)) {
    if (index)
      text += ", ";
    text += result.type;
  }
  text += ")";
  return text;
}

BodyOpSpec bodyOpWithResultLine(std::vector<BodyResultSpec> results,
                                std::vector<BodyLineSpec> lines,
                                llvm::StringRef prefix,
                                llvm::StringRef suffix) {
  lines.push_back(
      exactBodyLine((prefix + bodyResultTypes(results) + suffix).str()));
  return BodyOpSpec{std::move(results), std::move(lines)};
}

BodyLineSpec bodyResultTypeLine(llvm::ArrayRef<BodyResultSpec> results,
                                llvm::StringRef prefix,
                                llvm::StringRef suffix = "") {
  return exactBodyLine((prefix + bodyResultTypes(results) + suffix).str());
}

BodyLineSpec directOperandListLine(std::string prefix,
                                   llvm::ArrayRef<std::string> operands,
                                   std::string suffix,
                                   llvm::StringRef separator) {
  if (operands.empty())
    return exactBodyLine(std::move(prefix) + std::move(suffix));
  std::vector<std::string> fragments;
  fragments.reserve(operands.size() + 1);
  fragments.push_back(std::move(prefix));
  for (std::size_t i = 1; i < operands.size(); ++i)
    fragments.push_back(separator.str());
  fragments.push_back(std::move(suffix));
  return directBodyLine(
      std::move(fragments),
      std::vector<std::string>(operands.begin(), operands.end()));
}

BodyLineSpec directOperandListLine(std::string prefix,
                                   llvm::ArrayRef<llvm::StringRef> operands,
                                   std::string suffix,
                                   llvm::StringRef separator) {
  std::vector<std::string> operandNames;
  operandNames.reserve(operands.size());
  for (llvm::StringRef operand : operands)
    operandNames.push_back(operand.str());
  return directOperandListLine(std::move(prefix), operandNames,
                               std::move(suffix), separator);
}

BodyLineSpec
directOperandListLine(std::string prefix,
                      std::initializer_list<llvm::StringRef> operands,
                      std::string suffix, llvm::StringRef separator) {
  return directOperandListLine(
      std::move(prefix),
      llvm::ArrayRef<llvm::StringRef>(operands.begin(), operands.size()),
      std::move(suffix), separator);
}

BodyLineSpec directHeadAndListLine(std::string prefix, std::string head,
                                   std::string infix,
                                   llvm::ArrayRef<std::string> operands,
                                   std::string suffix) {
  if (operands.empty())
    return directBodyLine({std::move(prefix), std::move(infix) + suffix},
                          {std::move(head)});
  std::vector<std::string> fragments;
  fragments.reserve(operands.size() + 2);
  fragments.push_back(std::move(prefix));
  fragments.push_back(std::move(infix));
  for (std::size_t i = 1; i < operands.size(); ++i)
    fragments.push_back(", ");
  fragments.push_back(std::move(suffix));

  std::vector<std::string> allOperands;
  allOperands.reserve(operands.size() + 1);
  allOperands.push_back(std::move(head));
  allOperands.insert(allOperands.end(), operands.begin(), operands.end());
  return directBodyLine(std::move(fragments), std::move(allOperands));
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

std::string uniformConnectivityRows(std::size_t rowCount,
                                    std::size_t inputCount) {
  std::string row(inputCount, '1');
  std::string text = "[{connectivity_table = [";
  for (std::size_t index = 0; index < rowCount; ++index) {
    if (index)
      text += ", ";
    text += "\"";
    text += row;
    text += "\"";
  }
  text += "]}]";
  return text;
}

void addUniformSwitch(ModuleBuilder &module,
                      llvm::ArrayRef<std::string> results,
                      llvm::ArrayRef<std::string> inputs,
                      llvm::StringRef type) {
  appendBodyOp(
      module,
      bodyOpWithResultLine(
          uniformBodyResults(results, type),
          {directOperandListLine("fabric.switch [spatial] ", inputs),
           exactBodyLine("      " + uniformConnectivityRows(results.size(),
                                                            inputs.size())),
           exactBodyLine("      : " + uniformTypeList(inputs.size(), type))},
          "      -> "));
}

void addUniformSwitch(ModuleBuilder &module,
                      std::initializer_list<llvm::StringRef> results,
                      std::initializer_list<llvm::StringRef> inputs,
                      llvm::StringRef type) {
  std::vector<std::string> resultNames;
  resultNames.reserve(results.size());
  for (llvm::StringRef result : results)
    resultNames.push_back(result.str());
  std::vector<std::string> inputNames;
  inputNames.reserve(inputs.size());
  for (llvm::StringRef input : inputs)
    inputNames.push_back(input.str());
  addUniformSwitch(module, resultNames, inputNames, type);
}

void addUniformSwitch(ModuleBuilder &module,
                      std::initializer_list<llvm::StringRef> results,
                      llvm::ArrayRef<std::string> inputs,
                      llvm::StringRef type) {
  std::vector<std::string> resultNames;
  resultNames.reserve(results.size());
  for (llvm::StringRef result : results)
    resultNames.push_back(result.str());
  addUniformSwitch(module, resultNames, inputs, type);
}

std::string singleManagerDispatchEligibility(unsigned requestSourceCount) {
  std::string text = ", dispatch_eligibility = {operation_port_requests = [";
  for (unsigned source = 0; source < requestSourceCount; ++source) {
    if (source)
      text += ", ";
    text += "[0 : i32]";
  }
  text += "], subordinate_requests = []}";
  return text;
}

void addSpatialMemLoad(ModuleBuilder &module) {
  appendBodyOp(
      module,
      bodyOpWithResultLine(
          {BodyResultSpec{"data", "!fabric.bits<32>"},
           BodyResultSpec{"done", "!fabric.bits<0>"}},
          {directBodyLine({"fabric.mem [spatial] mgr(", ") load(", ", ", ")"},
                          {"mgr", "addr", "ctrl"}),
           exactBodyLine(
               "      [{load_group_size = 1 : i32, store_group_size = 0 : "
               "i32, data_width = 32 : i32" +
               singleManagerDispatchEligibility(1) + "}]"),
           exactBodyLine(
               "      : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, "
               "!fabric.bits<0>)")},
          "      -> "));
}

void addSpatialSwitch(ModuleBuilder &module,
                      llvm::ArrayRef<llvm::StringRef> results,
                      llvm::ArrayRef<llvm::StringRef> inputs,
                      llvm::ArrayRef<llvm::StringRef> rows) {
  std::vector<std::string> resultNames;
  resultNames.reserve(results.size());
  for (llvm::StringRef result : results)
    resultNames.push_back(result.str());
  appendBodyOp(module,
               bodyOpWithResultLine(
                   uniformBodyResults(resultNames, "!fabric.bits<32>"),
                   {directOperandListLine("fabric.switch [spatial] ", inputs),
                    exactBodyLine("      " + switchConnectivity(rows)),
                    exactBodyLine("      : " + bits32TypeList(inputs.size()))},
                   "      -> "));
}

void addSpatialAddPe(ModuleBuilder &module, llvm::StringRef result,
                     llvm::StringRef lhs, llvm::StringRef rhs,
                     llvm::StringRef opName) {
  std::vector<BodyResultSpec> results = {
      BodyResultSpec{result.str(), "!fabric.bits<32>"}};
  appendBodyOp(
      module,
      BodyOpSpec{
          results,
          {directBodyLine(
               {"fabric.pe [spatial] (%lhs = ", " : !fabric.bits<32>,"},
               {lhs.str()}),
           directBodyLine(
               {"                         %rhs = ", " : !fabric.bits<32>)"},
               {rhs.str()}),
           bodyResultTypeLine(results, "        -> ", " {"),
           nestedBodyLine("      fabric.fu(%fu_lhs = %lhs : !fabric.bits<32>,"),
           nestedBodyLine(
               "                %fu_rhs = %rhs : !fabric.bits<32>) -> "
               "!fabric.bits<32> {"),
           nestedBodyLine("        %value = fabric.op [@" + opName.str() +
                          "] (%fu_lhs, %fu_rhs)"),
           nestedBodyLine(
               "                 : (!fabric.bits<32>, !fabric.bits<32>) -> "
               "!fabric.bits<32>"),
           nestedBodyLine("        fabric.yield %value : !fabric.bits<32>"),
           nestedBodyLine("      }"), nestedBodyLine("    }")}});
}

void addUnaryPe(ModuleBuilder &module, llvm::StringRef result,
                llvm::StringRef input, llvm::StringRef opName) {
  std::vector<BodyResultSpec> results = {
      BodyResultSpec{result.str(), "!fabric.bits<32>"}};
  appendBodyOp(
      module,
      BodyOpSpec{
          results,
          {directBodyLine(
               {"fabric.pe [spatial] (%value = ", " : !fabric.bits<32>)"},
               {input.str()}),
           bodyResultTypeLine(results, "        -> ", " {"),
           nestedBodyLine(
               "      fabric.fu(%input = %value : !fabric.bits<32>) -> "
               "!fabric.bits<32> {"),
           nestedBodyLine("        %result = fabric.op [@" + opName.str() +
                          "] (%input)"),
           nestedBodyLine(
               "                 : (!fabric.bits<32>) -> !fabric.bits<32>"),
           nestedBodyLine("        fabric.yield %result : !fabric.bits<32>"),
           nestedBodyLine("      }"), nestedBodyLine("    }")}});
}

void addWideExtensionPe(ModuleBuilder &module, llvm::StringRef result,
                        llvm::StringRef input, llvm::StringRef opName) {
  std::vector<BodyResultSpec> results = {
      BodyResultSpec{result.str(), "!fabric.bits<64>"}};
  appendBodyOp(
      module,
      BodyOpSpec{
          results,
          {directBodyLine({"fabric.pe [spatial] (%value = ",
                           " : !fabric.bits<32> to !fabric.bits<64>)"},
                          {input.str()}),
           bodyResultTypeLine(results, "        -> ", " {"),
           nestedBodyLine(
               "      fabric.fu(%input = %value : !fabric.bits<64> to "
               "!fabric.bits<32>) -> !fabric.bits<64> {"),
           nestedBodyLine("        %result = fabric.op [@" + opName.str() +
                          "] (%input)"),
           nestedBodyLine(
               "                 : (!fabric.bits<32>) -> !fabric.bits<64>"),
           nestedBodyLine("        fabric.yield %result : !fabric.bits<64>"),
           nestedBodyLine("      }"), nestedBodyLine("    }")}});
}

void addWideNarrowingPe(ModuleBuilder &module, llvm::StringRef result,
                        llvm::StringRef input, llvm::StringRef opName) {
  std::vector<BodyResultSpec> results = {
      BodyResultSpec{result.str(), "!fabric.bits<64>"}};
  appendBodyOp(
      module,
      BodyOpSpec{
          results,
          {directBodyLine(
               {"fabric.pe [spatial] (%value = ", " : !fabric.bits<64>)"},
               {input.str()}),
           bodyResultTypeLine(results, "        -> ", " {"),
           nestedBodyLine(
               "      fabric.fu(%input = %value : !fabric.bits<64>) -> "
               "!fabric.bits<64> {"),
           nestedBodyLine("        %result = fabric.op [@" + opName.str() +
                          "] (%input)"),
           nestedBodyLine(
               "                 : (!fabric.bits<64>) -> !fabric.bits<32>"),
           nestedBodyLine("        fabric.yield %result : !fabric.bits<32> to "
                          "!fabric.bits<64>"),
           nestedBodyLine("      }"), nestedBodyLine("    }")}});
}

void addWideTruncPe(ModuleBuilder &module, llvm::StringRef result,
                    llvm::StringRef input) {
  addWideNarrowingPe(module, result, input, "llvm.trunc");
}

void addTernaryPe(ModuleBuilder &module, llvm::StringRef result,
                  llvm::StringRef lhs, llvm::StringRef rhs, llvm::StringRef acc,
                  llvm::StringRef opName) {
  std::vector<BodyResultSpec> results = {
      BodyResultSpec{result.str(), "!fabric.bits<32>"}};
  appendBodyOp(
      module,
      BodyOpSpec{
          results,
          {directBodyLine(
               {"fabric.pe [spatial] (%lhs = ", " : !fabric.bits<32>,"},
               {lhs.str()}),
           directBodyLine(
               {"                         %rhs = ", " : !fabric.bits<32>,"},
               {rhs.str()}),
           directBodyLine(
               {"                         %acc = ", " : !fabric.bits<32>)"},
               {acc.str()}),
           bodyResultTypeLine(results, "        -> ", " {"),
           nestedBodyLine("      fabric.fu(%a = %lhs : !fabric.bits<32>,"),
           nestedBodyLine("                %b = %rhs : !fabric.bits<32>,"),
           nestedBodyLine("                %c = %acc : !fabric.bits<32>) -> "
                          "!fabric.bits<32> {"),
           nestedBodyLine("        %value = fabric.op [@" + opName.str() +
                          "] (%a, %b, %c)"),
           nestedBodyLine(
               "                 : (!fabric.bits<32>, !fabric.bits<32>, "
               "!fabric.bits<32>) -> !fabric.bits<32>"),
           nestedBodyLine("        fabric.yield %value : !fabric.bits<32>"),
           nestedBodyLine("      }"), nestedBodyLine("    }")}});
}

std::string numbered(llvm::StringRef prefix, unsigned index) {
  return (prefix + llvm::Twine(index)).str();
}

void addConfigurableConstantPe(ModuleBuilder &module, llvm::StringRef result,
                               llvm::StringRef control,
                               llvm::ArrayRef<llvm::StringRef> constHexValues) {
  PeSpec pe;
  pe.inputs = {{"pa", control.str(), "!fabric.bits<0>", "!fabric.bits<32>"}};
  pe.resultNames = {result.str()};
  pe.resultTypes = {"!fabric.bits<32>"};
  FuSpec fu;
  fu.inputs = {{"token", "pa", "!fabric.bits<32>", "!fabric.bits<0>"}};
  fu.resultTypes = {"!fabric.bits<32>"};
  FabricOpSpec op;
  op.results = {"value"};
  op.opList = {"dataflow.constant"};
  op.operands = {"token"};
  op.operandTypes = {"!fabric.bits<0>"};
  op.resultTypes = {"!fabric.bits<32>"};
  op.hwParams["const_hex_value"] = stringList(constHexValues);
  fu.operations.push_back(std::move(op));
  fu.yieldValues = {"value"};
  pe.fus.push_back(std::move(fu));
  module.addPe(std::move(pe));
}

void addConfigurableWideConstantPe(
    ModuleBuilder &module, llvm::StringRef result, llvm::StringRef control,
    llvm::ArrayRef<llvm::StringRef> constHexValues) {
  PeSpec pe;
  pe.inputs = {{"pa", control.str(), "!fabric.bits<0>", "!fabric.bits<64>"}};
  pe.resultNames = {result.str()};
  pe.resultTypes = {"!fabric.bits<64>"};
  FuSpec fu;
  fu.inputs = {{"token", "pa", "!fabric.bits<64>", "!fabric.bits<0>"}};
  fu.resultTypes = {"!fabric.bits<64>"};
  FabricOpSpec op;
  op.results = {"value"};
  op.opList = {"dataflow.constant"};
  op.operands = {"token"};
  op.operandTypes = {"!fabric.bits<0>"};
  op.resultTypes = {"!fabric.bits<64>"};
  op.hwParams["const_hex_value"] = stringList(constHexValues);
  fu.operations.push_back(std::move(op));
  fu.yieldValues = {"value"};
  pe.fus.push_back(std::move(fu));
  module.addPe(std::move(pe));
}

void addConfigurableBinaryPe(ModuleBuilder &module, llvm::StringRef result,
                             llvm::StringRef lhs, llvm::StringRef rhs,
                             llvm::ArrayRef<llvm::StringRef> opNames) {
  PeSpec pe;
  pe.inputs = {{"lhs", lhs.str(), "!fabric.bits<32>", ""},
               {"rhs", rhs.str(), "!fabric.bits<32>", ""}};
  pe.resultNames = {result.str()};
  pe.resultTypes = {"!fabric.bits<32>"};
  FuSpec fu;
  fu.inputs = {{"a", "lhs", "!fabric.bits<32>", ""},
               {"b", "rhs", "!fabric.bits<32>", ""}};
  fu.resultTypes = {"!fabric.bits<32>"};
  fu.operations.push_back(FabricOpSpec{{"value"},
                                       stringList(opNames),
                                       {"a", "b"},
                                       {"!fabric.bits<32>", "!fabric.bits<32>"},
                                       {"!fabric.bits<32>"},
                                       {},
                                       {}});
  fu.yieldValues = {"value"};
  pe.fus.push_back(std::move(fu));
  module.addPe(std::move(pe));
}

void addConfigurableWideBinaryPe(ModuleBuilder &module, llvm::StringRef result,
                                 llvm::StringRef lhs, llvm::StringRef rhs,
                                 llvm::ArrayRef<llvm::StringRef> opNames) {
  PeSpec pe;
  pe.inputs = {{"lhs", lhs.str(), "!fabric.bits<64>", ""},
               {"rhs", rhs.str(), "!fabric.bits<64>", ""}};
  pe.resultNames = {result.str()};
  pe.resultTypes = {"!fabric.bits<64>"};
  FuSpec fu;
  fu.inputs = {{"a", "lhs", "!fabric.bits<64>", ""},
               {"b", "rhs", "!fabric.bits<64>", ""}};
  fu.resultTypes = {"!fabric.bits<64>"};
  fu.operations.push_back(FabricOpSpec{{"value"},
                                       stringList(opNames),
                                       {"a", "b"},
                                       {"!fabric.bits<64>", "!fabric.bits<64>"},
                                       {"!fabric.bits<64>"},
                                       {},
                                       {}});
  fu.yieldValues = {"value"};
  pe.fus.push_back(std::move(fu));
  module.addPe(std::move(pe));
}

void addComparisonPe(ModuleBuilder &module, llvm::StringRef result,
                     llvm::StringRef lhs, llvm::StringRef rhs,
                     llvm::StringRef boundaryType,
                     std::vector<std::string> opNames,
                     std::vector<std::string> predicates) {
  PeSpec pe;
  pe.inputs = {{"lhs", lhs.str(), boundaryType.str(), ""},
               {"rhs", rhs.str(), boundaryType.str(), ""}};
  pe.resultNames = {result.str()};
  pe.resultTypes = {boundaryType.str()};
  FuSpec fu;
  fu.inputs = {{"a", "lhs", boundaryType.str(), ""},
               {"b", "rhs", boundaryType.str(), ""}};
  fu.resultTypes = {boundaryType.str()};
  FabricOpSpec op;
  op.results = {"pred"};
  op.opList = std::move(opNames);
  op.operands = {"a", "b"};
  op.operandTypes = {boundaryType.str(), boundaryType.str()};
  op.resultTypes = {"!fabric.bits<1>"};
  op.hwParams["predicate"] = std::move(predicates);
  fu.operations.push_back(std::move(op));
  fu.yieldValues = {"pred"};
  fu.yieldTypes = {"!fabric.bits<1>"};
  pe.fus.push_back(std::move(fu));
  module.addPe(std::move(pe));
}

void addCmpPe(ModuleBuilder &module, llvm::StringRef result,
              llvm::StringRef lhs, llvm::StringRef rhs) {
  addComparisonPe(
      module, result, lhs, rhs, "!fabric.bits<32>", {"arith.cmpi", "llvm.icmp"},
      {"eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"});
}

void addWideCmpPe(ModuleBuilder &module, llvm::StringRef result,
                  llvm::StringRef lhs, llvm::StringRef rhs) {
  addComparisonPe(
      module, result, lhs, rhs, "!fabric.bits<64>", {"arith.cmpi", "llvm.icmp"},
      {"eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"});
}

void addFloatCmpPe(ModuleBuilder &module, llvm::StringRef result,
                   llvm::StringRef lhs, llvm::StringRef rhs) {
  addComparisonPe(module, result, lhs, rhs, "!fabric.bits<32>", {"arith.cmpf"},
                  {"oeq", "ogt", "oge", "olt", "ole", "one", "ord", "ueq",
                   "ugt", "uge", "ult", "ule", "une", "uno"});
}

void addControlSyncPe(ModuleBuilder &module, llvm::StringRef prefix,
                      unsigned inputCount) {
  PeSpec pe;
  for (unsigned index = 0; index < inputCount; ++index) {
    pe.inputs.push_back(
        {(llvm::Twine("p") + llvm::Twine(index)).str(),
         (prefix + llvm::Twine("_in") + llvm::Twine(index)).str(),
         "!fabric.bits<0>", ""});
    pe.resultNames.push_back(
        (prefix + llvm::Twine("_done") + llvm::Twine(index)).str());
    pe.resultTypes.push_back("!fabric.bits<0>");
  }

  FuSpec fu;
  FabricOpSpec sync;
  for (unsigned index = 0; index < inputCount; ++index) {
    std::string local = (llvm::Twine("f") + llvm::Twine(index)).str();
    fu.inputs.push_back(
        {local, pe.inputs[index].localName, "!fabric.bits<0>", ""});
    fu.resultTypes.push_back("!fabric.bits<0>");
    std::string result = (llvm::Twine("s") + llvm::Twine(index)).str();
    sync.results.push_back(result);
    sync.operands.push_back(local);
    sync.operandTypes.push_back("!fabric.bits<0>");
    sync.resultTypes.push_back("!fabric.bits<0>");
    fu.yieldValues.push_back(result);
  }
  sync.opList.push_back("dataflow.sync");
  sync.swConfigs["bitmask"] = std::string(inputCount, '1');
  fu.operations.push_back(std::move(sync));
  pe.fus.push_back(std::move(fu));
  module.addPe(std::move(pe));
}

void addSelectPe(ModuleBuilder &module, llvm::StringRef result,
                 llvm::StringRef pred, llvm::StringRef trueValue,
                 llvm::StringRef falseValue) {
  PeSpec pe;
  pe.inputs = {{"pred", pred.str(), "!fabric.bits<32>", ""},
               {"true_value", trueValue.str(), "!fabric.bits<32>", ""},
               {"false_value", falseValue.str(), "!fabric.bits<32>", ""}};
  pe.resultNames = {result.str()};
  pe.resultTypes = {"!fabric.bits<32>"};
  auto makeSelectFu = [](llvm::StringRef opName) {
    return FuSpec{{{"sel", "pred", "!fabric.bits<32>", "!fabric.bits<1>"},
                   {"a", "true_value", "!fabric.bits<32>", ""},
                   {"b", "false_value", "!fabric.bits<32>", ""}},
                  {"!fabric.bits<32>"},
                  {FabricOpSpec{{"value"},
                                {opName.str()},
                                {"sel", "a", "b"},
                                {"!fabric.bits<1>", "!fabric.bits<32>",
                                 "!fabric.bits<32>"},
                                {"!fabric.bits<32>"},
                                {},
                                {}}},
                  {"value"}};
  };
  pe.fus.push_back(makeSelectFu("arith.select"));
  pe.fus.push_back(makeSelectFu("llvm.select"));
  module.addPe(std::move(pe));
}

void addWideSelectPe(ModuleBuilder &module, llvm::StringRef result,
                     llvm::StringRef pred, llvm::StringRef trueValue,
                     llvm::StringRef falseValue) {
  PeSpec pe;
  pe.inputs = {{"pred", pred.str(), "!fabric.bits<64>", ""},
               {"true_value", trueValue.str(), "!fabric.bits<64>", ""},
               {"false_value", falseValue.str(), "!fabric.bits<64>", ""}};
  pe.resultNames = {result.str()};
  pe.resultTypes = {"!fabric.bits<64>"};
  auto makeSelectFu = [](llvm::StringRef opName) {
    return FuSpec{{{"sel", "pred", "!fabric.bits<64>", "!fabric.bits<1>"},
                   {"a", "true_value", "!fabric.bits<64>", ""},
                   {"b", "false_value", "!fabric.bits<64>", ""}},
                  {"!fabric.bits<64>"},
                  {FabricOpSpec{{"value"},
                                {opName.str()},
                                {"sel", "a", "b"},
                                {"!fabric.bits<1>", "!fabric.bits<64>",
                                 "!fabric.bits<64>"},
                                {"!fabric.bits<64>"},
                                {},
                                {}}},
                  {"value"}};
  };
  pe.fus.push_back(makeSelectFu("arith.select"));
  pe.fus.push_back(makeSelectFu("llvm.select"));
  module.addPe(std::move(pe));
}

void addDataMuxPe(ModuleBuilder &module, llvm::StringRef result,
                  llvm::StringRef pred, llvm::StringRef falseValue,
                  llvm::StringRef trueValue) {
  PeSpec pe;
  pe.inputs = {{"pred", pred.str(), "!fabric.bits<32>", ""},
               {"false_value", falseValue.str(), "!fabric.bits<32>", ""},
               {"true_value", trueValue.str(), "!fabric.bits<32>", ""}};
  pe.resultNames = {result.str()};
  pe.resultTypes = {"!fabric.bits<32>"};
  pe.fus.push_back(FuSpec{
      {{"sel", "pred", "!fabric.bits<32>", "!fabric.bits<1>"},
       {"a", "false_value", "!fabric.bits<32>", ""},
       {"b", "true_value", "!fabric.bits<32>", ""}},
      {"!fabric.bits<32>"},
      {FabricOpSpec{{"value"},
                    {"dataflow.mux"},
                    {"sel", "a", "b"},
                    {"!fabric.bits<1>", "!fabric.bits<32>", "!fabric.bits<32>"},
                    {"!fabric.bits<32>"},
                    {},
                    {}}},
      {"value"}});
  module.addPe(std::move(pe));
}

void addWideDataMuxPe(ModuleBuilder &module, llvm::StringRef result,
                      llvm::StringRef pred, llvm::StringRef falseValue,
                      llvm::StringRef trueValue) {
  PeSpec pe;
  pe.inputs = {{"pred", pred.str(), "!fabric.bits<64>", ""},
               {"false_value", falseValue.str(), "!fabric.bits<64>", ""},
               {"true_value", trueValue.str(), "!fabric.bits<64>", ""}};
  pe.resultNames = {result.str()};
  pe.resultTypes = {"!fabric.bits<64>"};
  pe.fus.push_back(FuSpec{
      {{"sel", "pred", "!fabric.bits<64>", "!fabric.bits<1>"},
       {"a", "false_value", "!fabric.bits<64>", ""},
       {"b", "true_value", "!fabric.bits<64>", ""}},
      {"!fabric.bits<64>"},
      {FabricOpSpec{{"value"},
                    {"dataflow.mux"},
                    {"sel", "a", "b"},
                    {"!fabric.bits<1>", "!fabric.bits<64>", "!fabric.bits<64>"},
                    {"!fabric.bits<64>"},
                    {},
                    {}}},
      {"value"}});
  module.addPe(std::move(pe));
}

void addControlMuxPe(ModuleBuilder &module, llvm::StringRef result,
                     llvm::StringRef pred, llvm::StringRef falseValue,
                     llvm::StringRef trueValue) {
  PeSpec pe;
  pe.inputs = {
      {"pred", pred.str(), "!fabric.bits<32>", ""},
      {"false_value", falseValue.str(), "!fabric.bits<0>", "!fabric.bits<32>"},
      {"true_value", trueValue.str(), "!fabric.bits<0>", "!fabric.bits<32>"}};
  pe.resultNames = {result.str()};
  pe.resultTypes = {"!fabric.bits<32>"};
  FuSpec fu;
  fu.inputs = {
      {"sel", "pred", "!fabric.bits<32>", "!fabric.bits<1>"},
      {"false_lane", "false_value", "!fabric.bits<32>", "!fabric.bits<0>"},
      {"true_lane", "true_value", "!fabric.bits<32>", "!fabric.bits<0>"}};
  fu.resultTypes = {"!fabric.bits<32>"};
  fu.operations.push_back(
      FabricOpSpec{{"selected"},
                   {"dataflow.mux"},
                   {"sel", "false_lane", "true_lane"},
                   {"!fabric.bits<1>", "!fabric.bits<0>", "!fabric.bits<0>"},
                   {"!fabric.bits<0>"},
                   {},
                   {}});
  fu.yieldValues = {"selected"};
  fu.yieldTypes = {"!fabric.bits<0>"};
  pe.fus.push_back(std::move(fu));
  module.addPe(std::move(pe));
}

void addDataDemuxPe(ModuleBuilder &module, llvm::StringRef falseResult,
                    llvm::StringRef trueResult, llvm::StringRef pred,
                    llvm::StringRef value) {
  PeSpec pe;
  pe.inputs = {{"pred", pred.str(), "!fabric.bits<32>", ""},
               {"value", value.str(), "!fabric.bits<32>", ""}};
  pe.resultNames = {falseResult.str(), trueResult.str()};
  pe.resultTypes = {"!fabric.bits<32>", "!fabric.bits<32>"};
  pe.fus.push_back(
      FuSpec{{{"sel", "pred", "!fabric.bits<32>", "!fabric.bits<1>"},
              {"data", "value", "!fabric.bits<32>", ""}},
             {"!fabric.bits<32>", "!fabric.bits<32>"},
             {FabricOpSpec{{"false_lane", "true_lane"},
                           {"dataflow.demux"},
                           {"sel", "data"},
                           {"!fabric.bits<1>", "!fabric.bits<32>"},
                           {"!fabric.bits<32>", "!fabric.bits<32>"},
                           {},
                           {}}},
             {"false_lane", "true_lane"}});
  module.addPe(std::move(pe));
}

void addWideDataDemuxPe(ModuleBuilder &module, llvm::StringRef falseResult,
                        llvm::StringRef trueResult, llvm::StringRef pred,
                        llvm::StringRef value) {
  PeSpec pe;
  pe.inputs = {{"pred", pred.str(), "!fabric.bits<32>", "!fabric.bits<64>"},
               {"value", value.str(), "!fabric.bits<64>", ""}};
  pe.resultNames = {falseResult.str(), trueResult.str()};
  pe.resultTypes = {"!fabric.bits<64>", "!fabric.bits<64>"};
  pe.fus.push_back(
      FuSpec{{{"sel", "pred", "!fabric.bits<64>", "!fabric.bits<1>"},
              {"data", "value", "!fabric.bits<64>", ""}},
             {"!fabric.bits<64>", "!fabric.bits<64>"},
             {FabricOpSpec{{"false_lane", "true_lane"},
                           {"dataflow.demux"},
                           {"sel", "data"},
                           {"!fabric.bits<1>", "!fabric.bits<64>"},
                           {"!fabric.bits<64>", "!fabric.bits<64>"},
                           {},
                           {}}},
             {"false_lane", "true_lane"}});
  module.addPe(std::move(pe));
}

void addControlDemuxPe(ModuleBuilder &module, llvm::StringRef falseResult,
                       llvm::StringRef trueResult, llvm::StringRef pred,
                       llvm::StringRef value) {
  PeSpec pe;
  pe.inputs = {{"pred", pred.str(), "!fabric.bits<32>", ""},
               {"value", value.str(), "!fabric.bits<0>", "!fabric.bits<32>"}};
  pe.resultNames = {falseResult.str(), trueResult.str()};
  pe.resultTypes = {"!fabric.bits<32>", "!fabric.bits<32>"};
  FuSpec fu;
  fu.inputs = {{"sel", "pred", "!fabric.bits<32>", "!fabric.bits<1>"},
               {"data", "value", "!fabric.bits<32>", "!fabric.bits<0>"}};
  fu.resultTypes = {"!fabric.bits<32>", "!fabric.bits<32>"};
  fu.operations.push_back(FabricOpSpec{{"false_lane", "true_lane"},
                                       {"dataflow.demux"},
                                       {"sel", "data"},
                                       {"!fabric.bits<1>", "!fabric.bits<0>"},
                                       {"!fabric.bits<0>", "!fabric.bits<0>"},
                                       {},
                                       {}});
  fu.yieldValues = {"false_lane", "true_lane"};
  fu.yieldTypes = {"!fabric.bits<0>", "!fabric.bits<0>"};
  pe.fus.push_back(std::move(fu));
  module.addPe(std::move(pe));
}

void addMemoryReductionMem(ModuleBuilder &module, unsigned loadCount,
                           unsigned storeCount) {
  std::vector<BodyResultSpec> results;
  for (unsigned index = 0; index < loadCount; ++index) {
    results.push_back(
        BodyResultSpec{numbered("data", index), "!fabric.bits<32>"});
    results.push_back(
        BodyResultSpec{numbered("done", index), "!fabric.bits<0>"});
  }
  for (unsigned index = 0; index < storeCount; ++index)
    results.push_back(
        BodyResultSpec{numbered("store_done", index), "!fabric.bits<0>"});

  std::vector<std::string> loadOperands;
  for (unsigned index = 0; index < loadCount; ++index) {
    loadOperands.push_back(numbered("load_addr", index));
    loadOperands.push_back(numbered("load_ctrl", index));
  }
  std::vector<std::string> storeOperands;
  for (unsigned index = 0; index < storeCount; ++index) {
    storeOperands.push_back(numbered("store_addr", index));
    storeOperands.push_back(numbered("store_value", index));
    storeOperands.push_back(numbered("store_ctrl", index));
  }

  std::string operandTypes = "(memref<?x!fabric.bits<32>>";
  for (unsigned index = 0; index < loadCount; ++index)
    operandTypes += ", !fabric.bits<32>, !fabric.bits<0>";
  for (unsigned index = 0; index < storeCount; ++index)
    operandTypes += ", !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<0>";
  operandTypes += ")";

  appendBodyOp(
      module,
      bodyOpWithResultLine(
          std::move(results),
          {directHeadAndListLine("fabric.mem [spatial] mgr(", "mgr", ") load(",
                                 loadOperands, ")"),
           directOperandListLine("                              store(",
                                 storeOperands, ")"),
           exactBodyLine(
               "      [{load_group_size = " + std::to_string(loadCount) +
               " : i32, store_group_size = " + std::to_string(storeCount) +
               " : i32, data_width = 32 : i32" +
               singleManagerDispatchEligibility(loadCount + storeCount) + "}]"),
           exactBodyLine("      : " + operandTypes)},
          "      -> "));
}

void addTwoLoadOneStoreMem(ModuleBuilder &module) {
  appendBodyOp(
      module,
      bodyOpWithResultLine(
          {BodyResultSpec{"data0", "!fabric.bits<32>"},
           BodyResultSpec{"done0", "!fabric.bits<0>"},
           BodyResultSpec{"data1", "!fabric.bits<32>"},
           BodyResultSpec{"done1", "!fabric.bits<0>"},
           BodyResultSpec{"store_done", "!fabric.bits<0>"}},
          {directBodyLine({"fabric.mem [spatial] mgr(", ")"}, {"mgr"}),
           directOperandListLine("      load(",
                                 {"idx0", "load_ctrl0", "idx1", "load_ctrl1"},
                                 ")"),
           directOperandListLine(
               "      store(", {"store_idx", "store_value", "store_ctrl"}, ")"),
           exactBodyLine(
               "      [{load_group_size = 2 : i32, store_group_size = 1 : "
               "i32, data_width = 32 : i32" +
               singleManagerDispatchEligibility(3) + "}]"),
           exactBodyLine(
               "      : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, "
               "!fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, "
               "!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<0>)")},
          "      -> "));
}

ModuleBuilder buildChain1DAdg() {
  ModuleBuilder module = makeTopologyMatrixModule(
      "matrix_chain1d_adg", false,
      {{"mem", 0, 0}, {"p0", 1, 0}, {"s0", 2, 0}, {"p1", 3, 0}, {"p2", 4, 0}});
  addSpatialMemLoad(module);
  addSpatialAddPe(module, "p0", "data", "a");
  addSpatialSwitch(module, {"s0"}, {"p0", "b"}, {"11"});
  addSpatialAddPe(module, "p1", "s0", "c");
  addSpatialAddPe(module, "p2", "p1", "d");
  return module;
}

ModuleBuilder buildMesh2DAdg() {
  ModuleBuilder module = makeTopologyMatrixModule("matrix_mesh2d_adg", false,
                                                  {{"mem", 0, 0},
                                                   {"n00", 1, 0},
                                                   {"n01", 2, 0},
                                                   {"n10", 1, 1},
                                                   {"n11", 2, 1}});
  addSpatialMemLoad(module);
  addSpatialAddPe(module, "n00", "data", "a");
  addSpatialAddPe(module, "n01", "data", "b");
  addSpatialSwitch(module, {"east", "south"}, {"n00", "n01"}, {"11", "11"});
  addSpatialAddPe(module, "n10", "east", "c");
  addSpatialAddPe(module, "n11", "south", "n10");
  return module;
}

ModuleBuilder buildTorusEdgeAdg() {
  ModuleBuilder module =
      makeTopologyMatrixModule("matrix_torus_edge_adg", false,
                               {{"mem", 0, 0},
                                {"n00", 1, 0},
                                {"n01", 2, 0},
                                {"n10", 1, 1},
                                {"n11", 2, 1},
                                {"wrap_north", 1, -1},
                                {"wrap_west", 0, 1}});
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
  ModuleBuilder module =
      makeTopologyMatrixModule("matrix_systolic_array_adg", false,
                               {{"mem", 0, 0},
                                {"broadcast", 1, 0},
                                {"cell0", 2, 0},
                                {"cell1", 3, 0},
                                {"cell2", 4, 0}});
  addSpatialMemLoad(module);
  addSpatialSwitch(module, {"broadcast"}, {"data", "a", "b"}, {"111"});
  addSpatialAddPe(module, "cell0", "broadcast", "c", "arith.mulf");
  addSpatialAddPe(module, "cell1", "cell0", "d", "arith.addf");
  addSpatialAddPe(module, "cell2", "cell1", "broadcast", "arith.addf");
  return module;
}

ModuleBuilder buildClusteredArrayAdg() {
  ModuleBuilder module =
      makeTopologyMatrixModule("matrix_clustered_array_adg", false,
                               {{"mem", 0, 0},
                                {"c0a", 1, 0},
                                {"c0b", 1, 1},
                                {"cluster0", 2, 0},
                                {"c1a", 3, 0},
                                {"c1b", 3, 1},
                                {"cluster1", 4, 0},
                                {"out", 5, 0}});
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

ModuleBuilder buildFoldedRingAdg() {
  ModuleBuilder module =
      makeTopologyMatrixModule("matrix_folded_ring_adg", false,
                               {{"mem", 0, 0},
                                {"n0", 1, 0},
                                {"n1", 2, 0},
                                {"n2", 2, 1},
                                {"n3", 1, 1},
                                {"wrap", 0, 1}});
  addSpatialMemLoad(module);
  addSpatialAddPe(module, "n0", "data", "a");
  addSpatialAddPe(module, "n1", "n0", "b");
  addSpatialAddPe(module, "n2", "n1", "c");
  addSpatialSwitch(module, {"wrap", "forward"}, {"n2", "n0", "d"},
                   {"110", "101"});
  addSpatialAddPe(module, "n3", "wrap", "forward");
  return module;
}

ModuleBuilder buildMeshDiagonalAdg() {
  ModuleBuilder module =
      makeTopologyMatrixModule("matrix_mesh_diagonal_adg", false,
                               {{"mem", 0, 0},
                                {"n00", 1, 0},
                                {"n01", 2, 0},
                                {"n10", 1, 1},
                                {"n11", 2, 1},
                                {"diag", 2, 2}});
  addSpatialMemLoad(module);
  addSpatialAddPe(module, "n00", "data", "a");
  addSpatialAddPe(module, "n01", "data", "b");
  addSpatialAddPe(module, "n10", "c", "d");
  addSpatialSwitch(module, {"east", "south", "diag"}, {"n00", "n01", "n10"},
                   {"110", "101", "011"});
  addSpatialAddPe(module, "n11", "diag", "south");
  return module;
}

ModuleBuilder buildMultiLanePipelineAdg() {
  ModuleBuilder module =
      makeTopologyMatrixModule("matrix_multi_lane_pipeline_adg", false,
                               {{"mem", 0, 0},
                                {"lane0_in", 1, 0},
                                {"lane1_in", 1, 1},
                                {"lane0_stage0", 2, 0},
                                {"lane1_stage0", 2, 1},
                                {"lane0_stage1", 3, 0},
                                {"lane1_stage1", 3, 1},
                                {"merged", 4, 0},
                                {"out", 5, 0}});
  addSpatialMemLoad(module);
  addSpatialSwitch(module, {"lane0_in", "lane1_in"}, {"data", "a", "b"},
                   {"110", "101"});
  addSpatialAddPe(module, "lane0_stage0", "lane0_in", "c");
  addSpatialAddPe(module, "lane1_stage0", "lane1_in", "d");
  addSpatialAddPe(module, "lane0_stage1", "lane0_stage0", "lane1_stage0");
  addSpatialAddPe(module, "lane1_stage1", "lane1_stage0", "lane0_stage0");
  addSpatialSwitch(module, {"merged"}, {"lane0_stage1", "lane1_stage1"},
                   {"11"});
  addSpatialAddPe(module, "out", "merged", "data");
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

ModuleBuilder buildDiamondBypassAdg() {
  ModuleBuilder module = makeTopologyMatrixModule("matrix_diamond_bypass_adg");
  addSpatialMemLoad(module);
  addSpatialAddPe(module, "entry", "data", "a");
  addSpatialAddPe(module, "upper", "entry", "b");
  addSpatialAddPe(module, "lower", "entry", "c");
  addSpatialSwitch(module, {"join", "bypass"}, {"upper", "lower", "entry"},
                   {"110", "001"});
  addSpatialAddPe(module, "exit", "join", "bypass");
  return module;
}

ModuleBuilder buildMemoryFanoutAdg() {
  ModuleBuilder module = makeTopologyMatrixModule("matrix_memory_fanout_adg");
  addSpatialMemLoad(module);
  addSpatialSwitch(module, {"data_lane0", "data_lane1", "data_lane2"},
                   {"data", "a", "b"}, {"111", "101", "110"});
  addSpatialAddPe(module, "lane0", "data_lane0", "c");
  addSpatialAddPe(module, "lane1", "data_lane1", "d");
  addSpatialAddPe(module, "lane2", "data_lane2", "lane0");
  addSpatialSwitch(module, {"combined"}, {"lane0", "lane1", "lane2"}, {"111"});
  addSpatialAddPe(module, "out", "combined", "data");
  return module;
}

ModuleBuilder buildMixedTemporalBridgeAdg() {
  ModuleBuilder module =
      makeTopologyMatrixModule("matrix_mixed_temporal_bridge_adg", true);
  TemporalPeConfig temporal;
  temporal.tagWidth = 4;
  temporal.numInstruction = 3;
  temporal.fuConfigMode = "per_fu_config";
  temporal.operandBufferMode = ::fabric::OperandBufferMode::PerInputPort;
  temporal.operandBufferSize = 4;
  module.addPe(makeMinimalAddPe(Schedule::Temporal, "lhs_t", "rhs_t",
                                "!fabric.bits_tag<32, 4>", "!fabric.bits<32>",
                                std::move(temporal)));
  addSpatialMemLoad(module);
  addSpatialAddPe(module, "spatial0", "data", "a");
  module
      .addBoundary(BoundarySpec{::fabric::BoundaryDirection::S2t,
                                {{"spatial0"}, {"tag"}},
                                {"tagged"},
                                {"!fabric.bits_tag<32, 4>"}})
      .addFifo(FifoSpec{"queued", "tagged", "!fabric.bits_tag<32, 4>", 4, true})
      .addBoundary(BoundarySpec{::fabric::BoundaryDirection::T2s,
                                {{"queued"}},
                                {"untagged"},
                                {"!fabric.bits<32>"}});
  addSpatialSwitch(module, {"bridge_out"}, {"untagged", "b", "c"}, {"111"});
  addSpatialAddPe(module, "spatial1", "bridge_out", "d");
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
  temporal.operandBufferMode = ::fabric::OperandBufferMode::PerInputPort;
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

SystemBuilder buildDualSpatialSharedMemorySocAdg() {
  SystemBuilder system("system_dual_spatial_shared_memory_soc", "sequential");
  system.addHostCore("host0", "rv64gc", axiManagerPort("mem"));
  system.addSpatialAccelerator("acc0", "shared_reduction_adg", "rv32im",
                               axiManagerPort("mem"));
  system.addSpatialAccelerator("acc1", "shared_vector_alu_adg", "rv32imc",
                               axiManagerPort("mem"));

  std::vector<std::string> dramPorts;
  appendPorts(dramPorts, axiSubordinatePort("host0"));
  appendPorts(dramPorts, axiSubordinatePort("acc0"));
  appendPorts(dramPorts, axiSubordinatePort("acc1"));
  system.addMemory("dram0", 2 * 1024 * 1024, std::move(dramPorts));

  connectAxiMemoryPort(system, "host0", "mem", "dram0", "host0");
  connectAxiMemoryPort(system, "acc0", "mem", "dram0", "acc0");
  connectAxiMemoryPort(system, "acc1", "mem", "dram0", "acc1");
  return system;
}

SystemBuilder buildCachedDualAccelSocAdg() {
  SystemBuilder system("system_cached_dual_accel_soc", "release_acquire");
  system.addHostCore("host0", "rv64gc", axiManagerPort("mem"));
  system.addSpatialAccelerator("acc0", "shared_reduction_adg", "rv32im",
                               axiManagerPort("mem"));
  system.addSpatialAccelerator("acc1", "shared_vector_alu_adg", "rv32imc",
                               axiManagerPort("mem"));

  std::vector<std::string> hostCachePorts;
  appendPorts(hostCachePorts, axiSubordinatePort("host"));
  appendPorts(hostCachePorts, axiManagerPort("mem"));
  system.addCache("l1d0", 64, 32 * 1024, std::move(hostCachePorts));

  std::vector<std::string> accCachePorts;
  appendPorts(accCachePorts, axiSubordinatePort("acc"));
  appendPorts(accCachePorts, axiManagerPort("mem"));
  system.addCache("acc_l1d0", 64, 16 * 1024, std::move(accCachePorts));

  std::vector<std::string> dramPorts;
  appendPorts(dramPorts, axiSubordinatePort("host_cache"));
  appendPorts(dramPorts, axiSubordinatePort("acc_cache"));
  appendPorts(dramPorts, axiSubordinatePort("acc1"));
  system.addMemory("dram0", 4 * 1024 * 1024, std::move(dramPorts));

  connectAxiMemoryPort(system, "host0", "mem", "l1d0", "host");
  connectAxiMemoryPort(system, "l1d0", "mem", "dram0", "host_cache");
  connectAxiMemoryPort(system, "acc0", "mem", "acc_l1d0", "acc");
  connectAxiMemoryPort(system, "acc_l1d0", "mem", "dram0", "acc_cache");
  connectAxiMemoryPort(system, "acc1", "mem", "dram0", "acc1");
  return system;
}

SystemBuilder buildDmaScratchpadSocAdg() {
  SystemBuilder system("system_dma_scratchpad_soc", "tso");
  system.addHostCore("host0", "rv64gc", axiManagerPort("mem"));
  system.addSpatialAccelerator("acc0", "shared_reduction_adg", "rv32im",
                               axiManagerPort("mem"));

  std::vector<std::string> dmaPorts;
  appendPorts(dmaPorts, axiSubordinatePort("ctrl"));
  appendPorts(dmaPorts, axiManagerPort("mem"));
  system.addDmaEngine("dma0", 8, std::move(dmaPorts));

  std::vector<std::string> scratchPorts;
  appendPorts(scratchPorts, axiSubordinatePort("acc0"));
  appendPorts(scratchPorts, axiSubordinatePort("dma0"));
  system.addMemory("scratch0", 256 * 1024, std::move(scratchPorts));

  connectAxiMemoryPort(system, "host0", "mem", "dma0", "ctrl");
  connectAxiMemoryPort(system, "dma0", "mem", "scratch0", "dma0");
  connectAxiMemoryPort(system, "acc0", "mem", "scratch0", "acc0");
  return system;
}

SystemBuilder buildFixedAndSpatialSocAdg() {
  SystemBuilder system("system_fixed_and_spatial_soc", "sequential");
  system.addHostCore("host0", "rv64gc", axiManagerPort("mem"));
  system.addSpatialAccelerator("acc0", "shared_reduction_adg", "rv32im",
                               axiManagerPort("mem"));
  system.addFixedAccelerator("crypto0", "xor_block", axiManagerPort("mem"));

  std::vector<std::string> dramPorts;
  appendPorts(dramPorts, axiSubordinatePort("host0"));
  appendPorts(dramPorts, axiSubordinatePort("acc0"));
  appendPorts(dramPorts, axiSubordinatePort("crypto0"));
  system.addMemory("dram0", 1024 * 1024, std::move(dramPorts));

  connectAxiMemoryPort(system, "host0", "mem", "dram0", "host0");
  connectAxiMemoryPort(system, "acc0", "mem", "dram0", "acc0");
  connectAxiMemoryPort(system, "crypto0", "mem", "dram0", "crypto0");
  return system;
}

SystemBuilder buildTriSpatialSharedMemorySocAdg() {
  SystemBuilder system("system_tri_spatial_shared_memory_soc", "sequential");
  system.addHostCore("host0", "rv64gc", axiManagerPort("mem"));
  system.addSpatialAccelerator("acc0", "shared_reduction_adg", "rv32im",
                               axiManagerPort("mem"));
  system.addSpatialAccelerator("acc1", "shared_vector_alu_adg", "rv32imc",
                               axiManagerPort("mem"));
  system.addSpatialAccelerator("acc2", "shared_memory_reduction_adg", "rv32im",
                               axiManagerPort("mem"));

  std::vector<std::string> dramPorts;
  appendPorts(dramPorts, axiSubordinatePort("host0"));
  appendPorts(dramPorts, axiSubordinatePort("acc0"));
  appendPorts(dramPorts, axiSubordinatePort("acc1"));
  appendPorts(dramPorts, axiSubordinatePort("acc2"));
  system.addMemory("dram0", 4 * 1024 * 1024, std::move(dramPorts));

  connectAxiMemoryPort(system, "host0", "mem", "dram0", "host0");
  connectAxiMemoryPort(system, "acc0", "mem", "dram0", "acc0");
  connectAxiMemoryPort(system, "acc1", "mem", "dram0", "acc1");
  connectAxiMemoryPort(system, "acc2", "mem", "dram0", "acc2");
  return system;
}

SystemBuilder buildDualHostSharedMemorySocAdg() {
  SystemBuilder system("system_dual_host_shared_memory_soc", "release_acquire");
  system.addHostCore("host0", "rv64gc", axiManagerPort("mem"));
  system.addHostCore("host1", "rv64gc", axiManagerPort("mem"));
  system.addSpatialAccelerator("acc0", "shared_reduction_adg", "rv32im",
                               axiManagerPort("mem"));
  system.addSpatialAccelerator("acc1", "shared_vector_alu_adg", "rv32imc",
                               axiManagerPort("mem"));

  std::vector<std::string> dramPorts;
  appendPorts(dramPorts, axiSubordinatePort("host0"));
  appendPorts(dramPorts, axiSubordinatePort("host1"));
  appendPorts(dramPorts, axiSubordinatePort("acc0"));
  appendPorts(dramPorts, axiSubordinatePort("acc1"));
  system.addMemory("dram0", 8 * 1024 * 1024, std::move(dramPorts));

  connectAxiMemoryPort(system, "host0", "mem", "dram0", "host0");
  connectAxiMemoryPort(system, "host1", "mem", "dram0", "host1");
  connectAxiMemoryPort(system, "acc0", "mem", "dram0", "acc0");
  connectAxiMemoryPort(system, "acc1", "mem", "dram0", "acc1");
  return system;
}

SystemBuilder buildPrivateScratchpadPairSocAdg() {
  SystemBuilder system("system_private_scratchpad_pair_soc", "sequential");
  std::vector<std::string> hostPorts;
  appendPorts(hostPorts, axiManagerPort("mem0"));
  appendPorts(hostPorts, axiManagerPort("mem1"));
  system.addHostCore("host0", "rv64gc", std::move(hostPorts));
  system.addSpatialAccelerator("acc0", "shared_reduction_adg", "rv32im",
                               axiManagerPort("mem"));
  system.addSpatialAccelerator("acc1", "shared_vector_alu_adg", "rv32imc",
                               axiManagerPort("mem"));

  std::vector<std::string> scratch0Ports;
  appendPorts(scratch0Ports, axiSubordinatePort("host"));
  appendPorts(scratch0Ports, axiSubordinatePort("acc0"));
  system.addMemory("scratch0", 256 * 1024, std::move(scratch0Ports));

  std::vector<std::string> scratch1Ports;
  appendPorts(scratch1Ports, axiSubordinatePort("host"));
  appendPorts(scratch1Ports, axiSubordinatePort("acc1"));
  system.addMemory("scratch1", 256 * 1024, std::move(scratch1Ports));

  connectAxiMemoryPort(system, "host0", "mem0", "scratch0", "host");
  connectAxiMemoryPort(system, "host0", "mem1", "scratch1", "host");
  connectAxiMemoryPort(system, "acc0", "mem", "scratch0", "acc0");
  connectAxiMemoryPort(system, "acc1", "mem", "scratch1", "acc1");
  return system;
}

SystemBuilder buildHostCacheDualMemorySocAdg() {
  SystemBuilder system("system_host_cache_dual_memory_soc", "release_acquire");
  system.addHostCore("host0", "rv64gc", axiManagerPort("mem"));
  system.addSpatialAccelerator("acc0", "shared_reduction_adg", "rv32im",
                               axiManagerPort("mem"));
  system.addSpatialAccelerator("acc1", "shared_vector_alu_adg", "rv32imc",
                               axiManagerPort("mem"));

  std::vector<std::string> cachePorts;
  appendPorts(cachePorts, axiSubordinatePort("host"));
  appendPorts(cachePorts, axiManagerPort("mem0"));
  appendPorts(cachePorts, axiManagerPort("mem1"));
  system.addCache("l1d0", 64, 64 * 1024, std::move(cachePorts));

  std::vector<std::string> dram0Ports;
  appendPorts(dram0Ports, axiSubordinatePort("cache"));
  appendPorts(dram0Ports, axiSubordinatePort("acc0"));
  system.addMemory("dram0", 4 * 1024 * 1024, std::move(dram0Ports));

  std::vector<std::string> dram1Ports;
  appendPorts(dram1Ports, axiSubordinatePort("cache"));
  appendPorts(dram1Ports, axiSubordinatePort("acc1"));
  system.addMemory("dram1", 4 * 1024 * 1024, std::move(dram1Ports));

  connectAxiMemoryPort(system, "host0", "mem", "l1d0", "host");
  connectAxiMemoryPort(system, "l1d0", "mem0", "dram0", "cache");
  connectAxiMemoryPort(system, "l1d0", "mem1", "dram1", "cache");
  connectAxiMemoryPort(system, "acc0", "mem", "dram0", "acc0");
  connectAxiMemoryPort(system, "acc1", "mem", "dram1", "acc1");
  return system;
}

SystemBuilder buildDmaDualMemorySocAdg() {
  SystemBuilder system("system_dma_dual_memory_soc", "tso");
  system.addHostCore("host0", "rv64gc", axiManagerPort("mem"));
  system.addSpatialAccelerator("acc0", "shared_reduction_adg", "rv32im",
                               axiManagerPort("mem"));

  std::vector<std::string> dmaPorts;
  appendPorts(dmaPorts, axiSubordinatePort("ctrl"));
  appendPorts(dmaPorts, axiManagerPort("src"));
  appendPorts(dmaPorts, axiManagerPort("dst"));
  system.addDmaEngine("dma0", 16, std::move(dmaPorts));

  std::vector<std::string> srcPorts;
  appendPorts(srcPorts, axiSubordinatePort("dma0"));
  system.addMemory("src_mem", 1024 * 1024, std::move(srcPorts));

  std::vector<std::string> dstPorts;
  appendPorts(dstPorts, axiSubordinatePort("dma0"));
  appendPorts(dstPorts, axiSubordinatePort("acc0"));
  system.addMemory("dst_mem", 1024 * 1024, std::move(dstPorts));

  connectAxiMemoryPort(system, "host0", "mem", "dma0", "ctrl");
  connectAxiMemoryPort(system, "dma0", "src", "src_mem", "dma0");
  connectAxiMemoryPort(system, "dma0", "dst", "dst_mem", "dma0");
  connectAxiMemoryPort(system, "acc0", "mem", "dst_mem", "acc0");
  return system;
}

SystemBuilder buildCachedAcceleratorClusterSocAdg() {
  SystemBuilder system("system_cached_accelerator_cluster_soc",
                       "release_acquire");
  system.addHostCore("host0", "rv64gc", axiManagerPort("mem"));
  system.addSpatialAccelerator("acc0", "shared_reduction_adg", "rv32im",
                               axiManagerPort("mem"));
  system.addSpatialAccelerator("acc1", "shared_vector_alu_adg", "rv32imc",
                               axiManagerPort("mem"));

  std::vector<std::string> hostCachePorts;
  appendPorts(hostCachePorts, axiSubordinatePort("host"));
  appendPorts(hostCachePorts, axiManagerPort("mem"));
  system.addCache("l1d0", 64, 32 * 1024, std::move(hostCachePorts));

  std::vector<std::string> acc0CachePorts;
  appendPorts(acc0CachePorts, axiSubordinatePort("acc"));
  appendPorts(acc0CachePorts, axiManagerPort("mem"));
  system.addCache("acc_l1d0", 64, 16 * 1024, std::move(acc0CachePorts));

  std::vector<std::string> acc1CachePorts;
  appendPorts(acc1CachePorts, axiSubordinatePort("acc"));
  appendPorts(acc1CachePorts, axiManagerPort("mem"));
  system.addCache("acc_l1d1", 64, 16 * 1024, std::move(acc1CachePorts));

  std::vector<std::string> dramPorts;
  appendPorts(dramPorts, axiSubordinatePort("host_cache"));
  appendPorts(dramPorts, axiSubordinatePort("acc0_cache"));
  appendPorts(dramPorts, axiSubordinatePort("acc1_cache"));
  system.addMemory("dram0", 8 * 1024 * 1024, std::move(dramPorts));

  connectAxiMemoryPort(system, "host0", "mem", "l1d0", "host");
  connectAxiMemoryPort(system, "l1d0", "mem", "dram0", "host_cache");
  connectAxiMemoryPort(system, "acc0", "mem", "acc_l1d0", "acc");
  connectAxiMemoryPort(system, "acc_l1d0", "mem", "dram0", "acc0_cache");
  connectAxiMemoryPort(system, "acc1", "mem", "acc_l1d1", "acc");
  connectAxiMemoryPort(system, "acc_l1d1", "mem", "dram0", "acc1_cache");
  return system;
}

SystemBuilder buildMixedFixedSpatialPipelineSocAdg() {
  SystemBuilder system("system_mixed_fixed_spatial_pipeline_soc", "sequential");
  system.addHostCore("host0", "rv64gc", axiManagerPort("mem"));
  system.addSpatialAccelerator("acc0", "shared_reduction_adg", "rv32im",
                               axiManagerPort("mem"));
  system.addSpatialAccelerator("acc1", "shared_vector_alu_adg", "rv32imc",
                               axiManagerPort("mem"));
  system.addFixedAccelerator("fft0", "fft", axiManagerPort("mem"));
  system.addFixedAccelerator("crypto0", "xor_block", axiManagerPort("mem"));

  std::vector<std::string> dramPorts;
  appendPorts(dramPorts, axiSubordinatePort("host0"));
  appendPorts(dramPorts, axiSubordinatePort("acc0"));
  appendPorts(dramPorts, axiSubordinatePort("acc1"));
  appendPorts(dramPorts, axiSubordinatePort("fft0"));
  appendPorts(dramPorts, axiSubordinatePort("crypto0"));
  system.addMemory("dram0", 4 * 1024 * 1024, std::move(dramPorts));

  connectAxiMemoryPort(system, "host0", "mem", "dram0", "host0");
  connectAxiMemoryPort(system, "acc0", "mem", "dram0", "acc0");
  connectAxiMemoryPort(system, "acc1", "mem", "dram0", "acc1");
  connectAxiMemoryPort(system, "fft0", "mem", "dram0", "fft0");
  connectAxiMemoryPort(system, "crypto0", "mem", "dram0", "crypto0");
  return system;
}

SystemBuilder buildSignalQuantizedPairSocAdg() {
  SystemBuilder system("system_signal_quantized_pair_soc", "sequential");
  system.addHostCore("host0", "rv64gc", axiManagerPort("mem"));
  system.addSpatialAccelerator("acc0", "shared_signal_window_adg", "rv32im",
                               axiManagerPort("mem"));
  system.addSpatialAccelerator("acc1", "shared_quantized_window_adg", "rv32imc",
                               axiManagerPort("mem"));

  std::vector<std::string> dramPorts;
  appendPorts(dramPorts, axiSubordinatePort("host0"));
  appendPorts(dramPorts, axiSubordinatePort("acc0"));
  appendPorts(dramPorts, axiSubordinatePort("acc1"));
  system.addMemory("dram0", 8 * 1024 * 1024, std::move(dramPorts));

  connectAxiMemoryPort(system, "host0", "mem", "dram0", "host0");
  connectAxiMemoryPort(system, "acc0", "mem", "dram0", "acc0");
  connectAxiMemoryPort(system, "acc1", "mem", "dram0", "acc1");
  return system;
}

llvm::Error printReusableSpatialTemplates(llvm::raw_ostream &os,
                                          bool includeVectorAlu,
                                          bool includeMemoryReduction,
                                          bool includeSignalWindow,
                                          bool includeQuantizedWindow) {
  if (llvm::Error err = buildSharedReductionAdg().print(os))
    return err;
  if (includeVectorAlu) {
    os << '\n';
    if (llvm::Error err = buildSharedVectorAluAdg().print(os))
      return err;
  }
  if (includeMemoryReduction) {
    os << '\n';
    if (llvm::Error err = buildSharedMemoryReductionAdg().print(os))
      return err;
  }
  if (includeSignalWindow) {
    os << '\n';
    if (llvm::Error err = buildSharedSignalWindowAdg().print(os))
      return err;
  }
  if (includeQuantizedWindow) {
    os << '\n';
    if (llvm::Error err = buildSharedQuantizedWindowAdg().print(os))
      return err;
  }
  os << '\n';
  return llvm::Error::success();
}

} // namespace loom::adg::detail

ModuleBuilder loom::adg::buildMinimalSpatialAdg() {
  ModuleBuilder module("minimal_spatial_adg");
  addVisualLayout(module, {{"mem", 0, 0}, {"pe", 1, 0}, {"switch", 2, 0}});
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
  MemSpec mem(Schedule::Spatial, {"mgr"}, {},
              MemDispatchEligibility{{{0}}, {}});
  mem.loads = {{"addr", "ctrl"}};
  mem.dataWidth = 32;
  module.addMem(std::move(mem));
  return module;
}

ModuleBuilder loom::adg::buildMinimalTemporalAdg() {
  ModuleBuilder module("minimal_temporal_adg");
  addVisualLayout(module, {{"mem", 0, 0}, {"pe", 1, 0}, {"switch", 2, 0}});
  module.addInput("mgr", "memref<?x!fabric.bits<32>>")
      .addInput("lhs", "!fabric.bits_tag<32, 4>")
      .addInput("rhs", "!fabric.bits_tag<32, 4>")
      .addInput("addr", "!fabric.bits_tag<32, 4>")
      .addInput("ctrl", "!fabric.bits_tag<0, 4>");

  TemporalPeConfig temporal;
  temporal.tagWidth = 4;
  temporal.numInstruction = 1;
  temporal.fuConfigMode = "per_fu_config";
  temporal.operandBufferMode = ::fabric::OperandBufferMode::PerInstruction;
  temporal.operandBufferSize = 2;
  module.addPe(makeMinimalAddPe(Schedule::Temporal, "!fabric.bits_tag<32, 4>",
                                "!fabric.bits<32>", std::move(temporal)));

  module.addSwitch(
      SwitchSpec{Schedule::Temporal,
                 {"lhs", "rhs"},
                 {"!fabric.bits_tag<32, 4>", "!fabric.bits_tag<32, 4>"},
                 {"11", "11"},
                 1});

  MemSpec mem(Schedule::Temporal, {"mgr"}, {},
              MemDispatchEligibility{{{0}}, {}});
  mem.loads = {{"addr", "ctrl"}};
  mem.dataWidth = 32;
  mem.temporalTagWidth = 4;
  mem.temporalOperationTableSize = 1;
  module.addMem(std::move(mem));
  return module;
}
