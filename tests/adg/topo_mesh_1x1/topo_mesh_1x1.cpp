//===-- topo_mesh_1x1.cpp - ADG test: minimal 1x1 mesh ---------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

#include <cassert>
#include <fstream>
#include <regex>
#include <set>
#include <string>

static unsigned mlirCount(const std::string &path, const std::string &substr) {
  std::ifstream f(path);
  std::string content((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
  unsigned count = 0;
  size_t pos = 0;
  while ((pos = content.find(substr, pos)) != std::string::npos) {
    ++count;
    pos += substr.size();
  }
  return count;
}

// Count direct inter-switch edges: switch result used as operand to another switch.
static unsigned countInterSwitchEdges(const std::string &path) {
  std::ifstream f(path);
  std::string content((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
  // Collect SSA names defined by fabric.switch lines (%N:K = fabric.switch).
  std::regex swDef(R"((%\d+):\d+\s*=\s*fabric\.switch)");
  std::set<std::string> swNames;
  for (std::sregex_iterator it(content.begin(), content.end(), swDef), end; it != end; ++it)
    swNames.insert((*it)[1].str());
  // Count operands to fabric.switch that reference a switch result (%N#M).
  unsigned edges = 0;
  std::regex swLine(R"(fabric\.switch\s*\[.*?\]\s*(.*?)\s*:)");
  std::regex operand(R"((%\d+)#\d+)");
  for (std::sregex_iterator it(content.begin(), content.end(), swLine), end; it != end; ++it) {
    std::string ops = (*it)[1].str();
    for (std::sregex_iterator oit(ops.begin(), ops.end(), operand), oend; oit != oend; ++oit)
      if (swNames.count((*oit)[1].str()))
        ++edges;
  }
  return edges;
}

using namespace loom::adg;

int main() {
  ADGBuilder builder("topo_mesh_1x1");

  auto pe = builder.newPE("alu")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto sw = builder.newSwitch("xbar")
      .setPortCount(5, 5)
      .setType(Type::i32());

  auto mesh = builder.buildMesh(1, 1, pe, sw, Topology::Mesh);

  // Single PE: module inputs -> PE[0][0] -> module output
  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());

  builder.connectToModuleInput(a, mesh.peGrid[0][0], 0);
  builder.connectToModuleInput(b, mesh.peGrid[0][0], 1);

  builder.exportMLIR("Output/topo_mesh_1x1.fabric.mlir");

  const char *mlir = "Output/topo_mesh_1x1.fabric.mlir";
  assert(mlirCount(mlir, "fabric.instance") == 1 && "expected 1 PE instances");
  assert(mlirCount(mlir, "fabric.switch") == 1 && "expected 1 switch instances");
  // 5-port switch: connectivity table has 25 entries
  assert(mlirCount(mlir, "connectivity_table") == 1 && "each switch has connectivity_table");
  assert(countInterSwitchEdges(mlir) == 0 && "expected 0 inter-switch edges");

  return 0;
}
