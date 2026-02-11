//===-- topo_mesh_4x4.cpp - ADG test: 4x4 mesh topology --------*- C++ -*-===//
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
  ADGBuilder builder("topo_mesh_4x4");

  auto pe = builder.newPE("alu")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto sw = builder.newSwitch("xbar")
      .setPortCount(5, 5)
      .setType(Type::i32());

  auto mesh = builder.buildMesh(4, 4, pe, sw, Topology::Mesh);

  // Chain 16 PEs in row-major order:
  // [0][0] -> [0][1] -> [0][2] -> [0][3] -> [1][0] -> ... -> [3][3]
  auto in0 = builder.addModuleInput("in0", Type::i32());
  auto in1 = builder.addModuleInput("in1", Type::i32());
  auto in2 = builder.addModuleInput("in2", Type::i32());
  auto in3 = builder.addModuleInput("in3", Type::i32());
  auto in4 = builder.addModuleInput("in4", Type::i32());
  auto in5 = builder.addModuleInput("in5", Type::i32());
  auto in6 = builder.addModuleInput("in6", Type::i32());
  auto in7 = builder.addModuleInput("in7", Type::i32());
  auto in8 = builder.addModuleInput("in8", Type::i32());
  auto in9 = builder.addModuleInput("in9", Type::i32());
  auto in10 = builder.addModuleInput("in10", Type::i32());
  auto in11 = builder.addModuleInput("in11", Type::i32());
  auto in12 = builder.addModuleInput("in12", Type::i32());
  auto in13 = builder.addModuleInput("in13", Type::i32());
  auto in14 = builder.addModuleInput("in14", Type::i32());
  auto in15 = builder.addModuleInput("in15", Type::i32());
  auto in16 = builder.addModuleInput("in16", Type::i32());

  // PE[0][0]: first PE gets both inputs from module inputs
  builder.connectToModuleInput(in0, mesh.peGrid[0][0], 0);
  builder.connectToModuleInput(in1, mesh.peGrid[0][0], 1);

  // PE[0][1]
  auto bcast_0 = builder.addModuleInput("bcast_0", Type::i32());
  builder.connectToModuleInput(bcast_0, mesh.peGrid[0][1], 0);
  builder.connectToModuleInput(in2, mesh.peGrid[0][1], 1);
  // PE[0][2]
  auto bcast_1 = builder.addModuleInput("bcast_1", Type::i32());
  builder.connectToModuleInput(bcast_1, mesh.peGrid[0][2], 0);
  builder.connectToModuleInput(in3, mesh.peGrid[0][2], 1);
  // PE[0][3]
  auto bcast_2 = builder.addModuleInput("bcast_2", Type::i32());
  builder.connectToModuleInput(bcast_2, mesh.peGrid[0][3], 0);
  builder.connectToModuleInput(in4, mesh.peGrid[0][3], 1);
  // PE[1][0]
  auto bcast_3 = builder.addModuleInput("bcast_3", Type::i32());
  builder.connectToModuleInput(bcast_3, mesh.peGrid[1][0], 0);
  builder.connectToModuleInput(in5, mesh.peGrid[1][0], 1);
  // PE[1][1]
  auto bcast_4 = builder.addModuleInput("bcast_4", Type::i32());
  builder.connectToModuleInput(bcast_4, mesh.peGrid[1][1], 0);
  builder.connectToModuleInput(in6, mesh.peGrid[1][1], 1);
  // PE[1][2]
  auto bcast_5 = builder.addModuleInput("bcast_5", Type::i32());
  builder.connectToModuleInput(bcast_5, mesh.peGrid[1][2], 0);
  builder.connectToModuleInput(in7, mesh.peGrid[1][2], 1);
  // PE[1][3]
  auto bcast_6 = builder.addModuleInput("bcast_6", Type::i32());
  builder.connectToModuleInput(bcast_6, mesh.peGrid[1][3], 0);
  builder.connectToModuleInput(in8, mesh.peGrid[1][3], 1);
  // PE[2][0]
  auto bcast_7 = builder.addModuleInput("bcast_7", Type::i32());
  builder.connectToModuleInput(bcast_7, mesh.peGrid[2][0], 0);
  builder.connectToModuleInput(in9, mesh.peGrid[2][0], 1);
  // PE[2][1]
  auto bcast_8 = builder.addModuleInput("bcast_8", Type::i32());
  builder.connectToModuleInput(bcast_8, mesh.peGrid[2][1], 0);
  builder.connectToModuleInput(in10, mesh.peGrid[2][1], 1);
  // PE[2][2]
  auto bcast_9 = builder.addModuleInput("bcast_9", Type::i32());
  builder.connectToModuleInput(bcast_9, mesh.peGrid[2][2], 0);
  builder.connectToModuleInput(in11, mesh.peGrid[2][2], 1);
  // PE[2][3]
  auto bcast_10 = builder.addModuleInput("bcast_10", Type::i32());
  builder.connectToModuleInput(bcast_10, mesh.peGrid[2][3], 0);
  builder.connectToModuleInput(in12, mesh.peGrid[2][3], 1);
  // PE[3][0]
  auto bcast_11 = builder.addModuleInput("bcast_11", Type::i32());
  builder.connectToModuleInput(bcast_11, mesh.peGrid[3][0], 0);
  builder.connectToModuleInput(in13, mesh.peGrid[3][0], 1);
  // PE[3][1]
  auto bcast_12 = builder.addModuleInput("bcast_12", Type::i32());
  builder.connectToModuleInput(bcast_12, mesh.peGrid[3][1], 0);
  builder.connectToModuleInput(in14, mesh.peGrid[3][1], 1);
  // PE[3][2]
  auto bcast_13 = builder.addModuleInput("bcast_13", Type::i32());
  builder.connectToModuleInput(bcast_13, mesh.peGrid[3][2], 0);
  builder.connectToModuleInput(in15, mesh.peGrid[3][2], 1);
  // PE[3][3]
  auto bcast_14 = builder.addModuleInput("bcast_14", Type::i32());
  builder.connectToModuleInput(bcast_14, mesh.peGrid[3][3], 0);
  builder.connectToModuleInput(in16, mesh.peGrid[3][3], 1);


  builder.exportMLIR("Output/topo_mesh_4x4.fabric.mlir");

  const char *mlir = "Output/topo_mesh_4x4.fabric.mlir";
  assert(mlirCount(mlir, "fabric.instance") == 16 && "expected 16 PE instances");
  assert(mlirCount(mlir, "fabric.switch") == 16 && "expected 16 switch instances");
  // 5-port switch: connectivity table has 25 entries
  assert(mlirCount(mlir, "connectivity_table") == 16 && "each switch has connectivity_table");
  assert(countInterSwitchEdges(mlir) == 24 && "expected 24 inter-switch edges");

  return 0;
}
