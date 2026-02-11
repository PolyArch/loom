//===-- topo_torus_3x3.cpp - ADG test: 3x3 torus topology ------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verifies that buildMesh with Topology::Torus creates:
//   - 3x3 PE grid and 3x3 switch grid
//   - Internal fifo instances for wraparound (3 EW + 3 NS = 6 fifos)
//   - All PEs and switches connected
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

#include <algorithm>
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
  ADGBuilder builder("topo_torus_3x3");

  auto pe = builder.newPE("alu")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto sw = builder.newSwitch("xbar")
      .setPortCount(5, 5)
      .setType(Type::i32());

  auto mesh = builder.buildMesh(3, 3, pe, sw, Topology::Torus);

  // Verify grid dimensions.
  assert(mesh.peGrid.size() == 3 && "expected 3 PE rows");
  assert(mesh.peGrid[0].size() == 3 && "expected 3 PE cols");
  assert(mesh.swGrid.size() == 3 && "expected 3 switch rows");
  assert(mesh.swGrid[0].size() == 3 && "expected 3 switch cols");

  // Chain PEs in row-major order:
  // [0][0] -> [0][1] -> [0][2] -> [1][0] -> [1][1] -> [1][2] ->
  // [2][0] -> [2][1] -> [2][2]
  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto c = builder.addModuleInput("c", Type::i32());
  auto d = builder.addModuleInput("d", Type::i32());
  auto e = builder.addModuleInput("e", Type::i32());
  auto f = builder.addModuleInput("f", Type::i32());
  auto g = builder.addModuleInput("g", Type::i32());
  auto h = builder.addModuleInput("h", Type::i32());
  auto i = builder.addModuleInput("i", Type::i32());
  auto j = builder.addModuleInput("j", Type::i32());

  builder.connectToModuleInput(a, mesh.peGrid[0][0], 0);
  builder.connectToModuleInput(b, mesh.peGrid[0][0], 1);
  auto bcast_0 = builder.addModuleInput("bcast_0", Type::i32());
  builder.connectToModuleInput(bcast_0, mesh.peGrid[0][1], 0);
  builder.connectToModuleInput(c, mesh.peGrid[0][1], 1);
  auto bcast_1 = builder.addModuleInput("bcast_1", Type::i32());
  builder.connectToModuleInput(bcast_1, mesh.peGrid[0][2], 0);
  builder.connectToModuleInput(d, mesh.peGrid[0][2], 1);
  auto bcast_2 = builder.addModuleInput("bcast_2", Type::i32());
  builder.connectToModuleInput(bcast_2, mesh.peGrid[1][0], 0);
  builder.connectToModuleInput(e, mesh.peGrid[1][0], 1);
  auto bcast_3 = builder.addModuleInput("bcast_3", Type::i32());
  builder.connectToModuleInput(bcast_3, mesh.peGrid[1][1], 0);
  builder.connectToModuleInput(f, mesh.peGrid[1][1], 1);
  auto bcast_4 = builder.addModuleInput("bcast_4", Type::i32());
  builder.connectToModuleInput(bcast_4, mesh.peGrid[1][2], 0);
  builder.connectToModuleInput(g, mesh.peGrid[1][2], 1);
  auto bcast_5 = builder.addModuleInput("bcast_5", Type::i32());
  builder.connectToModuleInput(bcast_5, mesh.peGrid[2][0], 0);
  builder.connectToModuleInput(h, mesh.peGrid[2][0], 1);
  auto bcast_6 = builder.addModuleInput("bcast_6", Type::i32());
  builder.connectToModuleInput(bcast_6, mesh.peGrid[2][1], 0);
  builder.connectToModuleInput(i, mesh.peGrid[2][1], 1);
  auto bcast_7 = builder.addModuleInput("bcast_7", Type::i32());
  builder.connectToModuleInput(bcast_7, mesh.peGrid[2][2], 0);
  builder.connectToModuleInput(j, mesh.peGrid[2][2], 1);

  // Validate ADG before export.
  auto validation = builder.validateADG();
  assert(validation.success && "validation failed");

  builder.exportMLIR("Output/topo_torus_3x3.fabric.mlir");

  // Verify MLIR contains expected instances: 9 PEs and 9 switches.
  const char *mlir = "Output/topo_torus_3x3.fabric.mlir";
  assert(mlirCount(mlir, "fabric.instance") == 9 && "expected 9 PE instances");
  assert(mlirCount(mlir, "fabric.switch") == 9 && "expected 9 switch instances");
  assert(mlirCount(mlir, "sym_name = \"pe_") == 9 && "expected pe_ sym_names");
  assert(countInterSwitchEdges(mlir) == 12 && "expected 12 inter-switch edges");

  // Verify wraparound uses internal fifo instances (no wrap module I/O).
  assert(mlirCount(mlir, "fabric.fifo") == 6 && "expected 6 fifo instances (3 EW + 3 NS)");
  auto outNames = builder.getModuleOutputNames();
  auto inNames = builder.getModuleInputNames();
  auto countMatching = [](const std::vector<std::string> &names,
                          const std::string &substr) -> unsigned {
    return std::count_if(names.begin(), names.end(),
        [&](const std::string &n) { return n.find(substr) != std::string::npos; });
  };
  // No wrap module I/O -- wraparound is handled by internal fifos.
  assert(countMatching(outNames, "wrap_") == 0 && "no wrap module outputs");
  assert(countMatching(inNames, "wrap_") == 0 && "no wrap module inputs");

  return 0;
}
