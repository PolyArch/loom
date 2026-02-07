//===-- topo_diag_torus_2x2.cpp - ADG test: 2x2 diagonal torus -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verifies that buildMesh with Topology::DiagonalTorus creates:
//   - 2x2 PE grid and 2x2 switch grid
//   - Internal fifo instances for wraparound (2 EW + 2 NS + 3 SE + 3 SW = 10 fifos)
//   - All PEs and switches connected
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

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
  ADGBuilder builder("topo_diag_torus_2x2");

  auto pe = builder.newPE("alu")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto sw = builder.newSwitch("xbar")
      .setPortCount(9, 9)
      .setType(Type::i32());

  auto mesh = builder.buildMesh(2, 2, pe, sw, Topology::DiagonalTorus);

  // Verify grid dimensions.
  assert(mesh.peGrid.size() == 2 && "expected 2 PE rows");
  assert(mesh.peGrid[0].size() == 2 && "expected 2 PE cols");
  assert(mesh.swGrid.size() == 2 && "expected 2 switch rows");
  assert(mesh.swGrid[0].size() == 2 && "expected 2 switch cols");

  // Chain PEs in row-major order: [0][0] -> [0][1] -> [1][0] -> [1][1]
  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto c = builder.addModuleInput("c", Type::i32());
  auto d = builder.addModuleInput("d", Type::i32());
  auto e = builder.addModuleInput("e", Type::i32());
  auto out = builder.addModuleOutput("result", Type::i32());

  builder.connectToModuleInput(a, mesh.peGrid[0][0], 0);
  builder.connectToModuleInput(b, mesh.peGrid[0][0], 1);
  builder.connectPorts(mesh.peGrid[0][0], 0, mesh.peGrid[0][1], 0);
  builder.connectToModuleInput(c, mesh.peGrid[0][1], 1);
  builder.connectPorts(mesh.peGrid[0][1], 0, mesh.peGrid[1][0], 0);
  builder.connectToModuleInput(d, mesh.peGrid[1][0], 1);
  builder.connectPorts(mesh.peGrid[1][0], 0, mesh.peGrid[1][1], 0);
  builder.connectToModuleInput(e, mesh.peGrid[1][1], 1);
  builder.connectToModuleOutput(mesh.peGrid[1][1], 0, out);

  // Validate ADG before export.
  auto validation = builder.validateADG();
  assert(validation.success && "validation failed");

  builder.exportMLIR("Output/topo_diag_torus_2x2.fabric.mlir");

  // Verify MLIR contains expected instances: 4 PEs and 4 switches (9-port).
  const char *mlir = "Output/topo_diag_torus_2x2.fabric.mlir";
  assert(mlirCount(mlir, "fabric.instance") == 4 && "expected 4 PE instances");
  assert(mlirCount(mlir, "fabric.switch") == 4 && "expected 4 switch instances");
  assert(mlirCount(mlir, "sym_name = \"pe_") == 4 && "expected pe_ sym_names");
  assert(countInterSwitchEdges(mlir) == 6 && "expected 6 inter-switch edges");

  // Verify wraparound uses internal fifo instances (no wrap module I/O).
  assert(mlirCount(mlir, "fabric.fifo") == 10 && "expected 10 fifo instances (2 EW + 2 NS + 3 SE + 3 SW)");
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
