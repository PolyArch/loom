//===-- topo_lattice_3x3.cpp - ADG test: 3x3 lattice mesh ------*- C++ -*-===//
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

static unsigned countInterSwitchEdges(const std::string &path) {
  std::ifstream f(path);
  std::string content((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
  std::regex swDef(R"((%\d+):\d+\s*=\s*fabric\.switch)");
  std::set<std::string> swNames;
  for (std::sregex_iterator it(content.begin(), content.end(), swDef), end;
       it != end; ++it)
    swNames.insert((*it)[1].str());
  unsigned edges = 0;
  std::regex swLine(R"(fabric\.switch\s*\[.*?\]\s*(.*?)\s*:)");
  std::regex operand(R"((%\d+)#\d+)");
  for (std::sregex_iterator it(content.begin(), content.end(), swLine), end;
       it != end; ++it) {
    std::string ops = (*it)[1].str();
    for (std::sregex_iterator oit(ops.begin(), ops.end(), operand), oend;
         oit != oend; ++oit)
      if (swNames.count((*oit)[1].str()))
        ++edges;
  }
  return edges;
}

using namespace loom::adg;

int main() {
  ADGBuilder builder("topo_lattice_3x3");

  auto pe = builder.newPE("alu")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto sw = builder.newSwitch("xbar")
      .setPortCount(8, 8)
      .setType(Type::i32());

  auto lattice = builder.latticeMesh(3, 3, sw);

  // 3x3 PE grid -> 4x4 switch grid = 16 switches
  assert(lattice.peRows == 3);
  assert(lattice.peCols == 3);

  // Place all 9 PEs
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      builder.placePEInLattice(lattice, r, c, pe,
          "pe_" + std::to_string(r) + "_" + std::to_string(c));

  builder.finalizeLattice(lattice);
  builder.exportMLIR("Output/topo_lattice_3x3.fabric.mlir");

  const char *mlir = "Output/topo_lattice_3x3.fabric.mlir";
  assert(mlirCount(mlir, "fabric.instance") == 9 && "expected 9 PE instances");
  assert(mlirCount(mlir, "fabric.switch") == 16 && "expected 16 switch instances");

  // Inter-switch edges: 4x4 grid has 3*4 east + 4*3 south = 24 edges
  assert(countInterSwitchEdges(mlir) == 24 && "expected 24 inter-switch edges");

  return 0;
}
