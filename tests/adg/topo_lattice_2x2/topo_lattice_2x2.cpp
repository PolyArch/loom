//===-- topo_lattice_2x2.cpp - ADG test: 2x2 lattice mesh ------*- C++ -*-===//
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
  ADGBuilder builder("topo_lattice_2x2");

  auto pe = builder.newPE("alu")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto sw = builder.newSwitch("xbar")
      .setPortCount(8, 8)
      .setType(Type::i32());

  auto lattice = builder.latticeMesh(2, 2, sw);

  // 2x2 PE grid -> 3x3 switch grid = 9 switches
  assert(lattice.peRows == 2);
  assert(lattice.peCols == 2);

  // Place all 4 PEs
  builder.placePEInLattice(lattice, 0, 0, pe, "pe_0_0");
  builder.placePEInLattice(lattice, 0, 1, pe, "pe_0_1");
  builder.placePEInLattice(lattice, 1, 0, pe, "pe_1_0");
  builder.placePEInLattice(lattice, 1, 1, pe, "pe_1_1");

  builder.finalizeLattice(lattice);
  builder.exportMLIR("Output/topo_lattice_2x2.fabric.mlir");

  const char *mlir = "Output/topo_lattice_2x2.fabric.mlir";
  assert(mlirCount(mlir, "fabric.instance") == 4 && "expected 4 PE instances");
  assert(mlirCount(mlir, "fabric.switch") == 9 && "expected 9 switch instances");

  // Inter-switch edges: 3x3 grid has 2*3 east + 3*2 south = 12 edges
  assert(countInterSwitchEdges(mlir) == 12 && "expected 12 inter-switch edges");

  return 0;
}
