//===-- fifo_type_alias.cpp - ADG test: fifo with Type::iN aliases -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Validates that ADG builder accepts Type::iN(32) as a valid FIFO element type,
// since it is semantically equivalent to Type::i32().
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

#include <cassert>
#include <fstream>
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

using namespace loom::adg;

int main() {
  ADGBuilder builder("fifo_type_alias");

  auto pe = builder.newPE("adder")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  // Use Type::iN(32) instead of Type::i32() -- semantically equivalent.
  auto fifo = builder.newFifo("buf");
  fifo.setDepth(2).setType(Type::iN(32));

  auto p0 = builder.clone(pe, "p0");
  auto f0 = builder.clone(fifo, "f0");

  auto in0 = builder.addModuleInput("in0", Type::i32());
  auto in1 = builder.addModuleInput("in1", Type::i32());
  auto out = builder.addModuleOutput("result", Type::i32());

  builder.connectToModuleInput(in0, p0, 0);
  builder.connectToModuleInput(in1, p0, 1);
  builder.connectPorts(p0, 0, f0, 0);
  builder.connectToModuleOutput(f0, 0, out);

  auto validation = builder.validateADG();
  assert(validation.success && "Type::iN(32) fifo should pass validation");

  builder.exportMLIR("Output/fifo_type_alias.fabric.mlir");

  const char *mlir = "Output/fifo_type_alias.fabric.mlir";
  assert(mlirCount(mlir, "fabric.fifo") == 1 && "expected 1 fifo instance");
  assert(mlirCount(mlir, "depth = 2") == 1 && "expected depth = 2");

  return 0;
}
