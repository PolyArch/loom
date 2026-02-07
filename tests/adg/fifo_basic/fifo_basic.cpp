//===-- fifo_basic.cpp - ADG test: basic fifo in pipeline ------*- C++ -*-===//
//
// Part of the Loom project.
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
  ADGBuilder builder("fifo_basic");

  auto pe = builder.newPE("adder")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto fifo = builder.newFifo("buf");
  fifo.setDepth(4).setType(Type::i32());

  auto p0 = builder.clone(pe, "p0");
  auto p1 = builder.clone(pe, "p1");
  auto f0 = builder.clone(fifo, "f0");

  auto in0 = builder.addModuleInput("in0", Type::i32());
  auto in1 = builder.addModuleInput("in1", Type::i32());
  auto in2 = builder.addModuleInput("in2", Type::i32());
  auto out = builder.addModuleOutput("result", Type::i32());

  // Pipeline: in0,in1 -> p0 -> f0 -> p1 -> out
  builder.connectToModuleInput(in0, p0, 0);
  builder.connectToModuleInput(in1, p0, 1);
  builder.connectPorts(p0, 0, f0, 0);
  builder.connectPorts(f0, 0, p1, 0);
  builder.connectToModuleInput(in2, p1, 1);
  builder.connectToModuleOutput(p1, 0, out);

  auto validation = builder.validateADG();
  assert(validation.success && "validation failed");

  builder.exportMLIR("Output/fifo_basic.fabric.mlir");

  const char *mlir = "Output/fifo_basic.fabric.mlir";
  assert(mlirCount(mlir, "fabric.instance") == 2 && "expected 2 PE instances");
  assert(mlirCount(mlir, "fabric.fifo") == 1 && "expected 1 fifo instance");
  assert(mlirCount(mlir, "depth = 4") == 1 && "expected depth = 4");

  return 0;
}
