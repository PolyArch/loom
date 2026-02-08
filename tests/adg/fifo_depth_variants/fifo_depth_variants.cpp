//===-- fifo_depth_variants.cpp - ADG test: various fifo depths -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

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
  ADGBuilder builder("fifo_depth_variants");

  auto pe = builder.newPE("adder")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto fifo1 = builder.newFifo("buf1");
  fifo1.setDepth(1).setType(Type::i32());
  auto fifo2 = builder.newFifo("buf2");
  fifo2.setDepth(2).setType(Type::i32());
  auto fifo16 = builder.newFifo("buf16");
  fifo16.setDepth(16).setType(Type::i32());

  auto p0 = builder.clone(pe, "p0");
  auto p1 = builder.clone(pe, "p1");
  auto p2 = builder.clone(pe, "p2");
  auto p3 = builder.clone(pe, "p3");
  auto f1 = builder.clone(fifo1, "f_d1");
  auto f2 = builder.clone(fifo2, "f_d2");
  auto f16 = builder.clone(fifo16, "f_d16");

  auto in0 = builder.addModuleInput("in0", Type::i32());
  auto in1 = builder.addModuleInput("in1", Type::i32());
  auto in2 = builder.addModuleInput("in2", Type::i32());
  auto in3 = builder.addModuleInput("in3", Type::i32());
  auto in4 = builder.addModuleInput("in4", Type::i32());
  auto out = builder.addModuleOutput("result", Type::i32());

  // Pipeline: p0 -> f1(depth=1) -> p1 -> f2(depth=2) -> p2 -> f16(depth=16) -> p3
  builder.connectToModuleInput(in0, p0, 0);
  builder.connectToModuleInput(in1, p0, 1);
  builder.connectPorts(p0, 0, f1, 0);
  builder.connectPorts(f1, 0, p1, 0);
  builder.connectToModuleInput(in2, p1, 1);
  builder.connectPorts(p1, 0, f2, 0);
  builder.connectPorts(f2, 0, p2, 0);
  builder.connectToModuleInput(in3, p2, 1);
  builder.connectPorts(p2, 0, f16, 0);
  builder.connectPorts(f16, 0, p3, 0);
  builder.connectToModuleInput(in4, p3, 1);
  builder.connectToModuleOutput(p3, 0, out);

  auto validation = builder.validateADG();
  assert(validation.success && "validation failed");

  builder.exportMLIR("Output/fifo_depth_variants.fabric.mlir");

  const char *mlir = "Output/fifo_depth_variants.fabric.mlir";
  assert(mlirCount(mlir, "fabric.instance") == 4 && "expected 4 PE instances");
  assert(mlirCount(mlir, "fabric.fifo") == 3 && "expected 3 fifo instances");
  assert(mlirCount(mlir, "depth = 1]") == 1 && "expected depth = 1");
  assert(mlirCount(mlir, "depth = 2]") == 1 && "expected depth = 2");
  assert(mlirCount(mlir, "depth = 16]") == 1 && "expected depth = 16");

  return 0;
}
