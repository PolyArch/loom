//===-- pe_graph.cpp - SV test: multi-PE compute graph ----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("pe_graph");

  auto mul = builder.newPE("pe_mul")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.muli");

  auto add = builder.newPE("pe_add")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto sw = builder.newSwitch("sw_sel")
      .setPortCount(2, 2)
      .setType(Type::i32());

  auto mul0 = builder.clone(mul, "mul0");
  auto add0 = builder.clone(add, "add0");
  auto sw0 = builder.clone(sw, "sw0");

  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto c = builder.addModuleInput("c", Type::i32());
  auto d = builder.addModuleInput("d", Type::i32());

  auto out0 = builder.addModuleOutput("out0", Type::i32());
  auto out1 = builder.addModuleOutput("out1", Type::i32());

  // Two independent PE branches:
  //   branch0 = a * b
  //   branch1 = c + d
  builder.connectToModuleInput(a, mul0, 0);
  builder.connectToModuleInput(b, mul0, 1);
  builder.connectToModuleInput(c, add0, 0);
  builder.connectToModuleInput(d, add0, 1);

  // Route branch outputs through switch for runtime-selectable mapping.
  builder.connectPorts(mul0, 0, sw0, 0);
  builder.connectPorts(add0, 0, sw0, 1);
  builder.connectToModuleOutput(sw0, 0, out0);
  builder.connectToModuleOutput(sw0, 1, out1);

  builder.exportMLIR("Output/pe_graph.fabric.mlir");
  builder.exportSV("Output/sv");
  return 0;
}
