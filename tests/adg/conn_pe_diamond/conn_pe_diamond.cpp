//===-- conn_pe_diamond.cpp - ADG test: diamond pattern -------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("conn_pe_diamond");

  auto adder = builder.newPE("adder")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto mul = builder.newPE("mul")
      .setLatency(2, 2, 2)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.muli");

  // Diamond: add0 fans out to mul0 and mul1, then add1 merges
  auto add0 = builder.clone(adder, "add0");
  auto mul0 = builder.clone(mul, "mul0");
  auto mul1 = builder.clone(mul, "mul1");
  auto add1 = builder.clone(adder, "add1");

  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto c = builder.addModuleInput("c", Type::i32());
  auto d = builder.addModuleInput("d", Type::i32());
  auto out = builder.addModuleOutput("result", Type::i32());

  builder.connectToModuleInput(a, add0, 0);
  builder.connectToModuleInput(b, add0, 1);
  auto bcast_0_sw_def = builder.newSwitch("bcast_0_sw")
      .setPortCount(1, 2)
      .setType(Type::i32());
  auto bcast_0 = builder.clone(bcast_0_sw_def, "bcast_0");
  builder.connectPorts(add0, 0, bcast_0, 0);
  builder.connectPorts(bcast_0, 0, mul0, 0);
  builder.connectPorts(bcast_0, 1, mul1, 0);
  builder.connectToModuleInput(c, mul0, 1);
  builder.connectToModuleInput(d, mul1, 1);
  builder.connectPorts(mul0, 0, add1, 0);
  builder.connectPorts(mul1, 0, add1, 1);
  builder.connectToModuleOutput(add1, 0, out);

  builder.exportMLIR("Output/conn_pe_diamond.fabric.mlir");
  return 0;
}
