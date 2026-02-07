//===-- conn_pe_switch_pe.cpp - ADG test: PE -> switch -> PE ---*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("conn_pe_switch_pe");

  auto adder = builder.newPE("adder")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto sw = builder.newSwitch("sw_2x2")
      .setPortCount(2, 2)
      .setType(Type::i32());

  auto add0 = builder.clone(adder, "add0");
  auto sw0 = builder.clone(sw, "sw0");
  auto add1 = builder.clone(adder, "add1");

  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto c = builder.addModuleInput("c", Type::i32());
  auto out = builder.addModuleOutput("result", Type::i32());

  builder.connectToModuleInput(a, add0, 0);
  builder.connectToModuleInput(b, add0, 1);
  builder.connectPorts(add0, 0, sw0, 0);
  builder.connectToModuleInput(c, sw0, 1);
  builder.connectPorts(sw0, 0, add1, 0);
  builder.connectPorts(sw0, 1, add1, 1);
  builder.connectToModuleOutput(add1, 0, out);

  builder.exportMLIR("Output/conn_pe_switch_pe.fabric.mlir");
  return 0;
}
