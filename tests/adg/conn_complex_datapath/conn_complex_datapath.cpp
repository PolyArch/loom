//===-- conn_complex_datapath.cpp - ADG test: heterogeneous datapath -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("conn_complex_datapath");

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

  auto sw = builder.newSwitch("sw_2x2")
      .setPortCount(2, 2)
      .setType(Type::i32());

  auto cpe = builder.newConstantPE("const_i32")
      .setOutputType(Type::i32());

  // Build: const -> add(input, const) -> switch -> mul -> output
  auto c0 = builder.clone(cpe, "c0");
  auto add0 = builder.clone(adder, "add0");
  auto sw0 = builder.clone(sw, "sw0");
  auto mul0 = builder.clone(mul, "mul0");
  auto add1 = builder.clone(adder, "add1");

  auto ctrl = builder.addModuleInput("ctrl", Type::none());
  auto x = builder.addModuleInput("x", Type::i32());
  auto y = builder.addModuleInput("y", Type::i32());
  auto out0 = builder.addModuleOutput("result0", Type::i32());
  auto out1 = builder.addModuleOutput("result1", Type::i32());

  builder.connectToModuleInput(ctrl, c0, 0);
  builder.connectPorts(c0, 0, add0, 0);
  builder.connectToModuleInput(x, add0, 1);
  builder.connectPorts(add0, 0, sw0, 0);
  builder.connectToModuleInput(y, sw0, 1);
  builder.connectPorts(sw0, 0, mul0, 0);
  builder.connectPorts(sw0, 1, mul0, 1);
  builder.connectToModuleOutput(mul0, 0, out0);
  builder.connectPorts(add0, 0, add1, 0);
  builder.connectPorts(mul0, 0, add1, 1);
  builder.connectToModuleOutput(add1, 0, out1);

  builder.exportMLIR("Output/conn_complex_datapath.fabric.mlir");
  return 0;
}
