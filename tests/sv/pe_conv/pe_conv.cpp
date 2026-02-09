//===-- pe_conv.cpp - SV test: mixed-width PE with conversions --*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Tests multi-op PE body with arith.extsi and arith.trunci conversion
// operations. Exercises the width-handling path in ADGExportSV for mixed-width
// pipelines: i16 input is sign-extended to i32, added with an i32 input, then
// truncated back to i16.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("pe_conv");

  // Mixed-width PE: (i16, i32) -> i16
  // Body: %ext = arith.extsi %a : i16 to i32
  //       %sum = arith.addi %ext, %b : i32
  //       %out = arith.trunci %sum : i32 to i16
  auto pe = builder.newPE("conv_pe")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i16(), Type::i32()})
      .setOutputPorts({Type::i16()})
      .setBodyMLIR(
          "^bb0(%a: i16, %b: i32):\n"
          "  %ext = arith.extsi %a : i16 to i32\n"
          "  %sum = arith.addi %ext, %b : i32\n"
          "  %out = arith.trunci %sum : i32 to i16\n"
          "  fabric.yield %out : i16\n");

  auto inst = builder.clone(pe, "conv0");

  // Second PE using arith.extui (zero-extension) instead of arith.extsi
  auto pe_u = builder.newPE("conv_pe_u")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i16(), Type::i32()})
      .setOutputPorts({Type::i16()})
      .setBodyMLIR(
          "^bb0(%a: i16, %b: i32):\n"
          "  %ext = arith.extui %a : i16 to i32\n"
          "  %sum = arith.addi %ext, %b : i32\n"
          "  %out = arith.trunci %sum : i32 to i16\n"
          "  fabric.yield %out : i16\n");
  auto inst_u = builder.clone(pe_u, "conv_u0");

  auto in0 = builder.addModuleInput("in0", Type::i16());
  auto in1 = builder.addModuleInput("in1", Type::i32());
  auto out = builder.addModuleOutput("out", Type::i16());
  auto out_u = builder.addModuleOutput("out_u", Type::i16());

  builder.connectToModuleInput(in0, inst, 0);
  builder.connectToModuleInput(in1, inst, 1);
  builder.connectToModuleOutput(inst, 0, out);

  builder.connectToModuleInput(in0, inst_u, 0);
  builder.connectToModuleInput(in1, inst_u, 1);
  builder.connectToModuleOutput(inst_u, 0, out_u);

  builder.exportMLIR("Output/pe_conv.fabric.mlir");
  builder.exportSV("Output/sv");
  return 0;
}
