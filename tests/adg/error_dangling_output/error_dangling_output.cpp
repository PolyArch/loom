//===-- error_dangling_output.cpp - ADG test: dangling output detection -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verifies that validateADG detects PE instances with unused output ports
// (CPL_OUTPUT_DANGLING).
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

#include <cassert>

using namespace loom::adg;

int main() {
  ADGBuilder builder("error_dangling_output");

  auto pe = builder.newPE("alu")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  // Create two PE instances. Only connect the second PE's output to module out.
  // The first PE's output feeds the second PE, so it is connected.
  // The second PE's output goes to module out, so it is connected.
  auto inst0 = builder.clone(pe, "pe0");
  auto inst1 = builder.clone(pe, "pe1");

  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto c = builder.addModuleInput("c", Type::i32());
  auto out = builder.addModuleOutput("out", Type::i32());

  builder.connectToModuleInput(a, inst0, 0);
  builder.connectToModuleInput(b, inst0, 1);
  builder.connectPorts(inst0, 0, inst1, 0);
  builder.connectToModuleInput(c, inst1, 1);
  builder.connectToModuleOutput(inst1, 0, out);

  // This should pass: all PE outputs are connected.
  auto v1 = builder.validateADG();
  assert(v1.success && "valid chain should pass validation");

  // Now create a third PE instance with a dangling output.
  ADGBuilder builder2("error_dangling_output_bad");
  auto pe2 = builder2.newPE("alu")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto i0 = builder2.clone(pe2, "pe0");
  auto i1 = builder2.clone(pe2, "pe1_dangling");

  auto a2 = builder2.addModuleInput("a", Type::i32());
  auto b2 = builder2.addModuleInput("b", Type::i32());
  auto c2 = builder2.addModuleInput("c", Type::i32());
  auto d2 = builder2.addModuleInput("d", Type::i32());
  auto out2 = builder2.addModuleOutput("out", Type::i32());

  builder2.connectToModuleInput(a2, i0, 0);
  builder2.connectToModuleInput(b2, i0, 1);
  builder2.connectToModuleInput(c2, i1, 0);
  builder2.connectToModuleInput(d2, i1, 1);
  builder2.connectToModuleOutput(i0, 0, out2);
  // i1's output port 0 is NOT connected (dangling).

  auto v2 = builder2.validateADG();
  assert(!v2.success && "dangling PE output should fail validation");

  bool foundDangling = false;
  for (const auto &e : v2.errors) {
    if (e.code == "CPL_OUTPUT_DANGLING")
      foundDangling = true;
  }
  assert(foundDangling && "should report CPL_OUTPUT_DANGLING error");

  // Export the valid ADG so the test harness MLIR check passes.
  builder.exportMLIR("Output/error_dangling_output.fabric.mlir");
  return 0;
}
