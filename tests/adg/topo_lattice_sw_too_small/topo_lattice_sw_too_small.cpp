//===-- topo_lattice_sw_too_small.cpp - ADG test: switch too small --*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
// Negative test: switch template with < 8 ports should trigger error.

#include <loom/adg.h>

#include <cassert>
#include <sys/wait.h>
#include <unistd.h>

using namespace loom::adg;

static bool expectError(void (*fn)()) {
  pid_t pid = fork();
  if (pid == 0) {
    fn();
    _exit(0);
  }
  int status = 0;
  waitpid(pid, &status, 0);
  return WIFEXITED(status) && WEXITSTATUS(status) != 0;
}

static void switchTooSmall() {
  ADGBuilder builder("test");

  auto sw = builder.newSwitch("small_xbar")
      .setPortCount(5, 5)
      .setType(Type::i32());

  builder.latticeMesh(2, 2, sw);
}

int main() {
  assert(expectError(switchTooSmall) && "switch with < 8 ports should fail");

  // Build a minimal valid ADG so the test harness MLIR check passes.
  {
    ADGBuilder b("topo_lattice_sw_too_small");
    auto pe = b.newPE("p")
        .setLatency(1, 1, 1).setInterval(1, 1, 1)
        .setInputPorts({Type::i32(), Type::i32()})
        .setOutputPorts({Type::i32()})
        .addOp("arith.addi");
    auto inst = b.clone(pe, "i0");
    auto a = b.addModuleInput("a", Type::i32());
    auto c = b.addModuleInput("c", Type::i32());
    auto out = b.addModuleOutput("out", Type::i32());
    b.connectToModuleInput(a, inst, 0);
    b.connectToModuleInput(c, inst, 1);
    b.connectToModuleOutput(inst, 0, out);
    b.exportMLIR("Output/topo_lattice_sw_too_small.fabric.mlir");
  }
  return 0;
}
