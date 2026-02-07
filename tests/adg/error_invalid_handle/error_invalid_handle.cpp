//===-- error_invalid_handle.cpp - ADG test: invalid handle detection -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verifies that the ADG builder rejects invalid handle IDs with deterministic
// runtime errors (not segfaults). Each invalid operation is tested in a
// subprocess to confirm non-zero exit without crashing.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

#include <cassert>
#include <cstdlib>
#include <sys/wait.h>
#include <unistd.h>

using namespace loom::adg;

/// Run a callable in a forked subprocess and return true if it exits non-zero
/// (indicating the builder correctly rejected the operation).
static bool expectError(void (*fn)()) {
  pid_t pid = fork();
  if (pid == 0) {
    // Child: run the test function. If it returns, exit 0.
    fn();
    _exit(0);
  }
  int status = 0;
  waitpid(pid, &status, 0);
  // We expect the child to have exited with non-zero status (builderError
  // calls std::exit(1)).
  return WIFEXITED(status) && WEXITSTATUS(status) != 0;
}

static void cloneInvalidPE() {
  ADGBuilder b("test");
  b.clone(PEHandle{999}, "bad");
}

static void cloneInvalidSwitch() {
  ADGBuilder b("test");
  b.clone(SwitchHandle{999}, "bad");
}

static void cloneInvalidMemory() {
  ADGBuilder b("test");
  b.clone(MemoryHandle{999}, "bad");
}

static void cloneInvalidExtMemory() {
  ADGBuilder b("test");
  b.clone(ExtMemoryHandle{999}, "bad");
}

static void cloneInvalidConstantPE() {
  ADGBuilder b("test");
  b.clone(ConstantPEHandle{999}, "bad");
}

static void cloneInvalidLoadPE() {
  ADGBuilder b("test");
  b.clone(LoadPEHandle{999}, "bad");
}

static void cloneInvalidStorePE() {
  ADGBuilder b("test");
  b.clone(StorePEHandle{999}, "bad");
}

static void cloneInvalidTemporalPE() {
  ADGBuilder b("test");
  b.clone(TemporalPEHandle{999}, "bad");
}

static void cloneInvalidTemporalSwitch() {
  ADGBuilder b("test");
  b.clone(TemporalSwitchHandle{999}, "bad");
}

static void cloneTagOpViaModuleHandle() {
  ADGBuilder b("test");
  auto tag = b.newAddTag("t").setValueType(Type::i32()).setTagType(Type::iN(4));
  // Tag ops auto-instantiate; try to clone via ModuleHandle.
  AddTagHandle th{0};
  ModuleHandle mh(th);
  b.clone(mh, "bad");
}

static void buildMeshInvalidSwitch() {
  ADGBuilder b("test");
  auto pe = b.newPE("p")
      .setLatency(1,1,1).setInterval(1,1,1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");
  b.buildMesh(2, 2, pe, SwitchHandle{42}, Topology::Mesh);
}

static void buildMeshDiagInsuffPorts() {
  ADGBuilder b("test");
  auto pe = b.newPE("p")
      .setLatency(1,1,1).setInterval(1,1,1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");
  auto sw = b.newSwitch("s").setPortCount(5, 5).setType(Type::i32());
  // DiagonalMesh requires >= 7 ports; this switch only has 5.
  b.buildMesh(2, 2, pe, sw, Topology::DiagonalMesh);
}

int main() {
  assert(expectError(cloneInvalidPE) && "clone(PEHandle{999}) should fail");
  assert(expectError(cloneInvalidSwitch) && "clone(SwitchHandle{999}) should fail");
  assert(expectError(cloneInvalidMemory) && "clone(MemoryHandle{999}) should fail");
  assert(expectError(cloneInvalidExtMemory) && "clone(ExtMemoryHandle{999}) should fail");
  assert(expectError(cloneInvalidConstantPE) && "clone(ConstantPEHandle{999}) should fail");
  assert(expectError(cloneInvalidLoadPE) && "clone(LoadPEHandle{999}) should fail");
  assert(expectError(cloneInvalidStorePE) && "clone(StorePEHandle{999}) should fail");
  assert(expectError(cloneInvalidTemporalPE) && "clone(TemporalPEHandle{999}) should fail");
  assert(expectError(cloneInvalidTemporalSwitch) && "clone(TemporalSwitchHandle{999}) should fail");
  assert(expectError(cloneTagOpViaModuleHandle) && "clone(ModuleHandle{AddTag}) should fail");
  assert(expectError(buildMeshInvalidSwitch) && "buildMesh with invalid switch should fail");
  assert(expectError(buildMeshDiagInsuffPorts) && "diagonal topology with 5-port switch should fail");

  // Build a minimal valid ADG and export so the test harness MLIR check passes.
  {
    ADGBuilder b("error_invalid_handle");
    auto pe = b.newPE("p")
        .setLatency(1,1,1).setInterval(1,1,1)
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
    b.exportMLIR("Output/error_invalid_handle.fabric.mlir");
  }
  return 0;
}
