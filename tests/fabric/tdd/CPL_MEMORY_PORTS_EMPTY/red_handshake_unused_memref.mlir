// RUN: mkdir -p %S/Output
// RUN: rm -f %S/Output/red_handshake_unused_memref.llvm.ll %S/Output/red_handshake_unused_memref.handshake.mlir
// RUN: not loom %S/unused_memref.cpp -I %S/../../../../include -o %S/Output/red_handshake_unused_memref.llvm.ll 2>&1 | FileCheck %s
// CHECK: CPL_MEMORY_PORTS_EMPTY
