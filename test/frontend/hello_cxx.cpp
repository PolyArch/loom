// RUN: %loom-c++ -emit-llvm -S %s -o - | FileCheck %s

#include <cstdio>

int main() {
    std::printf("hello\n");
    return 0;
}

// CHECK: define {{.*}} @main
// printf is declared extern "C", so the symbol must remain unmangled.
// CHECK: declare {{.*}} @printf(
// CHECK-NOT: declare {{.*}} @_Z{{.*}}printf
