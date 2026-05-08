// RUN: %loom-cc -emit-llvm -S %s -o - | FileCheck %s

#include <stdio.h>

int main(void) {
    printf("hello\n");
    return 0;
}

// CHECK: define {{.*}} @main
// CHECK: call {{.*}} @printf
