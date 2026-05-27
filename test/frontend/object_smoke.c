// RUN: %loom-cc -c %s -o %t.o && %objdump-h %t.o | FileCheck %s

int answer(void) {
    return 42;
}

// CHECK: .text
