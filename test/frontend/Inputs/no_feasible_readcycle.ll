target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-unknown-elf"

declare i64 @llvm.readcyclecounter()

define i32 @main() {
entry:
  %cycles = call i64 @llvm.readcyclecounter()
  %result = trunc i64 %cycles to i32
  ret i32 %result
}
