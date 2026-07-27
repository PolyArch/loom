target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-unknown-elf"

%struct.transform = type { i16, ptr, ptr, i16 }

@transform = external constant %struct.transform

define i16 @read_transform_field() {
entry:
  %value = load i16, ptr getelementptr inbounds nuw (i8, ptr @transform, i64 24), align 8
  ret i16 %value
}
