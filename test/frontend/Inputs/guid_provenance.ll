; The LTO pre-link pipeline assigns each definition a GUID; a local-linkage
; GUID hashes the originating file, so two checkouts of one program differ
; only in these link-time identities.
source_filename = "/first/tree/guid_provenance.c"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-unknown-elf"

define internal i32 @accumulate(i32 %value) !guid !0 {
  %sum = add i32 %value, 1
  ret i32 %sum
}

define i32 @entry(i32 %value) !guid !1 {
  %result = call i32 @accumulate(i32 %value)
  ret i32 %result
}

!0 = !{i64 3358523359079058097}
!1 = !{i64 -2624081020897602054}
