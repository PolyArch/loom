#include "ExecutionMatrixGuestPrograms.h"

namespace loom::system_test {

llvm::StringRef orderedChannelHostProgramSource() {
  return R"asm(
.section .text,"ax",@progbits
.align 2
.globl loom_host_entry
.type loom_host_entry,@function
loom_host_entry:
  mv s0, a0
  mv s1, a2
  mv s2, a3
  mv s3, a1
  li t0, 4
  bne s3, t0, host_fail
  li t0, 1
  bne s2, t0, host_fail
  lw t0, 0(s1)
  li t1, 0x494d474c
  bne t0, t1, host_fail
  lw t0, 4(s1)
  li t1, 1
  bne t0, t1, host_fail
  ld t0, 8(s1)
  bne t0, t1, host_fail

  # Consumer-first initial epoch: all targets are live before the producer
  # publishes SendSeq[0].
  li s4, 0
initial_submit:
  bgeu s4, s3, initial_wait_begin
  mv a0, s4
  call submit_target
  addi s4, s4, 1
  j initial_submit
initial_wait_begin:
  li s4, 0
initial_wait:
  bgeu s4, s3, fill_channel
  mv a0, s4
  call wait_target
  addi s4, s4, 1
  j initial_wait

  # The selected MessageTransfer contract has one outstanding credit. Fill it,
  # then launch another producer before either multicast consumer advances.
fill_channel:
  li a0, 3
  call submit_target
  li a0, 3
  call wait_target
  li a0, 3
  call submit_target

  # Consumer zero may advance independently, but the multicast message remains
  # resident until consumer two also commits it.
  li a0, 0
  call submit_target
  li a0, 0
  call wait_target
  li a0, 3
  call require_target_busy

  # The independent peer proves this backpressure is channel-local. Consumer
  # two then releases the credit needed to publish SendSeq[2].
  li a0, 1
  call submit_target
  li a0, 1
  call wait_target
  li a0, 2
  call submit_target
  li a0, 2
  call wait_target
  li a0, 3
  call wait_target

  # Consume the producer output that was held behind the full channel.
  li a0, 0
  call submit_target
  li a0, 2
  call submit_target
  li a0, 0
  call wait_target
  li a0, 2
  call wait_target

  ld t0, 24(s1)
  li t1, 7
  sw t1, 0(t0)
  li a0, 0
  call m5_exit
host_idle:
  wfi
  j host_idle
host_fail:
  li a0, 0
  li a1, 1
  call m5_fail
  j host_idle

submit_target:
  sw a0, 0(s0)
  sw zero, 4(s0)
  li t0, 2
  sw t0, 8(s0)
  li t0, 1
  fence iorw, iorw
  sw t0, 8(s0)
  lw t0, 12(s0)
  andi t1, t0, 4
  bnez t1, host_fail
  ret

wait_target:
  sw a0, 0(s0)
  sw zero, 4(s0)
wait_target_poll:
  lw t0, 12(s0)
  andi t1, t0, 4
  bnez t1, host_fail
  andi t1, t0, 2
  beqz t1, wait_target_poll
  fence iorw, iorw
  ret

require_target_busy:
  sw a0, 0(s0)
  lw t0, 12(s0)
  andi t1, t0, 4
  bnez t1, host_fail
  andi t1, t0, 1
  beqz t1, host_fail
  ret
.size loom_host_entry, .-loom_host_entry
)asm";
}

llvm::StringRef singleInvocationHostProgramSource() {
  return R"asm(
.section .text,"ax",@progbits
.align 2
.globl loom_host_entry
.type loom_host_entry,@function
loom_host_entry:
  mv s0, a0
  mv s1, a2
  mv s2, a3
  mv s3, a1
  li t0, 4
  bne s3, t0, host_fail
  li t0, 1
  bne s2, t0, host_fail
  lw t0, 0(s1)
  li t1, 0x494d474c
  bne t0, t1, host_fail
  lw t0, 4(s1)
  li t1, 1
  bne t0, t1, host_fail
  ld t0, 8(s1)
  bne t0, t1, host_fail
  # The RTL bridge engine is a one-shot provider. Publish the producer buffer
  # before launching either fixed-buffer consumer.
  li s4, 3
submit_one:
  sw s4, 0(s0)
  sw zero, 4(s0)
  li t0, 2
  sw t0, 8(s0)
  li t0, 1
  fence iorw, iorw
  sw t0, 8(s0)
  lw t0, 12(s0)
  andi t1, t0, 4
  bnez t1, host_fail
wait_one:
  lw t0, 12(s0)
  andi t1, t0, 4
  bnez t1, host_fail
  andi t1, t0, 2
  beqz t1, wait_one
  fence iorw, iorw
  li t2, 3
  bne s4, t2, advance
  li s4, 0
  j submit_one
advance:
  addi s4, s4, 1
  bltu s4, t2, submit_one
complete:
  ld t0, 24(s1)
  li t1, 7
  sw t1, 0(t0)
  li a0, 0
  call m5_exit
host_idle:
  wfi
  j host_idle
host_fail:
  li a0, 0
  li a1, 1
  call m5_fail
  j host_idle
.size loom_host_entry, .-loom_host_entry
)asm";
}

llvm::StringRef pairedInvocationHostProgramSource() {
  return R"asm(
.section .text,"ax",@progbits
.align 2
.globl loom_host_entry
.type loom_host_entry,@function
loom_host_entry:
  mv s0, a0
  mv s1, a2
  li t0, 1
  bne a1, t0, host_fail
  bne a3, t0, host_fail
  lw t0, 0(s1)
  li t1, 0x494d474c
  bne t0, t1, host_fail
  lw t0, 4(s1)
  li t1, 1
  bne t0, t1, host_fail
  ld t0, 8(s1)
  bne t0, t1, host_fail

  sw zero, 0(s0)
  sw zero, 4(s0)
  li t0, 2
  sw t0, 8(s0)
  li t0, 1
  fence iorw, iorw
  sw t0, 8(s0)
wait:
  lw t0, 12(s0)
  andi t1, t0, 4
  bnez t1, host_fail
  andi t1, t0, 2
  beqz t1, wait
  fence iorw, iorw

  ld t0, 24(s1)
  li t1, 7
  sw t1, 0(t0)
  li a0, 0
  call m5_exit
host_idle:
  wfi
  j host_idle
host_fail:
  li a0, 0
  li a1, 1
  call m5_fail
  j host_idle
.size loom_host_entry, .-loom_host_entry
)asm";
}

llvm::StringRef spatialInstructionProgramSource() {
  return R"asm(
.section .text,"ax",@progbits
.align 2
.globl __loom_thread_entry_0
.type __loom_thread_entry_0,@function
__loom_thread_entry_0:
  srli t0, a1, 32
  sw a1, 20(a0)
  sw t0, 24(a0)
  sw a2, 28(a0)
  li t0, 1
  fence iorw, iorw
  sw t0, 4(a0)
1:
  lw t0, 0(a0)
  andi t1, t0, 4
  bnez t1, 3f
  andi t1, t0, 2
  beqz t1, 1b
  fence iorw, iorw
  li t0, 1
  sw t0, 0(a3)
2:
  wfi
  j 2b
3:
  lw t0, 8(a0)
  bnez t0, 4f
  li t0, 1
4:
  sw t0, 4(a3)
  j 2b
.size __loom_thread_entry_0, .-__loom_thread_entry_0
)asm";
}

} // namespace loom::system_test
