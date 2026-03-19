    .section .text.init
    .globl _start
_start:
    la sp, _stack_top
    la t0, __bss_start
    la t1, __bss_end
1:
    bgeu t0, t1, 2f
    sd zero, 0(t0)
    addi t0, t0, 8
    j 1b
2:
    call main
    beqz a0, 3f
    mv a1, a0
    li a0, 0
    call m5_fail
    j 4f
3:
    li a0, 0
    call m5_exit
4:
    wfi
    j 4b
