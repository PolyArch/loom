OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
  ram (rwx) : ORIGIN = 0x80000000, LENGTH = 512M
}

SECTIONS
{
  . = ORIGIN(ram);

  .text : ALIGN(16)
  {
    *(.text.init)
    *(.text*)
  } > ram

  .rodata : ALIGN(16)
  {
    *(.rodata*)
  } > ram

  .data : ALIGN(16)
  {
    *(.data*)
    *(.sdata*)
  } > ram

  .bss : ALIGN(16)
  {
    __bss_start = .;
    *(.sbss*)
    *(.bss*)
    *(COMMON)
    __bss_end = .;
  } > ram

  . = ALIGN(16);
  _end = .;
  _stack_top = ORIGIN(ram) + LENGTH(ram) - 0x1000;
}
