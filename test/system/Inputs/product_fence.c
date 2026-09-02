__attribute__((noinline)) static int fenced_accumulate(int *cell, int addend) {
  int observed = *cell;
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  *cell = observed + addend;
  return observed + addend;
}

int main(void) {
  int cell = 17;
  volatile int addend = 25;
  int result = fenced_accumulate(&cell, addend);
  return (result != 42) | (cell != 42);
}
