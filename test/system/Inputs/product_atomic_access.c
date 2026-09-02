__attribute__((noinline)) static int atomic_accumulate(int *cell, int addend) {
  int observed = __atomic_load_n(cell, __ATOMIC_SEQ_CST);
  __atomic_store_n(cell, observed + addend, __ATOMIC_SEQ_CST);
  return observed + addend;
}

int main(void) {
  int cell = 17;
  volatile int addend = 25;
  int result = atomic_accumulate(&cell, addend);
  return (result != 42) | (cell != 42);
}
