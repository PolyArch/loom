__attribute__((noinline)) static int rmw_accumulate(int *cell, int addend) {
  int previous = __atomic_fetch_add(cell, addend, __ATOMIC_SEQ_CST);
  return previous + addend;
}

int main(void) {
  int cell = 17;
  volatile int addend = 25;
  int result = rmw_accumulate(&cell, addend);
  return (result != 42) | (cell != 42);
}
