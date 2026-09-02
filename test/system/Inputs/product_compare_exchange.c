__attribute__((noinline)) static int exchange_accumulate(int *cell, int expected,
                                                         int addend) {
  int observed = expected;
  int desired = expected + addend;
  int exchanged = __atomic_compare_exchange_n(cell, &observed, desired, 0,
                                              __ATOMIC_SEQ_CST,
                                              __ATOMIC_SEQ_CST);
  return exchanged ? desired : observed;
}

int main(void) {
  int cell = 17;
  volatile int addend = 25;
  int result = exchange_accumulate(&cell, 17, addend);
  return (result != 42) | (cell != 42);
}
