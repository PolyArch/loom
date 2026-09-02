__attribute__((noinline)) static int volatile_accumulate(volatile int *cell,
                                                         int addend) {
  int observed = *cell;
  *cell = observed + addend;
  return *cell;
}

int main(void) {
  volatile int cell = 17;
  volatile int addend = 25;
  int result = volatile_accumulate(&cell, addend);
  return (result != 42) | (cell != 42);
}
