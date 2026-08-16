__attribute__((noinline)) static int scalar_add(int lhs, int rhs) {
  return lhs + rhs;
}

int main(void) {
  volatile int lhs = 17;
  volatile int rhs = 25;
  return scalar_add(lhs, rhs) != 42;
}
