__attribute__((noinline)) static int branch_select(int when_true,
                                                    int when_false,
                                                    int predicate) {
  return predicate ? when_true + 3 : when_false - 5;
}

int main(void) {
  volatile int ordinary_true = 17;
  volatile int ordinary_false = 25;
  volatile int true_predicate = 1;
  volatile int boundary_true = -4;
  volatile int boundary_false = 9;
  volatile int false_predicate = 0;

  int true_result =
      branch_select(ordinary_true, ordinary_false, true_predicate);
  int false_result =
      branch_select(boundary_true, boundary_false, false_predicate);
  return (true_result != 20) | (false_result != 4);
}
