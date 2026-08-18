__attribute__((noinline)) static void
temporal_memory(const int *first, const int *second, const int *third,
                const int *fourth, int *result) {
  result[0] = first[0] + second[0] + third[0] + fourth[0];
}

int main(void) {
  int first[1] = {5};
  int second[1] = {8};
  int third[1] = {12};
  int fourth[1] = {17};
  int result[1] = {0};
  temporal_memory(first, second, third, fourth, result);
  return result[0] != 42;
}
