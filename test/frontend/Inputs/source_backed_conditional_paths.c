static volatile int take_cold_path;

__attribute__((noinline)) static void increment(int *values) {
  for (int index = 0; index < 4; ++index)
    values[index] += 1;
}

__attribute__((noinline)) static void cold_path(int *values) {
  if (take_cold_path)
    increment(values);
}

__attribute__((noinline)) static void hot_path(int *values) {
  increment(values);
}

int main(void) {
  int values[4] = {1, 2, 3, 4};
  cold_path(values);
  hot_path(values);
  return values[0] != 2;
}
