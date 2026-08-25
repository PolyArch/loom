int loop_candidate(int *values) {
#pragma loom candidate
  for (int index = 0; index != 2; ++index)
    values[index] = index;
  return values[0];
}
