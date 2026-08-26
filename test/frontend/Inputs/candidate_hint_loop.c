int loop_candidate(int *values) {
#ifdef LOOM_CANDIDATE_HINTS
#pragma loom candidate
#endif
  for (int index = 0; index != 2; ++index)
    values[index] = index;

  int index = 0;
#ifdef LOOM_CANDIDATE_HINTS
#pragma loom candidate
#endif
  while (index != 2) {
    values[index] += 1;
    ++index;
  }

#ifdef LOOM_CANDIDATE_HINTS
#pragma loom candidate
#endif
  do {
    --index;
    values[index] += 1;
  } while (index != 0);
  return values[0];
}

#ifdef LOOM_CANDIDATE_HINTS
#pragma loom candidate
#endif
int controlled_loop_candidate(int *values, int run) {
  if (run)
#ifdef LOOM_CANDIDATE_HINTS
#pragma loom candidate
#endif
    for (int index = 0; index != 2; ++index)
      values[index] = index;
  return values[0];
}

int main(void) {
  int values[2] = {9, 8};
  return controlled_loop_candidate(values, 0) == 9 ? 0 : 1;
}
