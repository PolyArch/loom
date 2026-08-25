static int hinted_internal(int value);

static int candidate_helper(int value) {
  return value == 0 ? 0 : hinted_internal(value - 1);
}

#ifdef LOOM_CANDIDATE_HINTS
#pragma loom candidate
#endif
static int hinted_internal(int value) {
  return value == 0 ? 0 : candidate_helper(value - 1);
}

int main(void) { return 0; }
