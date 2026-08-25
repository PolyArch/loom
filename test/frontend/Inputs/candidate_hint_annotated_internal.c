#ifdef LOOM_CANDIDATE_HINTS
#pragma loom candidate
#endif
__attribute__((annotate("other.annotation"))) static int
hinted_annotated_internal(int value) {
  return value + 1;
}

int main(void) { return 0; }
