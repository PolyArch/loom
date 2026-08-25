inline int hinted_inline(int value);

inline int candidate_inline_helper(int value) {
  return value == 0 ? 0 : hinted_inline(value - 1);
}

#ifdef LOOM_CANDIDATE_HINTS
#pragma loom candidate
#endif
inline int hinted_inline(int value) {
  return value == 0 ? 0 : candidate_inline_helper(value - 1);
}

int main() { return 0; }
