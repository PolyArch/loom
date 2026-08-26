#ifdef LOOM_CANDIDATE_HINTS
#pragma loom candidate
#endif
inline int hinted_inline(int value) { return value + 1; }

int call_hinted_inline(int value) { return hinted_inline(value); }
