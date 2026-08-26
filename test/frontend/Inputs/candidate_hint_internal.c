#ifdef LOOM_CANDIDATE_HINTS
#pragma loom candidate
#endif
static int hinted_internal(int value) { return value + 1; }

static int unrelated_internal(int value) { return value - 1; }
