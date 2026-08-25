#define CANDIDATE_LOOP for (int index = 0; index != 4; ++index)

int macro_loop_candidate(void) {
#pragma loom candidate
  CANDIDATE_LOOP {}
  return 0;
}
