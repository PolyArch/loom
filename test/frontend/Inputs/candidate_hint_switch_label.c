int labeled_loop_candidate(int selector) {
  int result = 0;
  switch (selector) {
#pragma loom candidate
    for (int index = 0; index != 4; ++index) {
    case 1:
      result += index;
    }
  }
  return result;
}
