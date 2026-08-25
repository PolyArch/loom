int lambda_loop_candidate(int value) {
  auto loop = [value] {
    int result = value;
#pragma loom candidate
    for (int index = 0; index != 4; ++index)
      result += index;
    return result;
  };
  return loop();
}
