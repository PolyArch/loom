#pragma loom candidate
__attribute__((annotate("other.annotation", "argument"))) static int
hinted_annotated_arguments(int value) {
  return value + 1;
}

int main(void) { return 0; }
