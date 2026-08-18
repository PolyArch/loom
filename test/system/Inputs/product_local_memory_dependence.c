__attribute__((noinline)) static void local_memory_dependence(int *memory) {
  memory[1] = memory[0];
}

int main(void) {
  int memory[2] = {42, 0};
  local_memory_dependence(memory);
  return memory[1] != 42;
}
