struct TableView {
  int *values;
  int count;
};

int table_values[64];
struct TableView table = {table_values, 64};

__attribute__((noinline)) static void read_table(const struct TableView *view,
                                                 int *output) {
  for (int index = 0; index < view->count; ++index)
    output[index] = view->values[index] + index;
}

int main(void) {
  int output[64];
  read_table(&table, output);
  return output[0] != 0 || output[63] != 63;
}
