// One exact static loop with two independent statements over distinct
// caller-owned arrays. The product funnel enumerates Structured schedule
// decisions for it, and the selected compilation's schedule lineage joins the
// Mapping evidence recorded against that exact program.
enum { lineage_extent = 8 };

__attribute__((noinline)) static void
schedule_lineage(const int *first, const int *second, int *sum, int *scaled) {
  for (int index = 0; index < lineage_extent; ++index) {
    sum[index] = first[index] + second[index];
    scaled[index] = first[index] * 3;
  }
}

int main(void) {
  int first[lineage_extent] = {1, 2, 3, 4, 5, 6, 7, 8};
  int second[lineage_extent] = {10, 20, 30, 40, 50, 60, 70, 80};
  int sum[lineage_extent] = {0};
  int scaled[lineage_extent] = {0};
  schedule_lineage(first, second, sum, scaled);
  int checksum = 0;
  for (int index = 0; index < lineage_extent; ++index)
    checksum += sum[index] + scaled[index];
  return checksum != 504;
}
