// One exact static loop with two independent statements over distinct
// caller-owned arrays. The product funnel enumerates Structured schedule
// decisions for it, and the selected compilation's schedule lineage joins the
// Mapping evidence recorded against that exact program. The extent is large
// enough that the parallel schedule pays for its per-core configuration
// residency, so the selected compilation carries the parallel lineage.
enum { lineage_extent = 8192 };

__attribute__((noinline)) static void
schedule_lineage(const int *first, const int *second, int *sum, int *scaled) {
  for (long index = 0; index < lineage_extent; ++index) {
    sum[index] = first[index] + second[index];
    scaled[index] = first[index] * 3;
  }
}

static int first[lineage_extent];
static int second[lineage_extent];
static int sum[lineage_extent];
static int scaled[lineage_extent];

int main(void) {
  for (int index = 0; index < lineage_extent; ++index) {
    first[index] = index + 1;
    second[index] = 10 * (index + 1);
  }
  schedule_lineage(first, second, sum, scaled);
  long checksum = 0;
  for (int index = 0; index < lineage_extent; ++index)
    checksum += sum[index] + scaled[index];
  // Each element contributes (1 + 10 + 3) * (index + 1).
  const long expected = 14L * lineage_extent * (lineage_extent + 1) / 2;
  return checksum != expected;
}
