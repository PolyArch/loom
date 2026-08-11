#include <stdint.h>

static int32_t project(int32_t sample) { return sample; }
static int32_t attention(int32_t projected) { return projected; }
static int32_t stats(int32_t projected) { return projected; }

int main(void) {
  const int32_t projected = project(7);
  return attention(projected) == 7 && stats(projected) == 7 ? 0 : 1;
}
