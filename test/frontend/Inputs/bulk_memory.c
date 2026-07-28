#include <stddef.h>

void copy_bytes(unsigned char *restrict destination,
                const unsigned char *restrict source, size_t byte_count) {
  __builtin_memcpy(destination, source, byte_count);
}

void move_bytes(unsigned char *destination, const unsigned char *source,
                size_t byte_count) {
  __builtin_memmove(destination, source, byte_count);
}

void fill_bytes(unsigned char *destination, size_t byte_count) {
  __builtin_memset(destination, 42, byte_count);
}
