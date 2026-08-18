#include <stddef.h>
void *mmap(void *, size_t, int, int, int, long);
int munmap(void *, size_t);
#define PROT_READ 1
#define MAP_PRIVATE 2
#define MAP_FAILED ((void *)-1)
