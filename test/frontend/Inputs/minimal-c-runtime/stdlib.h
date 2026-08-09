#ifndef LOOM_TEST_MINIMAL_STDLIB_H
#define LOOM_TEST_MINIMAL_STDLIB_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void *malloc(size_t size);
void free(void *pointer);

#ifdef __cplusplus
}
#endif

#endif
