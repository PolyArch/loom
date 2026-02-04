// Stream nested kernel declarations

#ifndef LOOM_TEST_STREAM_NESTED_H
#define LOOM_TEST_STREAM_NESTED_H

#include <cstdint>

void stream_nested_cpu(const uint32_t* __restrict input,
                       uint32_t* __restrict output, uint32_t n);
void stream_nested_dsa(const uint32_t* __restrict input,
                       uint32_t* __restrict output, uint32_t n);

#endif // LOOM_TEST_STREAM_NESTED_H
