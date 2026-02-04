// Stream update kernel declarations

#ifndef LOOM_TEST_STREAM_UPDATE_H
#define LOOM_TEST_STREAM_UPDATE_H

#include <cstdint>

void stream_update_cpu(const uint32_t* __restrict input,
                       uint32_t* __restrict output,
                       uint32_t n, uint32_t step);
void stream_update_dsa(const uint32_t* __restrict input,
                       uint32_t* __restrict output,
                       uint32_t n, uint32_t step);

#endif // LOOM_TEST_STREAM_UPDATE_H
