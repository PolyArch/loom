#ifndef LOOM_TEST_TESTALLOCATIONPROBE_H
#define LOOM_TEST_TESTALLOCATIONPROBE_H

#include <cstddef>

namespace loom::test {

void startAllocationProbe();
std::size_t stopAllocationProbe();
bool allocationProbeIsCalibrated();

} // namespace loom::test

#endif // LOOM_TEST_TESTALLOCATIONPROBE_H
