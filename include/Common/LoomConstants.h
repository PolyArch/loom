#ifndef LOOM_COMMON_LOOMCONSTANTS_H
#define LOOM_COMMON_LOOMCONSTANTS_H

namespace loom {

// Default address-space bit width for fabric.mem base_addr fields.
// Defaults to 48. Overridden once per process by the LOOM_ADDR_BITS
// environment variable when it is a positive integer.
unsigned getDefaultLoomAddrBits();

// Default fabric memory bus width (in bits). Defaults to 32768.
// Overridden once per process by the LOOM_MEM_BUS_WIDTH environment
// variable when it is a positive integer.
unsigned getDefaultLoomMemBusWidth();

} // namespace loom

#endif // LOOM_COMMON_LOOMCONSTANTS_H
