#ifndef LOOM_HARDWARE_RTL_CIRCTCONFORMANCE_H
#define LOOM_HARDWARE_RTL_CIRCTCONFORMANCE_H

#include "llvm/Support/Error.h"

#include <string>

namespace loom::hardware::rtl {

llvm::Expected<std::string> emitCirctConformanceSystemVerilog();

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_CIRCTCONFORMANCE_H
