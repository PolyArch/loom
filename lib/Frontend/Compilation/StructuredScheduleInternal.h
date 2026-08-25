#ifndef LOOM_FRONTEND_COMPILATION_STRUCTUREDSCHEDULEINTERNAL_H
#define LOOM_FRONTEND_COMPILATION_STRUCTUREDSCHEDULEINTERNAL_H

#include "Frontend/Compilation/StructuredSchedule.h"

namespace loom::frontend::detail {

llvm::Error validateStructuredVectorScheduleCoordinate(
    const StructuredVectorScheduleCoordinate &coordinate);

} // namespace loom::frontend::detail

#endif // LOOM_FRONTEND_COMPILATION_STRUCTUREDSCHEDULEINTERNAL_H
