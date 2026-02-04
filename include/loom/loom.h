//===-- loom.h - Loom DSA Pragma Macros -------------------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// This header provides macros for marking DSA-accelerated regions and
// expressing optimization hints for CGRA compilation.
//
// These macros use Clang's built-in attribute mechanism to emit annotations
// that survive to LLVM IR. The code compiles and executes correctly as
// standard C++; annotations are used by loom transformation passes.
//
// == Pragma Categories ==
//
// 1. Mapping Pragmas: Control what gets accelerated
//    - LOOM_ACCEL: Suggest accelerating a function
//    - LOOM_NO_ACCEL: Forbid accelerating a function
//    - LOOM_TARGET: Suggest mapping target (spatial/temporal/specific PE)
//
// 2. Interface Pragmas: Declare data access patterns
//    - LOOM_STREAM: Declare streaming (FIFO) access pattern
//
// 3. Loop Optimization Pragmas: Control loop transformations
//    - LOOM_PARALLEL: Suggest loop parallelism degree
//    - LOOM_NO_PARALLEL: Forbid loop parallelization
//    - LOOM_UNROLL: Suggest loop unroll factor
//    - LOOM_NO_UNROLL: Forbid loop unrolling
//    - LOOM_TRIPCOUNT: Provide loop trip count hint
//    - LOOM_REDUCE: Mark reduction operation
//    - LOOM_MEMORY_BANK: Suggest memory banking
//
//
// == Design Philosophy ==
//
// - All pragmas are optional: Compiler works without any pragmas
// - Two types: Suggestive (hints) and Prohibitive (hard constraints)
// - Compiler has final authority for suggestive pragmas
// - Use __restrict__ for no-alias hints (standard C99)
// - Use const for data direction inference
//
// == Basic Usage ==
//
//   LOOM_ACCEL
//   void vecadd(const float* a, const float* b, float* c, int n) {
//       LOOM_PARALLEL(4) LOOM_UNROLL(8)
//       for (int i = 0; i < n; ++i) {
//           c[i] = a[i] + b[i];
//       }
//   }
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_LOOM_H
#define LOOM_LOOM_H

//===----------------------------------------------------------------------===//
// Compiler Detection
//===----------------------------------------------------------------------===//

#if defined(__clang__)
#define LOOM_CLANG 1
#elif defined(__GNUC__)
#define LOOM_GCC 1
#else
#define LOOM_UNKNOWN_COMPILER 1
#endif

//===----------------------------------------------------------------------===//
// Helper Macros
//===----------------------------------------------------------------------===//

#define LOOM_STRINGIFY_(x) #x
#define LOOM_STRINGIFY(x) LOOM_STRINGIFY_(x)
#define LOOM_CONCAT_(a, b) a##b
#define LOOM_CONCAT(a, b) LOOM_CONCAT_(a, b)

//===----------------------------------------------------------------------===//
// Loom Pragma Macros (Clang)
//===----------------------------------------------------------------------===//

#ifdef LOOM_CLANG

//===========================================================================
// Marker Function Definitions (for loop pragma metadata)
//===========================================================================
//
// These marker functions are used to attach loop optimization metadata.
// They are empty (no-ops) by default. The compilation pass
// converts these calls to !llvm.loop metadata and removes the calls.
//
// The functions use noinline+used to prevent inlining/DCE while keeping
// a minimal runtime footprint. The pass will remove these calls entirely.

#ifdef __cplusplus
extern "C" {
#endif

// Marker for LOOM_PARALLEL: degree, schedule (0=default, 1=contiguous,
// 2=interleaved)
// Note: Using weak+noinline without 'used' so the pass can remove these.
// The inline asm prevents optimization from removing calls before linking.
__attribute__((weak, noinline)) void __loom_loop_parallel(int degree,
                                                          int schedule) {
  // Prevent optimization from removing the call
  __asm__ volatile("" : : "r"(degree), "r"(schedule));
}

// Marker for LOOM_PARALLEL() auto mode
__attribute__((weak, noinline)) void __loom_loop_parallel_auto(void) {
  __asm__ volatile("");
}

// Marker for LOOM_NO_PARALLEL
__attribute__((weak, noinline)) void __loom_loop_no_parallel(void) {
  __asm__ volatile("");
}

// Marker for LOOM_UNROLL: factor
__attribute__((weak, noinline)) void __loom_loop_unroll(int factor) {
  __asm__ volatile("" : : "r"(factor));
}

// Marker for LOOM_UNROLL() auto mode
__attribute__((weak, noinline)) void __loom_loop_unroll_auto(void) {
  __asm__ volatile("");
}

// Marker for LOOM_NO_UNROLL
__attribute__((weak, noinline)) void __loom_loop_no_unroll(void) {
  __asm__ volatile("");
}

// Marker for LOOM_TRIPCOUNT: typical, avg, min, max
__attribute__((weak, noinline)) void __loom_loop_tripcount(int typical, int avg,
                                                           int min, int max) {
  __asm__ volatile("" : : "r"(typical), "r"(avg), "r"(min), "r"(max));
}

#ifdef __cplusplus
}
#endif

//===========================================================================
// Category 1: Mapping Pragmas
//===========================================================================

// LOOM_ACCEL - Suggest that a function should be accelerated
// Usage:
//   LOOM_ACCEL()
//   void my_function(...) { ... }
//
//   LOOM_ACCEL("custom_name")
//   void my_function(...) { ... }
//
// LLVM IR: generates "loom.accel" or "loom.accel=custom_name" annotation
//
// Implementation uses variadic macros with ## extension (GCC/Clang) to handle
// 0-or-1 argument overloading.
#define LOOM_ACCEL_0() __attribute__((annotate("loom.accel")))
#define LOOM_ACCEL_1(name) __attribute__((annotate("loom.accel=" name)))
#define LOOM_ACCEL_EXPAND(x) x
#define LOOM_ACCEL_PICK(_0, _1, NAME, ...) NAME
#define LOOM_ACCEL(...)                                                        \
  LOOM_ACCEL_EXPAND(LOOM_ACCEL_PICK(, ##__VA_ARGS__, LOOM_ACCEL_1,             \
                                    LOOM_ACCEL_0)(__VA_ARGS__))

// LOOM_NO_ACCEL - Forbid accelerating a function (hard constraint)
// Usage:
//   LOOM_NO_ACCEL
//   void legacy_function(...) { ... }
//
// LLVM IR: generates "loom.no_accel" annotation
#define LOOM_NO_ACCEL __attribute__((annotate("loom.no_accel")))

// LOOM_TARGET - Suggest mapping target
// Usage:
//   LOOM_TARGET("spatial")
//   void my_kernel(...) { ... }
//
//   LOOM_TARGET("temporal")
//   void complex_control_flow(...) { ... }
//
//   LOOM_TARGET("pe[0,0]")
//   void critical_op(...) { ... }
//
// LLVM IR: generates "loom.target=<spec>" annotation
#define LOOM_TARGET(spec) __attribute__((annotate("loom.target=" spec)))

//===========================================================================
// Category 2: Interface Pragmas
//===========================================================================

// LOOM_STREAM - Declare streaming (FIFO) access pattern
// Usage:
//   void process(LOOM_STREAM const float* input,
//                const float* lookup_table,  // random access
//                LOOM_STREAM float* output,
//                int n);
//
// On local variables:
//   LOOM_STREAM float intermediate[N];
//
// LLVM IR: generates "loom.stream" annotation on the variable
#define LOOM_STREAM __attribute__((annotate("loom.stream")))

//===========================================================================
// Category 3: Loop Optimization Pragmas
//===========================================================================

// LOOM_PARALLEL - Suggest degree of parallelism for a loop
// Usage:
//   LOOM_PARALLEL(4)
//   for (int i = 0; i < n; ++i) { ... }
//
//   LOOM_PARALLEL(4, contiguous)  // or interleaved
//   for (int i = 0; i < n; ++i) { ... }
//
//   LOOM_PARALLEL()  // Auto: let compiler decide
//   for (int i = 0; i < n; ++i) { ... }
//
// LLVM IR: Emits __loom_loop_parallel(degree, schedule) marker call,
// or __loom_loop_parallel_auto() for auto mode.
// Schedule: 0=default, 1=contiguous, 2=interleaved
// The compilation pass converts these to !llvm.loop metadata.
#define LOOM_SCHED_DEFAULT 0
#define LOOM_SCHED_CONTIGUOUS 1
#define LOOM_SCHED_INTERLEAVED 2
// Lowercase aliases for convenience
#define LOOM_SCHED_contiguous 1
#define LOOM_SCHED_interleaved 2
#define LOOM_PARALLEL_0() __loom_loop_parallel_auto();
#define LOOM_PARALLEL_1(n)                                                     \
  static_assert((n) > 0, "LOOM_PARALLEL requires positive value; "             \
                         "use LOOM_NO_PARALLEL to forbid parallelization, "    \
                         "or LOOM_PARALLEL() for auto");                       \
  __loom_loop_parallel((n), LOOM_SCHED_DEFAULT);
#define LOOM_PARALLEL_2(n, sched)                                              \
  static_assert((n) > 0, "LOOM_PARALLEL requires positive value; "             \
                         "use LOOM_NO_PARALLEL to forbid parallelization, "    \
                         "or LOOM_PARALLEL() for auto");                       \
  __loom_loop_parallel((n), LOOM_SCHED_##sched);
#define LOOM_PARALLEL_EXPAND(x) x
#define LOOM_PARALLEL_GET(_0, _1, _2, NAME, ...) NAME
#define LOOM_PARALLEL(...)                                                     \
  LOOM_PARALLEL_EXPAND(LOOM_PARALLEL_GET(, ##__VA_ARGS__, LOOM_PARALLEL_2,     \
                                         LOOM_PARALLEL_1,                      \
                                         LOOM_PARALLEL_0)(__VA_ARGS__))

// LOOM_NO_PARALLEL - Forbid parallelizing a loop (hard constraint)
// Usage:
//   LOOM_NO_PARALLEL
//   for (int i = 0; i < n; ++i) { ... }
//
// LLVM IR: Emits __loom_loop_no_parallel() marker call.
#define LOOM_NO_PARALLEL __loom_loop_no_parallel();

// LOOM_UNROLL - Suggest loop unroll factor
// Usage:
//   LOOM_UNROLL(8)
//   for (int i = 0; i < n; ++i) { ... }
//
//   LOOM_UNROLL()  // Auto: let compiler decide
//   for (int i = 0; i < n; ++i) { ... }
//
// LLVM IR: Emits __loom_loop_unroll(factor) marker call,
// or __loom_loop_unroll_auto() for auto mode.
#define LOOM_UNROLL_0() __loom_loop_unroll_auto();
#define LOOM_UNROLL_1(n)                                                       \
  static_assert((n) > 0, "LOOM_UNROLL requires positive value; "               \
                         "use LOOM_NO_UNROLL to forbid unrolling, "            \
                         "or LOOM_UNROLL() for auto");                         \
  __loom_loop_unroll((n));
#define LOOM_UNROLL_EXPAND(x) x
#define LOOM_UNROLL_GET(_0, _1, NAME, ...) NAME
#define LOOM_UNROLL(...)                                                       \
  LOOM_UNROLL_EXPAND(LOOM_UNROLL_GET(, ##__VA_ARGS__, LOOM_UNROLL_1,           \
                                     LOOM_UNROLL_0)(__VA_ARGS__))

// LOOM_NO_UNROLL - Forbid unrolling a loop (hard constraint)
// Usage:
//   LOOM_NO_UNROLL
//   for (int i = 0; i < n; ++i) { ... }
//
// LLVM IR: Emits __loom_loop_no_unroll() marker call.
#define LOOM_NO_UNROLL __loom_loop_no_unroll();

// LOOM_TRIPCOUNT - Provide compile-time trip count hints for loop optimization
// All arguments must be compile-time constants. Runtime values are not
// supported.
//
// Usage:
//   LOOM_TRIPCOUNT(100)                          // typical (most common)
//   LOOM_TRIPCOUNT_RANGE(10, 1000)               // min, max
//   LOOM_TRIPCOUNT_TYPICAL(100, 10, 1000)        // typical, min, max
//   LOOM_TRIPCOUNT_FULL(100, 500, 10, 1000)      // typical, avg, min, max
//
//   Named-parameter syntax (C++ only):
//   LOOM_TRIPCOUNT(min=10, max=100)              // min/max only
//   LOOM_TRIPCOUNT(typical=64, min=1, max=1024)  // typical with range
//   LOOM_TRIPCOUNT(typical=64, avg=128, min=1, max=1024)  // all fields
//
// LLVM IR: Emits __loom_loop_tripcount(typical, avg, min, max) marker call.
// Single-value form sets typical; avg defaults to typical; min/max default to 0
// LOOM_TRIPCOUNT_RANGE only sets min/max; typical and avg are 0
//
// Validation:
// - All arguments must be compile-time constants
// - Single-argument form: enforced by backend (compilation pass)
// - Multi-argument forms: enforced via constexpr/static_assert
// - min <= max is validated at compile time

#ifdef __cplusplus
// Named-parameter support for LOOM_TRIPCOUNT (C++ only)
// Allows syntax like LOOM_TRIPCOUNT(min=10, max=100)

namespace loom {
namespace detail {

// Holds a single tripcount parameter value
struct TripcountArg {
  enum Kind { kTypical, kAvg, kMin, kMax };
  Kind kind;
  int value;
  constexpr TripcountArg(Kind k, int v) : kind(k), value(v) {}
  // Allow implicit conversion from int for single-value syntax
  constexpr TripcountArg(int v) : kind(kTypical), value(v) {}
};

// Proxy key that enables "name = value" syntax via operator=
struct TripcountKey {
  TripcountArg::Kind kind;
  constexpr TripcountArg operator=(int v) const {
    return TripcountArg(kind, v);
  }
};

// Named parameter keys
constexpr TripcountKey typical{TripcountArg::kTypical};
constexpr TripcountKey avg{TripcountArg::kAvg};
// clang-format off
constexpr TripcountKey min{TripcountArg::kMin};
constexpr TripcountKey max{TripcountArg::kMax};
// clang-format on

// Tripcount builder with validation
struct TripcountBuilder {
  int typical_v = 0;
  int avg_v = 0;
  int min_v = 0;
  int max_v = 0;
  bool has_min = false;
  bool has_max = false;

  constexpr TripcountBuilder() = default;

  constexpr TripcountBuilder add(TripcountArg arg) const {
    TripcountBuilder result = *this;
    switch (arg.kind) {
    case TripcountArg::kTypical:
      result.typical_v = arg.value;
      if (result.avg_v == 0)
        result.avg_v = arg.value; // avg defaults to typical
      break;
    case TripcountArg::kAvg:
      result.avg_v = arg.value;
      break;
    case TripcountArg::kMin:
      result.min_v = arg.value;
      result.has_min = true;
      break;
    case TripcountArg::kMax:
      result.max_v = arg.value;
      result.has_max = true;
      break;
    }
    return result;
  }

  constexpr bool valid() const {
    // If both min and max are specified, min must be <= max
    if (has_min && has_max && min_v > max_v)
      return false;
    return true;
  }
};

// Helper to build at compile time (for constexpr contexts)
// Variadic version for named parameters
template <typename... Args>
constexpr TripcountBuilder build_tripcount(Args... args) {
  TripcountBuilder builder;
  // Fold expression to process all arguments
  ((builder = builder.add(args)), ...);
  return builder;
}

// Single integer overload for backward compatibility
constexpr TripcountBuilder build_tripcount_single(int n) {
  TripcountBuilder builder;
  builder.typical_v = n;
  builder.avg_v = n;
  return builder;
}

// Builder for compile-time constant tripcount arguments only.
// All tripcount values must be compile-time constants.
template <typename... Args>
constexpr TripcountBuilder build_tripcount_unified(Args... args) {
  TripcountBuilder builder;
  ((builder = builder.add(args)), ...);
  return builder;
}

} // namespace detail
} // namespace loom

// Compile-time validation helper for tripcount arguments.
// All tripcount values must be compile-time constants.
#define LOOM_ASSERT_CONSTANT(val)                                              \
  static_assert(__builtin_constant_p(val),                                     \
                "LOOM_TRIPCOUNT: arguments must be compile-time constants")
#define LOOM_ASSERT_MIN_MAX(tc_min, tc_max)                                    \
  static_assert((tc_min) <= (tc_max), "LOOM_TRIPCOUNT: min must be <= max")

// Detect if first argument is an integer or a named parameter
// Using a helper macro to distinguish between LOOM_TRIPCOUNT(100) and
// LOOM_TRIPCOUNT(min=10, max=100)

// Count arguments: 0, 1, or many
#define LOOM_TC_ARG_COUNT_IMPL(_1, _2, _3, _4, N, ...) N
#define LOOM_TC_ARG_COUNT(...) LOOM_TC_ARG_COUNT_IMPL(__VA_ARGS__, M, M, M, 1)

// Single-argument version: preserves backward compatibility with trailing ;
// Compile-time constant is enforced by the backend (compilation
// pass).
#define LOOM_TRIPCOUNT_1(n) __loom_loop_tripcount((n), (n), 0, 0);

// Multi-argument version (named parameters)
// Uses do-while for multiple statements, requires trailing ; from caller
// All arguments must be compile-time constants.
#define LOOM_TRIPCOUNT_M(...)                                                  \
  do {                                                                         \
    using namespace loom::detail;                                              \
    constexpr auto _loom_tc_builder = build_tripcount_unified(__VA_ARGS__);    \
    static_assert(_loom_tc_builder.valid(),                                    \
                  "LOOM_TRIPCOUNT: min must be <= max");                       \
    __loom_loop_tripcount(_loom_tc_builder.typical_v, _loom_tc_builder.avg_v,  \
                          _loom_tc_builder.min_v, _loom_tc_builder.max_v);     \
  } while (0)

// Dispatch based on argument count
#define LOOM_TC_DISPATCH(N) LOOM_TRIPCOUNT_##N
#define LOOM_TC_DISPATCH_EXPAND(N) LOOM_TC_DISPATCH(N)
#define LOOM_TRIPCOUNT(...)                                                    \
  LOOM_TC_DISPATCH_EXPAND(LOOM_TC_ARG_COUNT(__VA_ARGS__))(__VA_ARGS__)

#else // C mode
// Compile-time validation for C mode (C11 _Static_assert)
// _Static_assert requires constant expressions, enforcing compile-time values.
#define LOOM_ASSERT_CONSTANT(val)                                              \
  _Static_assert((val) >= 0 || (val) < 0, "LOOM_TRIPCOUNT: arguments must be " \
                                          "compile-time constants")
#define LOOM_ASSERT_MIN_MAX(tc_min, tc_max)                                    \
  _Static_assert((tc_min) <= (tc_max), "LOOM_TRIPCOUNT: min must be <= max")

// C mode LOOM_TRIPCOUNT: compile-time constant is enforced by the backend.
#define LOOM_TRIPCOUNT(n) __loom_loop_tripcount((n), (n), 0, 0);
#endif // __cplusplus

// All tripcount macros require compile-time constant arguments.
// Runtime values are not supported and will cause a compilation error.

#define LOOM_TRIPCOUNT_RANGE(tc_min, tc_max)                                   \
  LOOM_ASSERT_MIN_MAX(tc_min, tc_max);                                         \
  __loom_loop_tripcount(0, 0, (tc_min), (tc_max));

#define LOOM_TRIPCOUNT_TYPICAL(typical, tc_min, tc_max)                        \
  LOOM_ASSERT_CONSTANT(typical);                                               \
  LOOM_ASSERT_MIN_MAX(tc_min, tc_max);                                         \
  __loom_loop_tripcount((typical), (typical), (tc_min), (tc_max));

#define LOOM_TRIPCOUNT_FULL(typical, avg, tc_min, tc_max)                      \
  LOOM_ASSERT_CONSTANT(typical);                                               \
  LOOM_ASSERT_CONSTANT(avg);                                                   \
  LOOM_ASSERT_MIN_MAX(tc_min, tc_max);                                         \
  __loom_loop_tripcount((typical), (avg), (tc_min), (tc_max));

// LOOM_REDUCE - Mark a variable as a reduction accumulator
// Usage:
//   LOOM_REDUCE(+)
//   float sum = 0;
//   for (int i = 0; i < n; ++i) { sum += data[i]; }
//
// Supported operators: +, *, min, max, &, |, ^
// LLVM IR: generates "loom.reduce=<op>" annotation on the variable
#define LOOM_REDUCE(op)                                                        \
  __attribute__((annotate("loom.reduce=" LOOM_STRINGIFY(op))))

// LOOM_MEMORY_BANK - Suggest memory banking configuration for an array
// Usage:
//   LOOM_MEMORY_BANK(8)
//   float scratchpad[1024];  // 8 banks, cyclic by default
//
//   LOOM_MEMORY_BANK(4, block)
//   float b[1024];  // 4 banks, block distribution
//
// LLVM IR: generates "loom.memory_bank=<n>" or "loom.memory_bank=<n>,<strat>"
#define LOOM_MEMORY_BANK_1(n)                                                  \
  __attribute__((annotate("loom.memory_bank=" LOOM_STRINGIFY(n))))
#define LOOM_MEMORY_BANK_2(n, strat)                                           \
  __attribute__((annotate(                                                     \
      "loom.memory_bank=" LOOM_STRINGIFY(n) "," LOOM_STRINGIFY(strat))))
#define LOOM_MEMORY_BANK_GET(_1, _2, NAME, ...) NAME
#define LOOM_MEMORY_BANK(...)                                                  \
  LOOM_MEMORY_BANK_GET(__VA_ARGS__, LOOM_MEMORY_BANK_2, LOOM_MEMORY_BANK_1)    \
  (__VA_ARGS__)

#endif // LOOM_CLANG

//===----------------------------------------------------------------------===//
// Loom Pragma Macros (GCC - fallback)
//===----------------------------------------------------------------------===//

#ifdef LOOM_GCC

// GCC uses similar attribute syntax to Clang
// LOOM_ACCEL: Support 0-or-1 argument overloading (same as Clang)
#define LOOM_ACCEL_GCC_0() __attribute__((annotate("loom.accel")))
#define LOOM_ACCEL_GCC_1(name) __attribute__((annotate("loom.accel=" name)))
#define LOOM_ACCEL_GCC_PICK(_0, _1, NAME, ...) NAME
#define LOOM_ACCEL(...)                                                        \
  LOOM_ACCEL_GCC_PICK(, ##__VA_ARGS__, LOOM_ACCEL_GCC_1, LOOM_ACCEL_GCC_0)     \
  (__VA_ARGS__)
#define LOOM_NO_ACCEL __attribute__((annotate("loom.no_accel")))
#define LOOM_TARGET(spec) __attribute__((annotate("loom.target=" spec)))
#define LOOM_STREAM __attribute__((annotate("loom.stream")))
#define LOOM_PARALLEL(...)                    /* no-op for GCC */
#define LOOM_NO_PARALLEL                      /* no-op for GCC */
#define LOOM_UNROLL(...)                      /* no-op for GCC */
#define LOOM_NO_UNROLL                        /* no-op for GCC */
#define LOOM_TRIPCOUNT(n)                     /* no-op for GCC */
#define LOOM_TRIPCOUNT_RANGE(tc_min, tc_max)  /* no-op for GCC */
#define LOOM_TRIPCOUNT_TYPICAL(typ, mi, mx)   /* no-op for GCC */
#define LOOM_TRIPCOUNT_FULL(typ, avg, mi, mx) /* no-op for GCC */
#define LOOM_REDUCE(op)                       /* no-op for GCC */
#define LOOM_MEMORY_BANK(n, ...)              /* no-op for GCC */

#endif // LOOM_GCC

//===----------------------------------------------------------------------===//
// Unknown Compiler - Empty Macros
//===----------------------------------------------------------------------===//

#ifdef LOOM_UNKNOWN_COMPILER

// Unknown compiler: all pragmas are no-ops
#define LOOM_ACCEL(...)
#define LOOM_NO_ACCEL
#define LOOM_TARGET(spec)
#define LOOM_STREAM
#define LOOM_PARALLEL(...)
#define LOOM_NO_PARALLEL
#define LOOM_UNROLL(...)
#define LOOM_NO_UNROLL
#define LOOM_TRIPCOUNT(n)
#define LOOM_TRIPCOUNT_RANGE(tc_min, tc_max)
#define LOOM_TRIPCOUNT_TYPICAL(typ, tc_min, tc_max)
#define LOOM_TRIPCOUNT_FULL(typ, avg, tc_min, tc_max)
#define LOOM_REDUCE(op)
#define LOOM_MEMORY_BANK(n, ...)

#endif // LOOM_UNKNOWN_COMPILER

#endif // LOOM_LOOM_H
