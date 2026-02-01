# Loom Pragma System Specification

## Quick Reference

| Pragma | Category | Type | Description | Jump |
|--------|----------|------|-------------|------|
| `LOOM_ACCEL` | Mapping | Suggest | Suggest accelerating a function | [Details](#loom_accel) |
| `LOOM_NO_ACCEL` | Mapping | Prohibit | Forbid accelerating a function | [Details](#loom_no_accel) |
| `LOOM_TARGET` | Mapping | Suggest | Suggest mapping target | [Details](#loom_target) |
| `LOOM_STREAM` | Interface | Suggest | Declare streaming access pattern | [Details](#loom_stream) |
| `LOOM_PARALLEL` | Hint | Suggest | Suggest loop parallelism degree | [Details](#loom_parallel) |
| `LOOM_NO_PARALLEL` | Hint | Prohibit | Forbid loop parallelization | [Details](#loom_no_parallel) |
| `LOOM_UNROLL` | Hint | Suggest | Suggest loop unroll factor | [Details](#loom_unroll) |
| `LOOM_NO_UNROLL` | Hint | Prohibit | Forbid loop unrolling | [Details](#loom_no_unroll) |
| `LOOM_TRIPCOUNT` | Hint | Info | Provide loop trip count hint | [Details](#loom_tripcount) |
| `LOOM_REDUCE` | Hint | Suggest | Mark reduction operation | [Details](#loom_reduce) |
| `LOOM_MEMORY_BANK` | Hint | Suggest | Suggest memory banking | [Details](#loom_memory_bank) |

---

## Design Philosophy

### Core Principles

1. **All pragmas are optional**: The compiler works without any pragmas. Pragmas provide hints or constraints, not requirements.

2. **Two types of pragmas**:
   - **Suggestive pragmas**: Hints to the compiler ("I suggest doing X")
   - **Prohibitive pragmas**: Hard constraints ("You must NOT do X")

3. **Compiler has final authority**: For suggestive pragmas, the compiler may ignore them if it determines a better strategy. For prohibitive pragmas, the compiler must respect them.

4. **Minimal and orthogonal**: Each pragma serves a distinct purpose. No overlapping functionality.

### Pragma vs Compiler Automation

| Aspect | Without Pragma | With Pragma |
|--------|----------------|-------------|
| Acceleration scope | Compiler auto-detects suitable regions | User suggests/forbids specific functions |
| Parallelism | Compiler auto-selects optimal degree | User suggests specific parallelism |
| Unroll factor | Compiler auto-selects optimal factor | User suggests specific factor |
| Memory banking | Compiler analyzes access patterns | User suggests banking configuration |
| Trip count | Compiler estimates or assumes worst case | User provides expected trip count |

---

## Category 1: Mapping Pragmas

### LOOM_ACCEL

**Purpose**: Suggest that a function should be accelerated on the CGRA.

**Syntax**:
```cpp
LOOM_ACCEL()
void my_function(...) { ... }

LOOM_ACCEL("custom_name")
void my_function(...) { ... }
```

**Semantics**:
- Suggests the compiler should map this function to the CGRA
- The compiler may still decide not to accelerate if it determines CPU execution is more efficient
- Optional name parameter for debugging and configuration reference
- Without name: compiler generates default name (e.g., `loom_kernel_0`)

**When to use**:
- When you know a function is compute-intensive and suitable for acceleration
- When compiler's auto-detection misses a function you want accelerated

**LLVM IR annotation**: `"loom.accel"` or `"loom.accel=custom_name"`

---

### LOOM_NO_ACCEL

**Purpose**: Forbid accelerating a function (hard constraint).

**Syntax**:
```cpp
LOOM_NO_ACCEL
void legacy_function(...) { ... }
```

**Semantics**:
- The compiler MUST NOT attempt to accelerate this function
- Useful for functions that are incompatible with CGRA execution
- Useful for debugging (isolate issues by disabling acceleration)

**When to use**:
- Legacy code that should not be modified
- Functions with features unsupported by the CGRA
- Debugging acceleration issues

**LLVM IR annotation**: `"loom.no_accel"`

---

### LOOM_TARGET

**Purpose**: Suggest mapping a function or operation to a specific hardware target.

**Syntax**:
```cpp
// Function level: suggest mapping entire function to a target
LOOM_TARGET("spatial")
void my_kernel(...) { ... }

LOOM_TARGET("temporal")
void complex_control_flow(...) { ... }

LOOM_TARGET("pe[0,0]")  // Specific PE (for fine-grained control)
void critical_op(...) { ... }
```

**Target specifiers**:
| Specifier | Meaning |
|-----------|---------|
| `"spatial"` | Map to spatial (dedicated) PEs |
| `"temporal"` | Map to temporal PEs |
| `"pe[x,y]"` | Map to specific PE at coordinates (x,y) |
| `"tile[n]"` | Map to specific tile |

**Semantics**:
- Suggests where the compiler should map the function/operation
- Compiler may override if the suggestion is infeasible

**LLVM IR annotation**: `"loom.target=<spec>"`

---

## Category 2: Interface Pragmas

### LOOM_STREAM

**Purpose**: Declare that a parameter or variable uses streaming (FIFO) access pattern.

**Syntax**:
```cpp
// On function parameters: declare streaming interface
LOOM_ACCEL()
void process(LOOM_STREAM const float* input,  // Streaming input
             const float* lookup_table,        // Random access (default)
             LOOM_STREAM float* output,        // Streaming output
             int n);

// On local variables: declare internal FIFO
LOOM_ACCEL()
void pipeline(...) {
    LOOM_STREAM float intermediate[N];  // Internal FIFO buffer
    producer(..., intermediate, ...);
    consumer(intermediate, ...);
}
```

**Semantics**:
- Indicates sequential, one-pass access pattern
- Enables FIFO-based hardware implementation (lower latency, less storage)
- Without LOOM_STREAM: compiler assumes buffer/memory access (random access capable)
- Compiler can auto-detect streaming patterns, but explicit marking helps

**Hardware implications**:
- LOOM_STREAM parameters may become hardware FIFOs
- Non-LOOM_STREAM parameters become scratchpad/memory interfaces

**LLVM IR annotation**: `"loom.stream"` on the variable/parameter

---

## Category 3: Loop Optimization Pragmas

Loop pragmas control the parallelization and optimization of for-loops. These pragmas work together to define the **design space** that the compiler explores.

### Loop Labeling

C++ supports labeled statements, which can be used to identify loops for pragma application and DSE constraints:

```cpp
outer_loop: LOOM_PARALLEL(4)
for (int i = 0; i < n; ++i) {
    inner_loop: LOOM_UNROLL(8)
    for (int j = 0; j < m; ++j) {
        // ...
    }
}
```

Labels enable:
- Referencing specific loops in `loom-config.yaml` for DSE constraints
- Clear identification in optimization reports
- MLIR location tracking for performance analysis

**Requirement**: Compile with `-g` flag to enable label capture. Without debug info, labels are not preserved in the IR and cannot be captured as `loom.label` metadata.

---

### LOOM_PARALLEL

**Purpose**: Suggest the degree of parallelism for a loop (number of parallel workers).

**Syntax**:
```cpp
// Auto mode: let compiler decide parallelism
LOOM_PARALLEL()
for (int i = 0; i < n; ++i) { ... }

// Explicit parallelism
LOOM_PARALLEL(4)
for (int i = 0; i < n; ++i) {
    c[i] = a[i] + b[i];
}

// With schedule strategy
LOOM_PARALLEL(4, contiguous)
for (int i = 0; i < n; ++i) { ... }

LOOM_PARALLEL(4, interleaved)
for (int i = 0; i < n; ++i) { ... }
```

**Schedule strategies**:
| Strategy | Distribution | Use case |
|----------|--------------|----------|
| `contiguous` (default) | Worker k gets iterations [k*chunk, (k+1)*chunk) | Cache locality, sequential access |
| `interleaved` | Worker k gets iterations [k, k+P, k+2P, ...] | Load balancing, strided access |

**Validation**:
- `LOOM_PARALLEL(0)` or negative values produce a compile-time error via static_assert
- Use `LOOM_NO_PARALLEL` to explicitly forbid parallelization
- Use `LOOM_PARALLEL()` (no args) for auto mode

**Semantics**:
- Suggests creating P parallel instances of the loop body
- Each instance processes a partition of the iteration space
- Compiler may choose different parallelism if P is infeasible
- Tail iterations (when n % P != 0) are handled automatically by the compiler
- Auto mode lets the compiler determine optimal parallelism

**Hardware mapping**:
- P parallel workers map to P groups of PEs processing different data partitions

**LLVM IR**: Emits `__loom_loop_parallel(degree, schedule)` marker call, or `__loom_loop_parallel_auto()` for auto mode.

---

### LOOM_NO_PARALLEL

**Purpose**: Forbid parallelizing a loop (hard constraint).

**Syntax**:
```cpp
LOOM_NO_PARALLEL
for (int i = 0; i < n; ++i) {
    // Loop with complex dependencies that cannot be parallelized
}
```

**Semantics**:
- The compiler MUST NOT parallelize this loop
- Useful for loops with complex control flow or dependencies
- Useful for debugging parallelization-related issues

**LLVM IR**: Emits `__loom_loop_no_parallel()` marker call.

---

### LOOM_UNROLL

**Purpose**: Suggest loop unroll factor (combines multiple adjacent iterations).

**Syntax**:
```cpp
// Auto mode: let compiler decide unroll factor
LOOM_UNROLL()
for (int i = 0; i < n; ++i) { ... }

// Explicit unroll factor
LOOM_UNROLL(8)
for (int i = 0; i < n; ++i) {
    c[i] = a[i] + b[i];
}
```

**Validation**:
- `LOOM_UNROLL(0)` or negative values produce a compile-time error via static_assert
- Use `LOOM_NO_UNROLL` to explicitly forbid unrolling
- Use `LOOM_UNROLL()` (no args) for auto mode

**Semantics**:
- Suggests the compiler should unroll this loop by factor U
- Auto mode lets the compiler determine optimal unrolling
- Compiler may use different factor if U is infeasible
- Unrolling connects U iterations of the dataflow graph together

**Hardware mapping**:
- Unrolling increases the size of the dataflow graph within each parallel worker
- Enables instruction-level parallelism within iterations

**LLVM IR**: Emits `__loom_loop_unroll(factor)` marker call, or `__loom_loop_unroll_auto()` for auto mode. 

---

### LOOM_NO_UNROLL

**Purpose**: Forbid unrolling a loop (hard constraint).

**Syntax**:
```cpp
LOOM_NO_UNROLL
for (int i = 0; i < n; ++i) {
    // Complex loop that should not be unrolled
}
```

**Semantics**:
- The compiler MUST NOT unroll this loop
- Useful for loops with complex control flow or dependencies
- Useful for debugging unroll-related issues

**LLVM IR**: Emits `__loom_loop_no_unroll()` marker call. 

---

### Combining PARALLEL and UNROLL

These two pragmas are orthogonal and can be combined:

```cpp
// 100 iterations, 4 parallel workers, each unrolls by 8
LOOM_PARALLEL(4) LOOM_UNROLL(8)
for (int i = 0; i < 100; ++i) {
    c[i] = a[i] + b[i];
}
```

**Execution model** (with `schedule=contiguous`):
1. Split 100 iterations into 4 partitions: [0-24], [25-49], [50-74], [75-99]
2. Each worker processes its partition with unroll factor 8
3. Worker 0: processes [0-24] as 8+8+8+1 iterations
4. Tail handling (1 remaining iteration) is automatic

**Design space**:
The combination of `parallel(P)`, `unroll(U)`, and `schedule` defines a 3-dimensional design space. The compiler explores this space using performance models before committing to expensive PnR. See [spec-optimization.md](./spec-optimization.md) for details.

---

### LOOM_TRIPCOUNT

**Purpose**: Provide trip count hints for loops with data-dependent bounds.

**Syntax**:
```cpp
// Single value (typical/expected) - avg defaults to typical
LOOM_TRIPCOUNT(100)
for (int i = 0; i < n; ++i) { ... }

// Named-parameter syntax (C++ only):
LOOM_TRIPCOUNT(min = 10, max = 100);  // Min/max bounds only
for (int i = 0; i < n; ++i) { ... }

LOOM_TRIPCOUNT(typical = 64, min = 1, max = 1024);  // Typical with range
for (int i = 0; i < n; ++i) { ... }

LOOM_TRIPCOUNT(typical = 64, avg = 128, min = 1, max = 1024);  // All fields
for (int i = 0; i < n; ++i) { ... }

// Range only (min, max) - typical and avg are 0
LOOM_TRIPCOUNT_RANGE(10, 1000)
for (int i = 0; i < n; ++i) { ... }

// Typical with range (typical, min, max) - avg defaults to typical
LOOM_TRIPCOUNT_TYPICAL(100, 10, 1000)
for (int i = 0; i < n; ++i) { ... }

// Complete specification (typical, avg, min, max)
LOOM_TRIPCOUNT_FULL(100, 500, 10, 1000)
for (int i = 0; i < n; ++i) { ... }
```

**Macro Variants**:
| Macro | Parameters | Description |
|-------|------------|-------------|
| `LOOM_TRIPCOUNT(n)` | typical | Sets typical to n; avg defaults to typical |
| `LOOM_TRIPCOUNT(min=, max=, ...)` | named params | C++ named-parameter syntax (see below) |
| `LOOM_TRIPCOUNT_RANGE(min, max)` | min, max | Sets only min/max bounds |
| `LOOM_TRIPCOUNT_TYPICAL(typ, min, max)` | typical, min, max | Sets typical (avg=typical) with bounds |
| `LOOM_TRIPCOUNT_FULL(typ, avg, min, max)` | all four | Complete specification |

**Named Parameter Syntax (C++ only)**:
In C++, `LOOM_TRIPCOUNT` supports named parameters using the syntax `name = value`:

| Parameter | Description |
|-----------|-------------|
| `typical` | Expected/most common trip count |
| `avg` | Average trip count (defaults to `typical` if not specified) |
| `min` | Minimum trip count |
| `max` | Maximum trip count |

Parameters can be specified in any order. Unspecified parameters default to 0.

**Note**: The named-parameter syntax requires a trailing semicolon (`;`) due to macro expansion. The single-value form `LOOM_TRIPCOUNT(n)` does not require a trailing semicolon for backward compatibility.

**Validation**:
- All tripcount arguments must be compile-time constants
- `min <= max` is enforced via compile-time `static_assert`
- Non-constant arguments will cause a compilation error

**Semantics**:
- Provides compile-time hints for loop optimization
- Critical for performance estimation: `Performance = tripcount * recurrence_length`
- Compiler uses these hints for design space exploration and resource allocation

**When to use**:
- Loops with known iteration bounds (compile-time constants)
- Providing expected ranges for bounded loops (e.g., `for (int i = 0; i < N; ++i)` where N is known)
- Tiled/blocked loops with fixed tile sizes

**LLVM IR**: Emits `__loom_loop_tripcount(typical, avg, min, max)` marker call. 

---

### LOOM_REDUCE

**Purpose**: Mark a variable as a reduction accumulator.

**Syntax**:
```cpp
LOOM_REDUCE(+)
float sum = 0;
for (int i = 0; i < n; ++i) {
    sum += data[i];
}
```

**Supported operators**: `+`, `*`, `min`, `max`, `&`, `|`, `^`

**Semantics**:
- Tells the compiler this apparent loop-carried dependency is actually a reduction
- Enables parallel reduction tree implementation
- Compiler can auto-detect common patterns; pragma helps with complex cases

**Why needed**:
- Loop `sum += x` looks like a dependency (each iteration needs previous sum)
- With LOOM_REDUCE, compiler knows `+` is associative and can parallelize

**LLVM IR annotation**: `"loom.reduce=<op>"` on the variable

---

### LOOM_MEMORY_BANK

**Purpose**: Suggest memory banking configuration for an array.

**Syntax**:
```cpp
LOOM_MEMORY_BANK(8)
float scratchpad[1024];  // 8 banks, cyclic by default

LOOM_MEMORY_BANK(8, cyclic)
float a[1024];  // Explicit cyclic: a[i] in bank[i % 8]

LOOM_MEMORY_BANK(4, block)
float b[1024];  // Block: a[0..255] in bank 0, a[256..511] in bank 1, etc.
```

**Banking strategies**:
| Strategy | Distribution | Use case |
|----------|--------------|----------|
| `cyclic` (default) | `element[i]` in `bank[i % n]` | Stride-1 access patterns |
| `block` | Contiguous chunks per bank | Block-based algorithms |

**Semantics**:
- Suggests the compiler/hwgen should create n memory banks for this array
- Enables parallel memory access (up to n simultaneous accesses)
- Compiler can auto-detect based on access patterns and unroll factors
- Maps to `handshake.memory` with bank configuration in hardware IR

**Bank conflicts**:
- If access pattern doesn't match banking, parallel accesses serialize
- Compiler should warn about potential bank conflicts

**LLVM IR annotation**: `"loom.memory_bank=<n>,<strategy>"`

---

## Memory Alias Handling

**No LOOM_NOALIAS pragma** - We use standard C99 `__restrict__` instead.

**Compiler behavior**:
1. Default: Conservative alias analysis (assume pointers may alias)
2. If compiler cannot prove no-alias: Warn user with performance impact estimate
3. User can add `__restrict__` to indicate no-alias (user's responsibility)

**Example**:
```cpp
LOOM_ACCEL()
void process(const float* __restrict__ a,  // User guarantees no alias
             const float* __restrict__ b,
             float* __restrict__ c,
             int n) {
    LOOM_PARALLEL(4) LOOM_UNROLL(8)
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}
```

---

## Data Direction Inference

**No LOOM_IN/LOOM_OUT pragmas** - Direction is inferred from `const` qualifier.

| Declaration | Inferred direction |
|-------------|-------------------|
| `const T*` | Input (read-only) |
| `T*` | Output or In-Out |
| `const T* __restrict__` | Input, no alias |
| `T* __restrict__` | Output, no alias |

---

## Complete Example

```cpp
#include <loom/loom.h>

// Leaf kernel: element-wise multiply
LOOM_ACCEL("vecmul")
void vecmul(LOOM_STREAM const float* __restrict__ a,
            LOOM_STREAM const float* __restrict__ b,
            LOOM_STREAM float* __restrict__ products,
            int n) {
    LOOM_PARALLEL(4) LOOM_UNROLL(8) LOOM_TRIPCOUNT(1024)
    for (int i = 0; i < n; ++i) {
        products[i] = a[i] * b[i];
    }
}

// Leaf kernel: reduction sum
LOOM_ACCEL("vecsum")
LOOM_TARGET("temporal")  // Complex control, use temporal PE
void vecsum(LOOM_STREAM const float* __restrict__ data,
            float* __restrict__ result,
            int n) {
    LOOM_REDUCE(+)
    float sum = 0.0f;
    LOOM_TRIPCOUNT(1024)
    for (int i = 0; i < n; ++i) {
        sum += data[i];
    }
    *result = sum;
}

// Composite kernel: dot product
LOOM_ACCEL("dotprod")
float dotprod(const float* __restrict__ a,
              const float* __restrict__ b,
              int n) {
    LOOM_MEMORY_BANK(8)
    float products[1024];  // Internal buffer, 8 banks

    float result;
    vecmul(a, b, products, n);
    vecsum(products, &result, n);
    return result;
}

// Legacy code that should not be accelerated
LOOM_NO_ACCEL
void debug_print(const float* data, int n) {
    for (int i = 0; i < n; ++i) {
        printf("%f ", data[i]);
    }
}
```

---

## LLVM IR Representation

### Function-Level Pragmas

Function-level pragmas (LOOM_ACCEL, LOOM_NO_ACCEL, LOOM_TARGET, etc.) generate `llvm.global.annotations`:

```llvm
; Function with LOOM_ACCEL("dotprod")
@.str.loom = private constant [18 x i8] c"loom.accel=dotprod\00"
@llvm.global.annotations = appending global [...] [
  { ptr @dotprod, ptr @.str.loom, ... }
]
```

### Loop-Level Pragmas and !llvm.loop Metadata

Loop pragmas (LOOM_PARALLEL, LOOM_UNROLL, LOOM_TRIPCOUNT, etc.) use the standard LLVM `!llvm.loop` metadata mechanism. The Loom compiler converts pragma marker calls to metadata.

**Metadata format**:
```llvm
; Loop with LOOM_PARALLEL(4) LOOM_UNROLL(2) LOOM_TRIPCOUNT(64)
br i1 %cmp, label %for.body, label %for.end, !llvm.loop !10

!10 = distinct !{!10, !11, !12, !13}
!11 = !{!"loom.parallel", i32 4}
!12 = !{!"loom.unroll", i32 2}
!13 = !{!"loom.tripcount", i32 64, i32 64, i32 0, i32 0}
```

**Metadata types**:
| Pragma | Metadata | Arguments |
|--------|----------|-----------|
| `LOOM_PARALLEL(n)` | `!"loom.parallel"` | i32 degree |
| `LOOM_PARALLEL()` | `!"loom.parallel.auto"` | (none) |
| `LOOM_PARALLEL(n, contiguous)` | `!"loom.parallel"`, `!"loom.schedule.contiguous"` | i32 degree |
| `LOOM_PARALLEL(n, interleaved)` | `!"loom.parallel"`, `!"loom.schedule.interleaved"` | i32 degree |
| `LOOM_NO_PARALLEL` | `!"loom.no_parallel"` | (none) |
| `LOOM_UNROLL(n)` | `!"loom.unroll"` | i32 factor |
| `LOOM_UNROLL()` | `!"loom.unroll.auto"` | (none) |
| `LOOM_NO_UNROLL` | `!"loom.no_unroll"` | (none) |
| `LOOM_TRIPCOUNT(n)` | `!"loom.tripcount"` | i32 typical, i32 avg, i32 min, i32 max |
| Loop label | `!"loom.label"` | MDString name |

### Loop Label Capture

When compiling with `-g` (debug info enabled), C++ statement labels preceding loops are captured as `loom.label` metadata. This enables:
- Clear loop identification in optimization reports
- DSE constraints via `loom-config.yaml` loop references
- MLIR location tracking for debugging

**Requirements**:
- Compile with `-g` flag (debug info must be enabled)
- Label must immediately precede the loop pragma(s)

**Example**:
```cpp
outer_loop:  // <-- C++ statement label
LOOM_PARALLEL(4)
for (int i = 0; i < n; i++) { ... }
```

**Generated metadata**:
```llvm
!10 = distinct !{!10, !11, !12}
!11 = !{!"loom.label", !"outer_loop"}
!12 = !{!"loom.parallel", i32 4}
```

**Note**: Without `-g`, labels are not preserved in the IR and `loom.label` metadata will not be generated.

---

## MLIR Representation

When LLVM IR is imported to MLIR during compilation, loop metadata is preserved as `loom.loop` attributes on `llvm.br` / `llvm.cond_br` operations.

### loom.loop Attribute Encoding

The `loom.loop` attribute is a DictionaryAttr containing keys corresponding to loop pragma metadata:

| Key | Type | Source Pragma |
|-----|------|---------------|
| `parallel` | IntegerAttr (i32) | `LOOM_PARALLEL(n)` |
| `parallel_auto` | UnitAttr | `LOOM_PARALLEL()` |
| `schedule` | StringAttr | `LOOM_PARALLEL(n, contiguous\|interleaved)` |
| `no_parallel` | UnitAttr | `LOOM_NO_PARALLEL` |
| `unroll` | IntegerAttr (i32) | `LOOM_UNROLL(n)` |
| `unroll_auto` | UnitAttr | `LOOM_UNROLL()` |
| `no_unroll` | UnitAttr | `LOOM_NO_UNROLL` |
| `tripcount` | DictionaryAttr | `LOOM_TRIPCOUNT(...)` |
| `label` | StringAttr | C++ statement label |

**Tripcount sub-dictionary**:
```mlir
tripcount = {typical = 100 : i32, avg = 100 : i32, min = 10 : i32, max = 1000 : i32}
```

### MLIR Example

```mlir
// Loop with LOOM_PARALLEL(4) LOOM_UNROLL(2) LOOM_TRIPCOUNT(64)
llvm.cond_br %cmp, ^bb1, ^bb2 {
  loom.loop = {parallel = 4 : i32, unroll = 2 : i32,
               tripcount = {typical = 64 : i32, avg = 64 : i32, min = 0 : i32, max = 0 : i32}}
}
```

### Preservation Across Import and Export

The `loom.loop` attribute survives the MLIR import and export path:
1. **LLVM IR → MLIR**: `!llvm.loop` metadata is parsed and converted to `loom.loop` DictionaryAttr
2. **MLIR → LLVM IR**: `loom.loop` attribute is serialized back to `!llvm.loop` metadata

This enables pragma information to pass through MLIR transformations without loss.

### Annotation Pragmas in MLIR

Function and variable annotation pragmas (LOOM_ACCEL, LOOM_STREAM, etc.) are preserved as string attributes in `llvm.mlir.global @llvm.global.annotations`:

```mlir
llvm.mlir.global appending @llvm.global.annotations() {
  // Contains entries like: "loom.accel", "loom.stream", "loom.memory_bank=8"
}
```

These annotations survive the MLIR import and export path without modification.