# Loom

A full-stack compiler and hardware generation framework for domain-specific accelerators. Compiles annotated C/C++ to synthesizable CGRA hardware with cycle-accurate simulation.

## Features

- **C++ to Hardware**: Automatic dataflow graph extraction from annotated C/C++ kernels
- **Flexible Architecture**: Programmatic CGRA topology design via C++ ADG API
- **Complete Backend**: Generates both cycle-accurate SystemC models and synthesizable SystemVerilog RTL
- **Integrated Verification**: Co-simulation framework for host-accelerator validation

## Quick Start

### Build from Source

```bash
# Initialize submodules (LLVM, CIRCT)
make init

# Build the compiler
make rebuild

# Run tests
make check
```

### Compile a Kernel

```cpp
#include <loom/loom.h>

LOOM_ACCEL("axpy")
void axpy_kernel(const uint32_t* x, const uint32_t* y,
                 uint32_t* out, uint32_t alpha, uint32_t N) {
    LOOM_PARALLEL(4, contiguous)
    for (uint32_t i = 0; i < N; i++) {
        out[i] = alpha * x[i] + y[i];
    }
}
```

```bash
# Compile to dataflow graph
loom axpy.cpp -o axpy.handshake.mlir
# Outputs: .llvm.ll, .llvm.mlir, .scf.mlir, .handshake.mlir
```

### Define Hardware Architecture

```cpp
#include <loom/adg.h>
using namespace loom::adg;

int main() {
    ADGBuilder builder("my_cgra");

    // Define PE template
    auto pe = builder.newPE("alu")
        .setLatency(1, 1, 1)
        .setInputPorts({Type::i32(), Type::i32()})
        .setOutputPorts({Type::i32()})
        .addOp("arith.addi")
        .addOp("arith.muli");

    // Build 4x4 mesh topology
    auto mesh = builder.buildMesh(4, 4, pe, swTemplate, Topology::Mesh);

    // Export hardware
    builder.validateADG();
    builder.exportMLIR("cgra.fabric.mlir");
    builder.exportSV("rtl/");           // SystemVerilog
    builder.exportSystemC("sim/");      // SystemC

    return 0;
}
```

```bash
# Compile ADG program
loom --as-clang my_cgra.cpp -o my_cgra
./my_cgra  # Generates hardware artifacts
```

## Project Structure

```
loom/
├── docs/              # Complete specifications (35+ markdown files)
│   ├── spec-loom.md       # Pipeline overview
│   ├── spec-dataflow.md   # Dataflow MLIR dialect
│   ├── spec-fabric*.md    # Hardware fabric specifications
│   ├── spec-adg*.md       # ADG API and backends
│   ├── spec-mapper*.md    # Place-and-route algorithms
│   └── spec-cosim*.md     # Co-simulation architecture
├── include/loom/      # Public API headers
│   ├── loom.h             # Pragma macros for kernels
│   └── adg.h              # ADG builder C++ API
├── lib/loom/          # Implementation
│   ├── Conversion/        # LLVM/MLIR passes
│   ├── Dialect/           # Dataflow and Fabric dialects
│   └── Hardware/          # Backend generation (SV, SystemC)
├── tools/loom/        # Main compiler executable
├── tests/             # 250+ tests (ADG, apps, SV, fabric)
└── externals/         # LLVM/MLIR and CIRCT submodules
```

## Compilation Pipeline

1. **Frontend**: C++ → LLVM IR → SCF MLIR → Handshake/Dataflow MLIR (software graph)
2. **ADG**: C++ ADG builder → Fabric MLIR (hardware graph)
3. **Mapper**: Software + Hardware graphs → Configuration bitstream
4. **Backend**: Configured SystemC model + SystemVerilog RTL
5. **Cosim**: Host software + simulated accelerator verification

## Documentation

All specifications are in `docs/`:
- Start with `spec-loom.md` for pipeline overview
- See `spec-pragma.md` for available kernel annotations
- See `spec-adg.md` for ADG API reference
- See `spec-cli.md` for command-line interface

## Requirements

- CMake 3.20+
- Clang/Clang++ compiler
- Ninja build system
- SystemC 3.0.1 (for SystemC backend)
- Verilator or Synopsys VCS (for RTL simulation)

## Testing

```bash
# End-to-end test pipeline
ninja -C build clean-loom
ninja -C build loom
ninja -C build check-loom
```

Load EDA tools if needed:
```bash
# Verilator
module load verilator

# Synopsys VCS/Verdi
module load synopsys/vcs synopsys/verdi
```

## License

See LICENSE file for details.
