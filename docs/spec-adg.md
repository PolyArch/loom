# ADG (Architecture Description Graph) Specification

## Overview

The ADG system provides a C++ API for programmatically constructing CGRA
(Coarse-Grained Reconfigurable Array) hardware descriptions. An ADG represents
the hardware structure of an accelerator, including processing elements,
switches, memory units, and their interconnections.

The ADG system produces four output formats:

1. **Fabric MLIR**: Hardware IR compatible with the Loom compiler toolchain
2. **DOT**: Graphviz visualization for documentation and debugging
3. **SystemC**: Cycle-accurate simulation model for modeling and verification
4. **SystemVerilog**: Synthesizable RTL for FPGA or ASIC implementation

This document defines the overall ADG design. Companion documents provide
detailed specifications:

- [spec-adg-api.md](./spec-adg-api.md): ADGBuilder C++ API reference
- [spec-adg-sysc.md](./spec-adg-sysc.md): SystemC generation specification
- [spec-adg-sv.md](./spec-adg-sv.md): SystemVerilog generation specification
- [spec-adg-tools.md](./spec-adg-tools.md): Simulation tools and waveform formats
- [spec-mapper.md](./spec-mapper.md): Place-and-route from software graph to ADG-generated hardware graph

## Design Philosophy

### Single-Source Semantics

The ADG API is implemented entirely in C++. This design choice ensures that the
semantics of the Fabric MLIR dialect are maintained in a single codebase. Using
Python bindings would require duplicating semantic validation logic, leading to
potential inconsistencies and increased maintenance burden.

### Builder Pattern

All ADG construction uses the `ADGBuilder` class. This pattern provides:

- Implicit state management (e.g., tracking module definitions)
- Automatic port ordering to satisfy `fabric.module` constraints
- Deferred validation until export time
- Consistent error handling

Users should never directly instantiate fabric operation objects. All
construction goes through builder methods.

### Hardware vs Runtime Configuration Separation

**The ADG builder exclusively describes physical hardware structure.**

The ADG defines what hardware exists (PEs, switches, memories, connections) but
has absolutely no influence on config_mem values. Runtime configuration
(instruction memories, route tables, tag mappings, constant values) is:

- Intentionally left empty at ADG construction time
- Populated by the mapper stage that performs place-and-route from software
  dataflow graphs (see [spec-mapper.md](./spec-mapper.md))
- Determined by the compiler, not the hardware architect

This strict separation ensures:

- Clean architectural exploration without P&R dependency
- Reusable ADG definitions across different software workloads
- Clear distinction between synthesis-time and runtime parameters
- Hardware description is purely structural, not behavioral

**What ADG builder controls:**
- Physical module definitions (PEs, switches, memories)
- Hardware parameters (latency, interval, port counts, connectivity)
- Topology and interconnect structure
- Visualization (DOT export)

**What ADG builder does NOT control:**
- config_mem contents (values are always default/empty in ADG output)
- Runtime behavior of the accelerator
- Instruction memory contents
- Route table configurations
- Tag mapping tables

### Instance and Template Semantics

The ADG builder uses a **modify-on-clone with deduplication** model:

**Clone creates reference:** When you clone a module template, the instance
initially references the original definition. Multiple clones share the same
underlying template.

**Modify forks a new template:** When you modify any hardware attribute on a
cloned instance, the builder automatically creates a new anonymous template
with the modified attributes. The instance is decoupled from the original and
now references this new template. Other instances remain unchanged.

**Deduplication at validation:** During `validateADG()`, the builder identifies
templates that are **hardware equivalent** and merges them:

- Two templates are equivalent if they produce identical physical hardware
- Name, instance ID, and port indices do not affect equivalence (unless they
  influence physical routing)
- Only attributes that affect synthesis output matter

This design allows flexible template variations while ensuring minimal RTL
output through automatic deduplication.

See [spec-adg-api.md](./spec-adg-api.md) for detailed clone and validation
semantics.

## Usage Model

A typical ADG construction workflow:

```cpp
#include <loom/adg.h>

int main() {
    // Create builder
    ADGBuilder builder("my_cgra");

    // Define module templates
    auto pe = builder.newPE("alu_pe")
        .setLatency(1, 1, 1)
        .setInterval(1, 1, 1)
        .setInputPorts({Type::i32(), Type::i32()})
        .setOutputPorts({Type::i32()})
        .setBodyMLIR(R"(
            ^bb0(%a: i32, %b: i32):
                %sum = arith.addi %a, %b : i32
                fabric.yield %sum : i32
        )");

    auto sw = builder.newSwitch("router")
        .setPortCount(4, 4);

    // Build topology
    builder.buildMesh(4, 4, pe, sw, Topology::Mesh);

    // Validate
    builder.validateADG();

    // Export
    builder.exportMLIR("output/my_cgra.fabric.mlir");
    builder.exportDOT("output/my_cgra.dot");
    builder.exportSysC("output/sysc/");
    builder.exportSV("output/rtl/");

    return 0;
}
```

The program is compiled with `loom --as-clang` (recommended) or standard
`clang++`. When using `loom --as-clang`, the ADG library is linked automatically.
When using `clang++` directly, you must manually link against the ADG library:

```bash
clang++ -I<loom-install>/include my_cgra.cpp -L<loom-install>/lib -lloom-adg -o my_cgra
```

Execution produces the specified output files.

## Fabric Module Hierarchy

An ADG maps to a single `fabric.module` containing:

- Processing elements (`fabric.pe`, `fabric.temporal_pe`)
- Routing switches (`fabric.switch`, `fabric.temporal_sw`)
- Tag operations (`fabric.add_tag`, `fabric.map_tag`, `fabric.del_tag`)
- Memory interfaces (`fabric.memory`, `fabric.extmemory`)
- Module instantiations (`fabric.instance`)

The builder automatically ensures correct port ordering (memref*, native*,
tagged*) as required by [spec-fabric.md](./spec-fabric.md).

## Configurable Modules

Certain fabric operations contain runtime configuration parameters that form
the CGRA bitstream. The following table lists all configurable operations and
their configuration requirements:

| Operation | Config Parameter | Description | Reference |
|-----------|-----------------|-------------|-----------|
| `fabric.pe` (tagged) | `output_tag` | Output tag values (tagged interface only) | [spec-fabric-pe.md](./spec-fabric-pe.md) |
| `fabric.pe` (constant, native) | `constant_value` | Constant value for handshake.constant body | [spec-fabric-pe.md](./spec-fabric-pe.md) |
| `fabric.pe` (constant, tagged) | `constant_value`, `output_tag` | Constant value + output tag | [spec-fabric-pe.md](./spec-fabric-pe.md) |
| `fabric.pe` (dataflow.stream, native) | `cont_cond_sel` | Runtime-selectable continue condition (`<`, `<=`, `>`, `>=`, `!=`) as 5-bit one-hot | [spec-fabric-pe.md](./spec-fabric-pe.md) |
| `fabric.add_tag` | `tag` | Constant tag to attach | [spec-fabric-tag.md](./spec-fabric-tag.md) |
| `fabric.map_tag` | `table` | Tag remapping table | [spec-fabric-tag.md](./spec-fabric-tag.md) |
| `fabric.switch` | `route_table` | Static routing configuration | [spec-fabric-switch.md](./spec-fabric-switch.md) |
| `fabric.temporal_pe` | `instruction_mem` | Time-multiplexed instruction slots | [spec-fabric-temporal_pe.md](./spec-fabric-temporal_pe.md) |
| `fabric.temporal_sw` | `route_table` | Tag-indexed routing tables | [spec-fabric-temporal_sw.md](./spec-fabric-temporal_sw.md) |

Operations without runtime configuration:

| Operation | Notes | Reference |
|-----------|-------|-----------|
| `fabric.pe` (compute) | PEs with arith/math body and `dataflow.{carry,invariant,gate}` body have no runtime config | [spec-fabric-pe.md](./spec-fabric-pe.md) |
| `fabric.memory` | All parameters are hardware (synthesis-time) | [spec-fabric-mem.md](./spec-fabric-mem.md) |
| `fabric.extmemory` | All parameters are hardware (synthesis-time) | [spec-fabric-mem.md](./spec-fabric-mem.md) |
| `fabric.del_tag` | No configuration; purely combinational | [spec-fabric-tag.md](./spec-fabric-tag.md) |
| `fabric.instance` | Instantiates existing definitions; config on referenced module | [spec-fabric.md](./spec-fabric.md) |

**Note:** The individual spec-fabric-*.md documents are authoritative for
runtime configuration parameter definitions and bit width formulas. If there
is any discrepancy between this table and the referenced documents, the
referenced documents take precedence.

A `fabric.pe` containing `handshake.constant` is a special case: the constant
value is runtime configurable. A `fabric.pe` containing `dataflow.stream` is
another special case with a 5-bit `cont_cond_sel` runtime field. All other
compute PEs (arith, math, `dataflow.carry`, `dataflow.invariant`,
`dataflow.gate`) have no runtime configuration beyond output_tag for tagged
interfaces.

### Configuration Bit Width Authority

`CONFIG_WIDTH` formulas and operation-local field packing are defined
authoritatively in [spec-fabric-config_mem.md](./spec-fabric-config_mem.md).
Operation-specific semantics remain authoritative in the corresponding
`spec-fabric-*.md` documents.

## config_mem Principles

All runtime configuration is consolidated into a single memory-mapped register
array called `config_mem`. For the formal definition of config_mem (word width,
depth calculation, CONFIG_WIDTH derivation), see
[spec-fabric-config_mem.md](./spec-fabric-config_mem.md).

This design provides:

- Unified configuration interface via AXI-Lite
- Simple software driver model
- Deterministic configuration sequence

### Access Model

**config_mem is read-write from host, read-only from accelerator.**

- The host CPU has full read-write access to config_mem via AXI-Lite
- The accelerator (`fabric.module`) can only read config_mem; it cannot modify
  configuration values during execution
- There is no write path from accelerator logic to config_mem

This asymmetric access model ensures deterministic behavior: once configured,
the accelerator's behavior is fixed until the host reconfigures it.

### Address Allocation

Address allocation rules are defined authoritatively in
[spec-fabric-config_mem.md](./spec-fabric-config_mem.md). This includes
operation order traversal, 32-bit word alignment, per-node isolation, and reset
requirements for full reconfiguration.

### Address Map Generation

The `exportSV()` function generates a C header file containing:

- Base address offsets for each configurable node
- Bit position macros for each field
- Total config_mem depth

See [spec-adg-sv.md](./spec-adg-sv.md) for header file format.

## Topology Construction

The ADGBuilder provides helper functions for common interconnect topologies:

| Function | Description |
|----------|-------------|
| `buildMesh(rows, cols, pe, sw, topology)` | Regular grid with configurable connectivity |
| `connect(src, dst)` | Direct point-to-point connection |
| `connectPorts(src, srcPort, dst, dstPort)` | Explicit port-level connection |

Supported topology patterns for `buildMesh`:

| Topology | Description |
|----------|-------------|
| `Topology::Mesh` | Nearest-neighbor with boundary (no wrap) |
| `Topology::Torus` | Nearest-neighbor with wraparound |
| `Topology::DiagonalMesh` | Mesh plus diagonal connections |
| `Topology::DiagonalTorus` | Torus plus diagonal connections |

See [spec-adg-api.md](./spec-adg-api.md) for complete method signatures.

### Builder Methods Overview

| Method | Creates | Reference |
|--------|---------|-----------|
| `newPE(name)` | `fabric.pe` | [spec-adg-api.md](./spec-adg-api.md) |
| `newConstantPE(name)` | Constant `fabric.pe` | [spec-adg-api.md](./spec-adg-api.md) |
| `newLoadPE(name)` | Load `fabric.pe` | [spec-adg-api.md](./spec-adg-api.md) |
| `newStorePE(name)` | Store `fabric.pe` | [spec-adg-api.md](./spec-adg-api.md) |
| `newTemporalPE(name)` | `fabric.temporal_pe` | [spec-adg-api.md](./spec-adg-api.md) |
| `newSwitch(name)` | `fabric.switch` | [spec-adg-api.md](./spec-adg-api.md) |
| `newTemporalSwitch(name)` | `fabric.temporal_sw` | [spec-adg-api.md](./spec-adg-api.md) |
| `newMemory(name)` | `fabric.memory` | [spec-adg-api.md](./spec-adg-api.md) |
| `newExtMemory(name)` | `fabric.extmemory` | [spec-adg-api.md](./spec-adg-api.md) |
| `newAddTag(name)` | `fabric.add_tag` | [spec-adg-api.md](./spec-adg-api.md) |
| `newMapTag(name)` | `fabric.map_tag` | [spec-adg-api.md](./spec-adg-api.md) |
| `newDelTag(name)` | `fabric.del_tag` | [spec-adg-api.md](./spec-adg-api.md) |
| `addModuleInput(name, type)` | Module input port | [spec-adg-api.md](./spec-adg-api.md) |
| `addModuleOutput(name, type)` | Module output port | [spec-adg-api.md](./spec-adg-api.md) |
| `injectMLIR(mlir)` | Raw MLIR injection | [spec-adg-api.md](./spec-adg-api.md) |

## Validation

`validateADG()` performs structural, type, resource, and template-dedup checks
before export. The authoritative validation checklist and error reporting model
are defined in [spec-adg-api.md](./spec-adg-api.md).

## Export Formats

`exportSysC()` and `exportSV()` produce self-contained output directories that
can be moved or archived independently of the Loom installation.

### MLIR Export

`exportMLIR(path)` produces a valid Fabric MLIR file containing:

- A single `fabric.module` definition
- All nested operation definitions
- Hardware parameters fully specified
- Runtime configuration parameters empty (default values)

The output is compatible with standard MLIR tools (mlir-opt, mlir-translate)
and the Loom compiler pipeline.

### DOT Export

`exportDOT(path, mode)` produces Graphviz DOT format with two modes:

| Mode | Description |
|------|-------------|
| `DOTMode::Structure` | Hardware modules and connections only |
| `DOTMode::Detailed` | Includes runtime config visualization |

For node styles, edge styles, and unmapped element conventions, see
[spec-viz-hw.md](./spec-viz-hw.md).

### SystemC Export

`exportSysC(directory)` produces a SystemC simulation model:

- Top-level module with TLM 2.0 configuration interface
- config_mem controller using `simple_target_socket`
- Example testbench with clock generation and VCD tracing
- Example `_main.cpp` for standalone simulation
- `_addr.h` C header with address definitions (shared with SV)
- CMake build configuration
- `lib/` directory with all parameterized library modules (copied)

The generated model targets SystemC 3.0.1 and supports two abstraction levels:

| Mode | Macro | Description |
|------|-------|-------------|
| Cycle-Accurate | `FABRIC_SYSC_CYCLE_ACCURATE` | Exact timing, verification reference |
| Loosely-Timed | `FABRIC_SYSC_LOOSELY_TIMED` | Fast simulation for software development |

See [spec-adg-sysc.md](./spec-adg-sysc.md) for complete specification.

### SystemVerilog Export

`exportSV(directory)` produces a hierarchical RTL design:

- Top-level instantiation module
- config_mem controller with AXI-Lite interface
- C header file with address definitions
- `lib/` directory with all parameterized module templates (copied)

See [spec-adg-sv.md](./spec-adg-sv.md) for complete specification.

### Simulation Tools

Both SystemC and SystemVerilog exports are designed to work with standard
simulation tools. For detailed information on supported simulators (VCS,
Verilator), waveform formats (FSDB, FST, VCD), and co-simulation workflows,
see [spec-adg-tools.md](./spec-adg-tools.md).

## Streaming Handshake Protocol

All streaming connections in the generated hardware use a valid/ready
handshake protocol:

- Transfer occurs when `valid && ready` on the rising clock edge.
- `valid` must not depend combinationally on `ready`.
- `ready` may depend combinationally on `valid`.
- Once asserted, `valid` must remain high until the transfer completes.

For tagged interfaces, value/tag packing follows the authoritative convention
defined in [spec-dataflow.md](./spec-dataflow.md) (`!dataflow.tagged` type).

Both SystemVerilog and SystemC backends implement this protocol.
See [spec-adg-sv.md](./spec-adg-sv.md) and
[spec-adg-sysc.md](./spec-adg-sysc.md) for backend-specific interface
definitions.

## Error Handling

ADGBuilder methods may throw exceptions or return error codes depending on
configuration. The default mode uses exceptions with descriptive messages.

Error categories:

| Category | When Detected | Example |
|----------|--------------|---------|
| Builder errors | During API calls | Invalid parameter values |
| Validation errors | During validateADG() | Unconnected ports |
| Export errors | During export | File I/O failures |

All error messages include the Fabric error code symbol (e.g.,
`COMP_SWITCH_PORT_LIMIT`) for cross-referencing with
[spec-fabric-error.md](./spec-fabric-error.md).

## File Organization

### Loom Source Tree (Implementation)

The ADG implementation is organized within the Loom source tree as follows.
**Note:** These files are used by the exporter implementation; they are NOT
referenced by the generated output. The export functions copy required templates
into the output directory.

```
include/loom/
  adg.h                            # Public API header

include/loom/Dialect/Fabric/
  ADGBuilder.h                     # Builder class declaration
  ADGTypes.h                       # Type definitions

include/loom/Hardware/SystemC/
  fabric_pe.h                      # PE module template
  fabric_pe_constant.h             # Constant PE template
  fabric_pe_load.h                 # Load PE template
  fabric_pe_store.h                # Store PE template
  fabric_temporal_pe.h             # Temporal PE template
  fabric_switch.h                  # Switch template
  fabric_temporal_sw.h             # Temporal switch template
  fabric_memory.h                  # Memory template
  fabric_extmemory.h               # External memory template
  fabric_add_tag.h                 # add_tag template
  fabric_map_tag.h                 # map_tag template
  fabric_del_tag.h                 # del_tag template
  fabric_stream.h                  # Streaming interface
  fabric_common.h                  # Common utilities

include/loom/Hardware/SystemVerilog/
  fabric_pe.sv                     # PE module template
  fabric_pe_constant.sv            # Constant PE template
  fabric_pe_load.sv                # Load PE template
  fabric_pe_store.sv               # Store PE template
  fabric_temporal_pe.sv            # Temporal PE template
  fabric_switch.sv                 # Switch template
  fabric_temporal_sw.sv            # Temporal switch template
  fabric_memory.sv                 # Memory template
  fabric_extmemory.sv              # External memory template
  fabric_add_tag.sv                # add_tag template
  fabric_map_tag.sv                # map_tag template
  fabric_del_tag.sv                # del_tag template
  fabric_common.svh                # Common definitions

lib/loom/Dialect/Fabric/
  ADGBuilder.cpp                   # Builder implementation
  ADGExportMLIR.cpp                # MLIR export
  ADGExportDOT.cpp                 # DOT export
  ADGValidation.cpp                # Validation logic

lib/loom/Hardware/SystemC/
  ADGExportSysC.cpp                # SystemC export logic

lib/loom/Hardware/SystemVerilog/
  ADGExportSV.cpp                  # SystemVerilog export logic
```

## Test Organization

ADG tests should be organized under `tests/adg/`:

```
tests/adg/
  cgra-4x4/          # Basic 4x4 CGRA example
  my-first-cgra/     # Tutorial example
  large-cgra-10x10/  # Stress test with larger grid
```

Each test directory should contain:
- Source C++ file defining the ADG
- Expected output files for verification
- CMakeLists.txt for build integration

## Related Documents

- [spec-fabric.md](./spec-fabric.md): Fabric MLIR dialect specification
- [spec-fabric-pe.md](./spec-fabric-pe.md): Processing element specification
- [spec-fabric-pe-ops.md](./spec-fabric-pe-ops.md): PE allowed operations (single source)
- [spec-fabric-temporal_pe.md](./spec-fabric-temporal_pe.md): Temporal PE specification
- [spec-fabric-switch.md](./spec-fabric-switch.md): Switch specification
- [spec-fabric-temporal_sw.md](./spec-fabric-temporal_sw.md): Temporal switch specification
- [spec-fabric-tag.md](./spec-fabric-tag.md): Tag operations specification
- [spec-fabric-mem.md](./spec-fabric-mem.md): Memory operations specification
- [spec-fabric-error.md](./spec-fabric-error.md): Error code definitions
- [spec-loom.md](./spec-loom.md): Loom full pipeline overview
- [spec-mapper.md](./spec-mapper.md): Mapper top-level specification
- [spec-cli.md](./spec-cli.md): Loom CLI specification (includes --as-clang)
- [spec-adg-api.md](./spec-adg-api.md): ADGBuilder API reference
- [spec-adg-tools.md](./spec-adg-tools.md): Simulation tools and waveform formats
- [spec-adg-sysc.md](./spec-adg-sysc.md): SystemC generation specification
- [spec-adg-sv.md](./spec-adg-sv.md): SystemVerilog generation specification
