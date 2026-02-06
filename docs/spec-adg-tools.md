# ADG Simulation Tools Specification

## Overview

This document specifies the preferred simulation tools, waveform formats, and
workflows for ADG-generated hardware models. Both SystemC and SystemVerilog
outputs follow these conventions.

For SystemC generation, see [spec-adg-sysc.md](./spec-adg-sysc.md).
For SystemVerilog generation, see [spec-adg-sv.md](./spec-adg-sv.md).

## Simulator Priority

| Priority | Simulator | Type | Notes |
|----------|-----------|------|-------|
| 1 | Synopsys VCS | Commercial | Industry-standard, best performance and debugging |
| 2 | Verilator | Open-source | Fast cycle-accurate simulation, no license required |

**Synopsys VCS** is the recommended simulator for:
- Production verification
- Coverage analysis
- Mixed SystemC/RTL co-simulation
- FSDB waveform generation for Verdi

**Verilator** is suitable for:
- Open-source development environments
- CI/CD pipelines without commercial licenses
- Quick iteration during development

## Waveform Viewer Priority

| Priority | Viewer | Type | Supported Formats |
|----------|--------|------|-------------------|
| 1 | Synopsys Verdi | Commercial | FSDB, VCD, EVCD |
| 2 | GTKWave | Open-source | FST, VCD, LXT2 |

**Synopsys Verdi** provides:
- Schematic view and signal tracing
- Transaction-level debugging
- Coverage visualization
- Source code correlation

**GTKWave** provides:
- Basic waveform viewing
- Good FST compression support
- Cross-platform compatibility

## Waveform Format Priority

| Priority | Format | Extension | Compression | Tool Support |
|----------|--------|-----------|-------------|--------------|
| 1 | FSDB | `.fsdb` | Excellent | VCS + Verdi (Synopsys license required) |
| 2 | FST | `.fst` | Good | Verilator + GTKWave (open-source) |
| 3 | VCD | `.vcd` | None | Universal (all tools) |

### FSDB (Fast Signal Database)

FSDB is the preferred format when using Synopsys tools:

- Best compression ratio (10-100x smaller than VCD)
- Fastest waveform loading in Verdi
- Supports incremental dumping
- Hierarchical signal selection

### FST (Fast Signal Trace)

FST is the preferred format for open-source workflows:

- Good compression (5-20x smaller than VCD)
- Native Verilator support
- GTKWave optimized loading
- LZ4/ZSTD compression options

### VCD (Value Change Dump)

VCD is the universal fallback:

- IEEE 1364 standard format
- Supported by all tools
- No compression (largest files)
- Slowest loading for large designs

## VCS Simulation Workflow

### SystemVerilog RTL Simulation

```bash
# Compile with FSDB support (from exported directory)
vcs -sverilog -fsdb -full64 \
    +incdir+./lib \
    my_cgra_top.sv my_cgra_config.sv lib/*.sv \
    tb_top.sv \
    -o simv

# Run simulation with FSDB dumping
./simv +fsdbfile+waves.fsdb

# View waveforms
verdi -ssf waves.fsdb &
```

### SystemC Simulation

```bash
# Compile SystemC with VCS
vcs -sysc -fsdb -full64 \
    -cpp g++ -cc gcc \
    my_cgra_top.cpp my_cgra_testbench.cpp my_cgra_main.cpp \
    -I${SYSTEMC_HOME}/include \
    -L${SYSTEMC_HOME}/lib64 -lsystemc \
    -CFLAGS "-DFABRIC_USE_FSDB" \
    -o simv

# Run simulation
./simv

# View waveforms
verdi -ssf waves.fsdb &
```

### Mixed SystemC/RTL Co-Simulation

```bash
# Compile mixed design
vcs -sysc -sverilog -fsdb -full64 \
    my_cgra_sysc.cpp \
    rtl_reference.sv \
    -I${SYSTEMC_HOME}/include \
    -L${SYSTEMC_HOME}/lib64 -lsystemc \
    -o simv

./simv +fsdbfile+waves.fsdb
```

### FSDB Dumping in Testbench

SystemVerilog:
```systemverilog
initial begin
    $fsdbDumpfile("waves.fsdb");
    $fsdbDumpvars(0, tb_top);  // Dump all signals from tb_top
end
```

SystemC:
```cpp
#ifdef FABRIC_USE_FSDB
#include "fsdb_trace.h"

void testbench::setup_trace() {
    fsdbDumpfile("waves.fsdb");
    fsdbDumpvars(0, this);
}
#endif
```

## Verilator Simulation Workflow

### SystemVerilog RTL Simulation

```bash
# Compile with FST support (from exported directory)
verilator --sv --trace-fst --trace-structs -Wno-fatal \
    +incdir+./lib \
    my_cgra_top.sv my_cgra_config.sv lib/*.sv \
    --exe tb_main.cpp \
    -o Vmy_cgra

# Build
make -C obj_dir -f Vmy_cgra.mk

# Run simulation
./obj_dir/Vmy_cgra

# View waveforms
gtkwave waves.fst &
```

### SystemC Integration

```bash
# Compile with SystemC support
verilator --sc --trace-fst -Wno-fatal \
    -CFLAGS "-I${SYSTEMC_HOME}/include" \
    -LDFLAGS "-L${SYSTEMC_HOME}/lib64 -lsystemc" \
    my_cgra_top.sv \
    --exe sc_main.cpp \
    -o Vmy_cgra

make -C obj_dir -f Vmy_cgra.mk
./obj_dir/Vmy_cgra
```

### FST Dumping in Testbench

C++ (Verilator):
```cpp
#include "verilated_fst_c.h"

int main(int argc, char** argv) {
    Verilated::traceEverOn(true);

    VerilatedFstC* tfp = new VerilatedFstC;
    top->trace(tfp, 99);  // Trace depth
    tfp->open("waves.fst");

    // Simulation loop
    while (!Verilated::gotFinish()) {
        top->eval();
        tfp->dump(sim_time++);
    }

    tfp->close();
}
```

## VCD Fallback Workflow

When neither VCS nor Verilator FST support is available:

SystemVerilog:
```systemverilog
initial begin
    $dumpfile("waves.vcd");
    $dumpvars(0, tb_top);
end
```

SystemC:
```cpp
sc_trace_file* tf = sc_create_vcd_trace_file("waves");
tf->set_time_unit(1, SC_NS);
sc_trace(tf, clk, "clk");
sc_trace(tf, rst_n, "rst_n");
// ... trace other signals
sc_close_vcd_trace_file(tf);
```

View with any tool:
```bash
gtkwave waves.vcd &
# or
verdi -ssf waves.vcd &
```

## Build System Integration

### CMake Detection

The generated CMakeLists.txt automatically detects available tools:

```cmake
# Detect VCS
find_program(VCS_EXECUTABLE vcs)
if(VCS_EXECUTABLE)
    set(FABRIC_USE_VCS ON)
    set(FABRIC_WAVEFORM_FORMAT "fsdb")
endif()

# Detect Verilator
find_program(VERILATOR_EXECUTABLE verilator)
if(VERILATOR_EXECUTABLE AND NOT FABRIC_USE_VCS)
    set(FABRIC_USE_VERILATOR ON)
    set(FABRIC_WAVEFORM_FORMAT "fst")
endif()

# Fallback to VCD
if(NOT FABRIC_WAVEFORM_FORMAT)
    set(FABRIC_WAVEFORM_FORMAT "vcd")
endif()
```

### Environment Variables

| Variable | Description | Reference Path |
|----------|-------------|----------------|
| `VCS_HOME` | Synopsys VCS installation | `/path/to/tools/synopsys/vcs/X-2025.06-SP1` |
| `VERDI_HOME` | Synopsys Verdi installation | `/path/to/tools/synopsys/verdi/X-2025.06-SP1` |
| `VERILATOR_ROOT` | Verilator installation | `/path/to/tools/verilator/5.044` |
| `SYSTEMC_HOME` | SystemC installation | `/path/to/tools/systemc/3.0.1` |

## Related Documents

- [spec-adg.md](./spec-adg.md): ADG overall design
- [spec-adg-api.md](./spec-adg-api.md): ADGBuilder API reference
- [spec-adg-sysc.md](./spec-adg-sysc.md): SystemC generation specification
- [spec-adg-sv.md](./spec-adg-sv.md): SystemVerilog generation specification
