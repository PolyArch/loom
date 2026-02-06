# ADGBuilder API Reference

## Overview

This document provides the complete API reference for the `ADGBuilder` class.
All ADG construction uses builder methods exclusively. Direct instantiation of
fabric operation objects is not supported.

For design rationale and usage examples, see [spec-adg.md](./spec-adg.md).
For software-to-hardware place-and-route semantics, see
[spec-mapper.md](./spec-mapper.md).

## Namespace and Headers

```cpp
#include <loom/adg.h>

using namespace loom::adg;
```

The public API is contained in `<loom/adg.h>`. Implementation details are in
internal headers under `include/loom/Dialect/Fabric/`.

## Class: ADGBuilder

### Constructor

```cpp
ADGBuilder(const std::string& moduleName);
```

Creates a new builder for constructing an ADG with the specified module name.

**Parameters:**
- `moduleName`: Symbol name for the generated `fabric.module`

**Constraints:**
- Module name must be a valid MLIR symbol (alphanumeric, underscores, no leading digit)

**Example:**
```cpp
ADGBuilder builder("my_cgra_4x4");
```

## Module Creation Methods

These methods create hardware module definitions. Each returns a handle that
can be used for instantiation or topology construction.

### newPE

```cpp
PEHandle newPE(const std::string& name);
```

Creates a new `fabric.pe` definition with the given symbol name.

**Parameters:**
- `name`: Symbol name for the PE (becomes `@name` in MLIR output)

**Returns:** `PEHandle` for further configuration

**Note:** The builder always creates named PEs (using `@name` syntax in MLIR).
This allows PEs to be referenced and instantiated via `clone()`.

**Chainable methods on PEHandle:**

| Method | Description |
|--------|-------------|
| `setLatency(min, typical, max)` | Set latency attribute [min, typical, max] |
| `setInterval(min, typical, max)` | Set initiation interval [min, typical, max] |
| `addInputPort(type)` | Add an input port with specified type |
| `addOutputPort(type)` | Add an output port with specified type |
| `setInputPorts(types)` | Set all input ports at once |
| `setOutputPorts(types)` | Set all output ports at once |
| `setInterfaceCategory(category)` | Set Native or Tagged category |
| `addOp(opName)` | Syntactic sugar for single-operation PE body (see below) |
| `setBodyMLIR(mlirString)` | Define PE body using inline Fabric MLIR (see below) |

#### PE Body Definition

A `fabric.pe` body contains a compute graph built from allowed operations. The
body is defined using inline Fabric MLIR syntax via `setBodyMLIR()`. This
provides full control over the PE's internal structure.

**Allowed operations and body constraints:**

See [spec-fabric-pe-ops.md](./spec-fabric-pe-ops.md) for the complete list of
allowed operations.

See [spec-fabric-pe.md](./spec-fabric-pe.md) for body constraints including:

- Homogeneous consumption rule (full-consume vs partial-consume groups)
- Load/store exclusivity (use `newLoadPE()`/`newStorePE()` instead)
- Constant exclusivity (use `newConstantPE()` instead)
- Dataflow exclusivity
- Instance-only prohibition

Special case: a PE whose body is exactly one `dataflow.stream` has a runtime
configuration field `cont_cond_sel` (5-bit one-hot for `<`, `<=`, `>`, `>=`,
`!=`) in `config_mem`. Other dataflow-only PE bodies (`dataflow.carry`,
`dataflow.invariant`, `dataflow.gate`) have no dataflow-specific runtime
configuration.

**Example with inline MLIR body:**
```cpp
auto pe = builder.newPE("adder_pe")
    .setLatency(1, 1, 1)
    .setInterval(1, 1, 1)
    .setInputPorts({Type::i32(), Type::i32()})
    .setOutputPorts({Type::i32()})
    .setBodyMLIR(R"(
        ^bb0(%a: i32, %b: i32):
            %sum = arith.addi %a, %b : i32
            fabric.yield %sum : i32
    )");
```

**Example with complex body (select + compare):**
```cpp
auto pe = builder.newPE("max_pe")
    .setLatency(1, 1, 1)
    .setInterval(1, 1, 1)
    .setInputPorts({Type::i32(), Type::i32()})
    .setOutputPorts({Type::i32()})
    .setBodyMLIR(R"(
        ^bb0(%a: i32, %b: i32):
            %cmp = arith.cmpi sgt, %a, %b : i32
            %max = arith.select %cmp, %a, %b : i32
            fabric.yield %max : i32
    )");
```

**Example with addOp (single-operation shorthand):**
```cpp
// Equivalent to setBodyMLIR with a single arith.addi
auto pe = builder.newPE("simple_adder")
    .setLatency(1, 1, 1)
    .setInterval(1, 1, 1)
    .setInputPorts({Type::i32(), Type::i32()})
    .setOutputPorts({Type::i32()})
    .addOp("arith.addi");
```

The `addOp` method is syntactic sugar for PEs with exactly one operation. It
automatically generates the MLIR body with appropriate block arguments and
`fabric.yield`. For PEs with multiple operations, use `setBodyMLIR`.

### newConstantPE

```cpp
ConstantPEHandle newConstantPE(const std::string& name);
```

Creates a `fabric.pe` containing a single `handshake.constant` operation. This
is a special PE type because the constant value is **runtime configurable**
and must be included in config_mem.

**Chainable methods on ConstantPEHandle:**

| Method | Description |
|--------|-------------|
| `setLatency(min, typical, max)` | Set latency attribute |
| `setInterval(min, typical, max)` | Set initiation interval |
| `setOutputType(type)` | Set the constant output type (native or tagged) |

ConstantPEHandle always defines exactly one output port.

For tagged constant PEs, use `Type::tagged(valueType, tagType)` as the output type.

**Config bits:**

Constant PE config width and field packing are defined authoritatively in
[spec-fabric-config_mem.md](./spec-fabric-config_mem.md).

**Example (native):**
```cpp
auto const_pe = builder.newConstantPE("const_42")
    .setLatency(0, 0, 0)
    .setInterval(1, 1, 1)
    .setOutputType(Type::i32());
```

**Example (tagged):**
```cpp
auto const_pe = builder.newConstantPE("tagged_const")
    .setLatency(0, 0, 0)
    .setInterval(1, 1, 1)
    .setOutputType(Type::tagged(Type::f32(), Type::iN(4)));  // f32 value with 4-bit tag
```

### newLoadPE / newStorePE

```cpp
LoadPEHandle newLoadPE(const std::string& name);
StorePEHandle newStorePE(const std::string& name);
```

Creates load/store PEs with strict body constraints (exactly one
`handshake.load` or `handshake.store` operation).

**Chainable methods on LoadPEHandle:**

| Method | Description |
|--------|-------------|
| `setDataType(type)` | Set the data element type |
| `setInterfaceCategory(category)` | Set Native or Tagged category |
| `setTagWidth(width)` | Set tag width (required if Tagged) |
| `setQueueDepth(depth)` | Set load queue depth (lqDepth, TagTransparent only) |
| `setHardwareType(type)` | Set hardware type: TagOverwrite or TagTransparent |

**Chainable methods on StorePEHandle:**

| Method | Description |
|--------|-------------|
| `setDataType(type)` | Set the data element type |
| `setInterfaceCategory(category)` | Set Native or Tagged category |
| `setTagWidth(width)` | Set tag width (required if Tagged) |
| `setQueueDepth(depth)` | Set store queue depth (sqDepth, TagTransparent only) |
| `setHardwareType(type)` | Set hardware type: TagOverwrite or TagTransparent |

**HardwareType enum:**

| Value | Description |
|-------|-------------|
| `HardwareType::TagOverwrite` | Output tags are overwritten with configured values (TagOverwrite) |
| `HardwareType::TagTransparent` | Tags are preserved and forwarded unchanged (TagTransparent) |

See [spec-fabric-pe.md](./spec-fabric-pe.md) for detailed hardware type semantics.
Load/store PE latency and interval are fixed hardware behavior and are not
configurable through `LoadPEHandle` or `StorePEHandle`.

**Example (TagTransparent load PE):**
```cpp
auto load_pe = builder.newLoadPE("mem_load")
    .setDataType(Type::i32())
    .setInterfaceCategory(InterfaceCategory::Tagged)
    .setTagWidth(4)
    .setQueueDepth(8)
    .setHardwareType(HardwareType::TagTransparent);
```

See [spec-fabric-pe.md](./spec-fabric-pe.md) for detailed load/store PE
semantics including hardware types and port roles.

### newTemporalPE

```cpp
TemporalPEHandle newTemporalPE(const std::string& name);
```

Creates a new `fabric.temporal_pe` definition.

**Parameters:**
- `name`: Symbol name for the temporal PE

**Returns:** `TemporalPEHandle` for further configuration

**Chainable methods on TemporalPEHandle:**

| Method | Description |
|--------|-------------|
| `setNumRegisters(n)` | Set number of internal registers |
| `setNumInstructions(n)` | Set maximum instruction slots |
| `setRegFifoDepth(n)` | Set FIFO depth for registers |
| `setInterface(taggedType)` | Set interface type (must be tagged) |
| `addFU(peHandle)` | Add a functional unit type |
| `enableShareOperandBuffer(size)` | Enable Mode B with specified buffer size |

**Operand Buffer Modes:**

By default, temporal PE uses Mode A (per-instruction operand buffer). Call
`enableShareOperandBuffer(size)` to switch to Mode B (shared buffer with
per-tag FIFO semantics).

- Mode A: Each instruction has dedicated operand buffer slots
- Mode B: Shared buffer prevents head-of-line blocking between tags

See [spec-fabric-temporal_pe.md](./spec-fabric-temporal_pe.md) for detailed
operand buffer architecture.

**Constraints:**
- All FUs must have matching input/output counts
- FU interfaces must be native (not tagged)
- FU value types must match the temporal PE interface value type
- At least one FU must be added
- Mode B `size` must be in range [1, 8192]

**Example (Mode A, default):**
```cpp
auto temporal_pe = builder.newTemporalPE("mux_pe")
    .setNumRegisters(4)
    .setNumInstructions(16)
    .setRegFifoDepth(2)
    .setInterface(Type::tagged(Type::i32(), Type::iN(4)))
    .addFU(alu_pe)
    .addFU(mul_pe);
```

**Example (Mode B, shared operand buffer):**
```cpp
auto temporal_pe = builder.newTemporalPE("mux_pe_shared")
    .setNumRegisters(4)
    .setNumInstructions(16)
    .setRegFifoDepth(2)
    .setInterface(Type::tagged(Type::i32(), Type::iN(4)))
    .enableShareOperandBuffer(64)  // 64-entry shared buffer
    .addFU(alu_pe)
    .addFU(mul_pe);
```

### newSwitch

```cpp
SwitchHandle newSwitch(const std::string& name);
```

Creates a new `fabric.switch` definition.

**Parameters:**
- `name`: Symbol name for the switch

**Returns:** `SwitchHandle` for further configuration

**Chainable methods on SwitchHandle:**

| Method | Description |
|--------|-------------|
| `setPortCount(inputs, outputs)` | Set number of input and output ports |
| `setConnectivity(table)` | Set physical connectivity (default: full crossbar) |
| `setType(type)` | Set port type (native or tagged) |

**Connectivity table format:**
`std::vector<std::vector<bool>>` with dimensions `[num_outputs][num_inputs]`.
`table[o][i] = true` means input `i` can physically route to output `o`.
This is output-major ordering, consistent with the Fabric specification
(see [spec-fabric-switch.md](./spec-fabric-switch.md)).

**Constraints:**
- Maximum 32 inputs and 32 outputs
- Each row must have at least one `true` (every output must have at least one source)
- Each column must have at least one `true` (every input must route to at least one output)

**Example (full crossbar):**
```cpp
auto sw = builder.newSwitch("router_4x4")
    .setPortCount(4, 4)
    .setType(Type::i32());
// Default is full crossbar: all inputs can route to all outputs
```

**Example (ring topology):**
```cpp
// 4-input, 4-output ring: each output receives from same-index and next-index input
std::vector<std::vector<bool>> ring = {
    {true, true, false, false},  // out0 <- in0, in1
    {false, true, true, false},  // out1 <- in1, in2
    {false, false, true, true},  // out2 <- in2, in3
    {true, false, false, true}   // out3 <- in3, in0 (wrap)
};
auto sw = builder.newSwitch("ring_4x4")
    .setPortCount(4, 4)
    .setConnectivity(ring)
    .setType(Type::i32());
```

### newTemporalSwitch

```cpp
TemporalSwitchHandle newTemporalSwitch(const std::string& name);
```

Creates a new `fabric.temporal_sw` definition.

**Parameters:**
- `name`: Symbol name for the temporal switch

**Returns:** `TemporalSwitchHandle` for further configuration

**Chainable methods on TemporalSwitchHandle:**

| Method | Description |
|--------|-------------|
| `setNumRouteTable(n)` | Set maximum route table slots |
| `setPortCount(inputs, outputs)` | Set number of input and output ports |
| `setConnectivity(table)` | Set physical connectivity |
| `setInterface(taggedType)` | Set interface type (must be tagged) |

**Example:**
```cpp
auto tsw = builder.newTemporalSwitch("temporal_router")
    .setNumRouteTable(8)
    .setPortCount(4, 4)
    .setInterface(Type::tagged(Type::i32(), Type::iN(4)));
```

### newMemory

```cpp
MemoryHandle newMemory(const std::string& name);
```

Creates a new `fabric.memory` definition (on-chip scratchpad).

**Parameters:**
- `name`: Symbol name for the memory

**Returns:** `MemoryHandle` for further configuration

**Chainable methods on MemoryHandle:**

| Method | Description |
|--------|-------------|
| `setLoadPorts(count)` | Set number of load ports |
| `setStorePorts(count)` | Set number of store ports |
| `setQueueDepth(depth)` | Set LSQ depth (`lsqDepth`) |
| `setPrivate(isPrivate)` | Set whether memory is private (default: true) |
| `setShape(memrefType)` | Set memory shape and element type |

**Interface type rules:**

Port counts affect interface types. When `ldCount > 1` or `stCount > 1`, the
corresponding address and data ports must be tagged types. Single-port
interfaces (`count == 1`) use native types. Changing from single to multi-port
is a breaking interface change.

See [spec-fabric-mem.md](./spec-fabric-mem.md) for complete tagging rules and
type constraints.

**Example:**
```cpp
auto mem = builder.newMemory("scratchpad")
    .setLoadPorts(2)
    .setStorePorts(2)
    .setQueueDepth(4)
    .setShape(MemrefType::static1D(1024, Type::i32()));
```

### newExtMemory

```cpp
ExtMemoryHandle newExtMemory(const std::string& name);
```

Creates a new `fabric.extmemory` definition (external memory interface).

**Parameters:**
- `name`: Symbol name for the external memory interface

**Returns:** `ExtMemoryHandle` for further configuration

**Chainable methods on ExtMemoryHandle:**

| Method | Description |
|--------|-------------|
| `setLoadPorts(count)` | Set number of load ports |
| `setStorePorts(count)` | Set number of store ports |
| `setQueueDepth(depth)` | Set LSQ depth (`lsqDepth`) |
| `setShape(memrefType)` | Set memory shape and element type |

`setPrivate` is not available for `fabric.extmemory`.

### API-to-MLIR Attribute Mapping

The builder API uses method names for ergonomics. Fabric MLIR attribute names
remain authoritative in Fabric specs.

| API Method | Handle Type | Fabric MLIR Attribute |
|-----------|-------------|------------------------|
| `setQueueDepth(depth)` | `LoadPEHandle` | `lqDepth` |
| `setQueueDepth(depth)` | `StorePEHandle` | `sqDepth` |
| `setQueueDepth(depth)` | `MemoryHandle` | `lsqDepth` |
| `setQueueDepth(depth)` | `ExtMemoryHandle` | `lsqDepth` |
| `setNumRegisters(n)` | `TemporalPEHandle` | `num_register` |
| `setNumInstructions(n)` | `TemporalPEHandle` | `num_instruction` |

Naming convention note:

- Fabric MLIR attributes use singular count names (for example,
  `num_register`, `num_instruction`).
- API and backend templates use plural count names where appropriate (for
  example, `setNumRegisters`, `NUM_REGISTERS`).

**Interface type rules:**

Port counts affect interface types exactly as in `MemoryHandle`. When
`ldCount > 1` or `stCount > 1`, the corresponding address and data ports must
be tagged types. Single-port interfaces (`count == 1`) use native types.

## Tag Operation Methods

### newAddTag

```cpp
AddTagHandle newAddTag(const std::string& name);
```

Creates a new `fabric.add_tag` operation.

**Chainable methods:**

| Method | Description |
|--------|-------------|
| `setValueType(type)` | Set input value type |
| `setTagType(type)` | Set output tag type |

**Constraints:**
- Input must be native type (not tagged)
- Tag width must be in range [1, 16] bits

Note: The `tag` value itself is a runtime configuration parameter and is not
set at ADG construction time.

### newMapTag

```cpp
MapTagHandle newMapTag(const std::string& name);
```

Creates a new `fabric.map_tag` operation.

**Chainable methods:**

| Method | Description |
|--------|-------------|
| `setValueType(type)` | Set value type (preserved) |
| `setInputTagType(type)` | Set input tag type |
| `setOutputTagType(type)` | Set output tag type |
| `setTableSize(size)` | Set hardware table size |

**Constraints:**
- Input must be tagged type
- `table_size` must be in range [1, 256]
- Violations: `COMP_MAP_TAG_TABLE_SIZE`

### newDelTag

```cpp
DelTagHandle newDelTag(const std::string& name);
```

Creates a new `fabric.del_tag` operation.

**Chainable methods:**

| Method | Description |
|--------|-------------|
| `setInputType(type)` | Set input tagged type |

**Constraints:**
- Input must be tagged type (not native)

## Instantiation and Connection Methods

### clone

```cpp
InstanceHandle clone(ModuleHandle source, const std::string& instanceName);
```

Creates a `fabric.instance` referencing an existing module definition.

**Parameters:**
- `source`: Handle to PE, Switch, TemporalPE, TemporalSwitch, Memory, or ExtMemory
- `instanceName`: Unique instance name within the module

**Returns:** `InstanceHandle` for connection and further modification

**Semantics:**
- Creates hardware instantiation initially referencing the source definition
- Instance does NOT copy any runtime configuration from definition
- Instance can be further modified (see Modify-on-Clone below)

**Modify-on-Clone semantics:**

When you modify a cloned instance's hardware attributes (latency, interval,
ports, etc.), the builder **forks a new template** automatically:

1. The instance is decoupled from the original definition
2. A new anonymous module definition is created with the modified attributes
3. The instance's `fabric.instance` now references this new definition
4. Other instances cloned from the same source remain unchanged

This enables creating variations without explicitly defining multiple templates.

**Example (fork on modify):**
```cpp
auto pe = builder.newPE("alu")
    .setLatency(1, 1, 1);

auto pe_fast = builder.clone(pe, "pe_fast");
auto pe_slow = builder.clone(pe, "pe_slow");

// Modifying pe_slow forks a new template; pe_fast still uses original
pe_slow.setLatency(2, 2, 2);

// Now: pe_fast -> @alu (latency 1,1,1)
//      pe_slow -> @alu_1 (latency 2,2,2, auto-generated name)
```

**config_mem allocation:**
Each instance gets a **separate config_mem address region**. If a PE requires
32 config bits and you create 4 instances, the total config_mem usage is
4 x 32 = 128 bits. Addresses are allocated sequentially in MLIR operation
order at export time.
For the formal config memory definition (fixed 32-bit words, depth
calculation, and alignment rules), see
[spec-fabric-config_mem.md](./spec-fabric-config_mem.md).

**Example:**
```cpp
// Define PE once
auto pe = builder.newPE("alu");

// Create 4 instances - each gets separate config_mem allocation
auto pe_00 = builder.clone(pe, "pe_0_0");  // config_mem[0:31]
auto pe_01 = builder.clone(pe, "pe_0_1");  // config_mem[32:63]
auto pe_10 = builder.clone(pe, "pe_1_0");  // config_mem[64:95]
auto pe_11 = builder.clone(pe, "pe_1_1");  // config_mem[96:127]
```

### connect

```cpp
void connect(Handle src, Handle dst);
```

Creates a connection from source to destination using default port indices.

**Parameters:**
- `src`: Source module/instance handle
- `dst`: Destination module/instance handle

**Behavior:**
- Single-output to single-input: direct connection
- Multi-output or multi-input: connects port 0 to port 0

For explicit port control, use `connectPorts`.

### connectPorts

```cpp
void connectPorts(Handle src, int srcPort, Handle dst, int dstPort);
```

Creates a connection between specific ports.

**Parameters:**
- `src`: Source module/instance handle
- `srcPort`: Output port index on source
- `dst`: Destination module/instance handle
- `dstPort`: Input port index on destination

**Constraints:**
- Port indices must be valid for the respective modules
- Port types must match exactly (enforced at validation)

**Example:**
```cpp
builder.connectPorts(pe_00, 0, sw_00, 2);
builder.connectPorts(sw_00, 1, pe_01, 0);
```

## Topology Helper Methods

### buildMesh

```cpp
MeshResult buildMesh(
    int rows,
    int cols,
    ModuleHandle peTemplate,
    ModuleHandle swTemplate,
    Topology topology
);
```

Constructs a regular grid of PEs and switches.

**Parameters:**
- `rows`: Number of rows
- `cols`: Number of columns
- `peTemplate`: PE module to instantiate at each grid point
- `swTemplate`: Switch module to instantiate for routing
- `topology`: Connectivity pattern

**Returns:** `MeshResult` containing:
- `peGrid`: 2D array of PE instance handles
- `swGrid`: 2D array of switch instance handles

**Topology options:**

| Value | Description |
|-------|-------------|
| `Topology::Mesh` | Nearest-neighbor, no wraparound |
| `Topology::Torus` | Nearest-neighbor with wraparound |
| `Topology::DiagonalMesh` | Mesh plus diagonal neighbors |
| `Topology::DiagonalTorus` | Torus plus diagonal neighbors |

**Grid layout:**

For a 2x2 mesh with `Topology::Mesh`:

```
[0,0] PE[0,0]                [0,2] PE[0,1]

        [1,1] SW[0,0]                [1,3] SW[0,1]

[2,0] PE[1,0]                [2,2] PE[1,1]

        [3,1] SW[1,0]                [3,3] SW[1,1]
```

`buildMesh` uses a single coordinate system with no overlapping modules:

- `peGrid[r][c]` maps to coordinate `(2*r, 2*c)`
- `swGrid[r][c]` maps to coordinate `(2*r + 1, 2*c + 1)`

Instance names encode module type and logical grid index
(e.g., `pe_0_0`, `sw_0_0`).

**Switch port ordering:**

Switches in the mesh use N/E/S/W (North, East, South, West) port ordering:
- Port 0: North (row - 1)
- Port 1: East (col + 1)
- Port 2: South (row + 1)
- Port 3: West (col - 1)

This ordering applies to both input and output ports on each switch.

**Example:**
```cpp
auto result = builder.buildMesh(4, 4, pe, sw, Topology::Torus);
auto pe_center = result.peGrid[2][2];
```

## Module Port Methods

These methods configure the top-level `fabric.module` interface.

### addModuleInput

```cpp
PortHandle addModuleInput(const std::string& name, Type type);
PortHandle addModuleInput(const std::string& name, MemrefType memrefType);
```

Adds an input port to the module interface.

**Parameters:**
- `name`: Port name
- `type`: Streaming port type (native or tagged) - maps to AXI-Stream interface
- `memrefType`: Memory port type - maps to AXI-MM interface

**Returns:** `PortHandle` for connection

**Hardware interface mapping:**
- `Type` (native/tagged): AXI-Stream with valid/ready/data signals
- `MemrefType`: AXI-MM (memory-mapped) for external memory access

**Note:** Ports are automatically reordered to satisfy the required ordering
(memref*, native*, tagged*) at export time.

**Example:**
```cpp
// Streaming input (AXI-Stream)
auto stream_in = builder.addModuleInput("data_in", Type::i32());

// Tagged streaming input (AXI-Stream)
auto tagged_in = builder.addModuleInput("tagged_in",
    Type::tagged(Type::i32(), Type::iN(4)));

// Memory input (AXI-MM)
auto mem_in = builder.addModuleInput("mem",
    MemrefType::static1D(1024, Type::i32()));
```

### addModuleOutput

```cpp
PortHandle addModuleOutput(const std::string& name, Type type);
PortHandle addModuleOutput(const std::string& name, MemrefType memrefType);
```

Adds an output port to the module interface. Same parameter semantics as
`addModuleInput`.

### connectToModuleInput

```cpp
void connectToModuleInput(PortHandle port, Handle dst, int dstPort);
```

Connects a module input port to an internal component.

### connectToModuleOutput

```cpp
void connectToModuleOutput(Handle src, int srcPort, PortHandle port);
```

Connects an internal component to a module output port.

## Validation

### validateADG

```cpp
ValidationResult validateADG();
```

Performs comprehensive validation of the constructed ADG.

**Returns:** `ValidationResult` containing:
- `success`: Boolean indicating overall success
- `errors`: Vector of `ValidationError` objects

**Validation checks:**

1. **Connectivity completeness**
   - All input ports have exactly one driver
   - All output ports are connected to at least one destination
   - No dangling wires

2. **Type matching**
   - Connected ports have identical types
   - Bit widths match exactly
   - Tagged types have matching tag widths

3. **Port ordering**
   - Module interface follows memref*, native*, tagged* order
   - Builder handles reordering automatically

4. **Instance resolution**
   - All `fabric.instance` references resolve to defined modules
   - Referenced modules exist in the current builder context

5. **Parameter bounds**
   - Latency: min <= typical <= max, min >= 0
   - Interval: min <= typical <= max, min >= 1
   - Switch ports: <= 32
   - Tag width: 1 to 16 bits

6. **Resource constraints**
   - Memory port counts valid (ldCount + stCount > 0)
   - LSQ depth requirements met

7. **Template deduplication**
   - Multiple module definitions with identical hardware structure are merged
   - Instances pointing to equivalent templates are updated to share one definition
   - See "Hardware Equivalence" below for equivalence rules

**Hardware Equivalence:**

Two module definitions are considered **hardware equivalent** if they produce
identical physical implementations. The following attributes affect equivalence:

- All hardware parameters (latency, interval, port counts, connectivity, etc.)
- PE body operations and their configurations
- FU types and counts (for temporal_pe)

The following attributes do **NOT** affect hardware equivalence:

- Module symbol name (e.g., `@alu` vs `@alu_1`)
- Instance names
- Port indices (unless used for physical routing decisions)
- Any metadata or debugging attributes

During validation, the builder performs deduplication:

1. Compute a hardware signature for each module definition
2. Group definitions with identical signatures
3. Select one canonical definition per group
4. Update all `fabric.instance` references to use canonical definitions
5. Remove redundant definitions from the output

This ensures minimal RTL output even when many forked templates exist.

**Error structure:**

```cpp
struct ValidationError {
    std::string code;      // e.g., "COMP_SWITCH_PORT_LIMIT"
    std::string message;   // Human-readable description
    std::string location;  // Format: "module_name::operation_name"
};
```

**Example:**
```cpp
auto result = builder.validateADG();
if (!result.success) {
    for (const auto& err : result.errors) {
        std::cerr << err.code << ": " << err.message << "\n";
    }
    return 1;
}
```

## Export Methods

### exportMLIR

```cpp
void exportMLIR(const std::string& path);
```

Exports the ADG as Fabric MLIR.

**Parameters:**
- `path`: Output file path

**Behavior:**
- Calls `validateADG()` internally; throws on validation failure
- Creates parent directories if needed
- Overwrites existing file

**Output format:**
- Valid MLIR syntax
- Single `fabric.module` definition
- All hardware parameters specified
- Runtime configuration at default (empty) values

### exportDOT

```cpp
void exportDOT(const std::string& path, DOTMode mode = DOTMode::Structure);
```

Exports the ADG as Graphviz DOT format.

**Parameters:**
- `path`: Output file path
- `mode`: Visualization detail level

**DOTMode options:**

| Mode | Description |
|------|-------------|
| `DOTMode::Structure` | Hardware modules and connections only |
| `DOTMode::Detailed` | Includes internal structure and config info |

For visual conventions (node styles, edge styles, unmapped elements), see
[spec-viz-hw.md](./spec-viz-hw.md).

### exportSV

```cpp
void exportSV(const std::string& directory);
```

Exports the ADG as synthesizable SystemVerilog.

**Parameters:**
- `directory`: Output directory path

**Output files (self-contained):**
- `<moduleName>_top.sv`: Top-level module with instantiations
- `<moduleName>_config.sv`: config_mem controller
- `<moduleName>_addr.h`: C header with address definitions
- `lib/`: Parameterized library modules (copied, no external dependencies)
  - `fabric_pe.sv`, `fabric_temporal_pe.sv`, `fabric_switch.sv`, etc.

Self-contained output is guaranteed by [spec-adg.md](./spec-adg.md).

See [spec-adg-sv.md](./spec-adg-sv.md) for complete specification.

### exportSysC

```cpp
void exportSysC(const std::string& directory);
```

Exports the ADG as a SystemC simulation model.

**Parameters:**
- `directory`: Output directory path

**Output files (self-contained):**
- `<moduleName>_top.h/cpp`: Top-level module
- `<moduleName>_config.h/cpp`: config_mem controller with TLM interface
- `<moduleName>_testbench.h/cpp`: Example testbench
- `<moduleName>_main.cpp`: Example main() for standalone simulation
- `<moduleName>_addr.h`: C header with address definitions (shared with SV)
- `CMakeLists.txt`: CMake build configuration
- `lib/`: Parameterized library modules (copied, no external dependencies)
  - `fabric_pe.h`, `fabric_temporal_pe.h`, `fabric_switch.h`, etc.

Self-contained output is guaranteed by [spec-adg.md](./spec-adg.md).

**Abstraction levels:**

| Mode | Macro | Description |
|------|-------|-------------|
| Cycle-Accurate | `FABRIC_SYSC_CYCLE_ACCURATE` | Default, exact timing |
| Loosely-Timed | `FABRIC_SYSC_LOOSELY_TIMED` | Fast simulation |

**Requirements:**
- SystemC 3.0.1
- C++17 compiler

See [spec-adg-sysc.md](./spec-adg-sysc.md) for complete specification.

## Type System

### Type Class

The `Type` class represents MLIR types in the ADG API.

**Factory methods:**

| Method | Description |
|--------|-------------|
| `Type::i1()` | 1-bit integer |
| `Type::i8()` | 8-bit integer |
| `Type::i16()` | 16-bit integer |
| `Type::i32()` | 32-bit integer |
| `Type::i64()` | 64-bit integer |
| `Type::iN(n)` | N-bit integer (primarily for tag types; see note below) |
| `Type::bf16()` | Brain floating-point (16-bit) |
| `Type::f16()` | IEEE 16-bit float |
| `Type::f32()` | 32-bit float |
| `Type::f64()` | 64-bit float |
| `Type::index()` | Index type |
| `Type::none()` | None type (for control tokens) |
| `Type::tagged(value, tag)` | Tagged type |

**Validation note:** `Type::iN(n)` creates an arbitrary-width integer type,
but its primary use is for **tag types** (range `i1` to `i16`). For value
types (data payload), only the fixed set of widths is allowed: `i1`, `i8`,
`i16`, `i32`, `i64`. Using `Type::iN(n)` with an unsupported value width
(e.g., `Type::iN(3)` as a data type) will be rejected during validation.
See [spec-dataflow.md](./spec-dataflow.md) for the complete list of allowed
value types.

**Example:**
```cpp
auto i32 = Type::i32();
auto tagged_i32 = Type::tagged(Type::i32(), Type::iN(4));
```

### MemrefType Class

The `MemrefType` class represents memref types.

**Factory methods:**

| Method | Description |
|--------|-------------|
| `MemrefType::static1D(size, elemType)` | 1D static memref |
| `MemrefType::dynamic1D(elemType)` | 1D dynamic memref |

### Raw MLIR Injection

For advanced use cases requiring total control, users can inject raw Fabric
MLIR directly into the ADG.

#### injectMLIR

```cpp
void injectMLIR(const std::string& mlirString);
```

Injects raw Fabric MLIR operations into the current module context. The MLIR
string is parsed and validated before insertion.

**Use cases:**
- Custom PE definitions not expressible through builder API
- Complex nested structures
- Direct copy-paste from MLIR files

**Example:**
```cpp
builder.injectMLIR(R"(
    fabric.pe @custom_fma(%a: f32, %b: f32, %c: f32) -> (f32)
        [latency = [3 : i16, 3 : i16, 3 : i16],
         interval = [1 : i16, 1 : i16, 1 : i16]] {
        %result = math.fma %a, %b, %c : f32
        fabric.yield %result : f32
    }
)");

// Get handle to injected PE by name and create instance
auto custom_fma = builder.getPEByName("custom_fma");
auto fma_inst = builder.clone(custom_fma, "fma_0");
```

#### Named Lookup Methods

```cpp
PEHandle getPEByName(const std::string& name);
SwitchHandle getSwitchByName(const std::string& name);
TemporalPEHandle getTemporalPEByName(const std::string& name);
TemporalSwitchHandle getTemporalSwitchByName(const std::string& name);
MemoryHandle getMemoryByName(const std::string& name);
ExtMemoryHandle getExtMemoryByName(const std::string& name);
```

Returns a typed handle to an existing named module definition.

**Parameters:**
- `name`: Symbol name of the definition (without `@`).

**Behavior:**
- On success, returns the corresponding handle type.
- If `name` does not resolve to that module category, error handling follows
  the current builder error mode (`ErrorMode::Throw` or `ErrorMode::Collect`).

These lookups are typically used after `injectMLIR()` or when definitions are
created in helper code and retrieved later by symbol name.

#### getMLIRContext

```cpp
mlir::MLIRContext* getMLIRContext();
```

Returns the underlying MLIR context for advanced manipulation. Use with
caution; direct context manipulation bypasses builder validation.

### Supported Operations Reference

For the complete and authoritative `fabric.pe` operation allowlist, see
[spec-fabric-pe-ops.md](./spec-fabric-pe-ops.md).

For body constraints (exclusivity rules, homogeneous consumption rule), see
[spec-fabric-pe.md](./spec-fabric-pe.md).

## Error Handling

By default, builder methods throw exceptions on errors. Alternative error
handling modes can be configured.

### setErrorMode

```cpp
void setErrorMode(ErrorMode mode);
```

**ErrorMode options:**

| Mode | Behavior |
|------|----------|
| `ErrorMode::Throw` | Throw exceptions (default) |
| `ErrorMode::Collect` | Collect errors; retrieve with `getErrors()` |

### getErrors

```cpp
std::vector<BuilderError> getErrors();
```

Returns collected errors when using `ErrorMode::Collect`.

## Thread Safety

`ADGBuilder` is not thread-safe. Each thread should use its own builder
instance. Multiple builders can operate concurrently on different ADGs.

## Related Documents

- [spec-loom.md](./spec-loom.md)
- [spec-adg.md](./spec-adg.md)
- [spec-adg-sv.md](./spec-adg-sv.md)
- [spec-adg-sysc.md](./spec-adg-sysc.md)
- [spec-adg-tools.md](./spec-adg-tools.md)
- [spec-fabric.md](./spec-fabric.md)
- [spec-fabric-config_mem.md](./spec-fabric-config_mem.md)
- [spec-fabric-pe.md](./spec-fabric-pe.md)
- [spec-fabric-pe-ops.md](./spec-fabric-pe-ops.md)
- [spec-mapper.md](./spec-mapper.md)
